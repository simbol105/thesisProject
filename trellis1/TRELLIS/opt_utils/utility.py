
import os
import numpy as np
import imageio
import torch
import trimesh


from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import open3d as o3d





def align_meshes_fixed_scale_ransac(pred_mesh, dtc_mesh):
    """Explicitly scales and centers the models before feature extraction"""
    source_pts, _ = trimesh.sample.sample_surface(unwrap_mesh(pred_mesh), 20000)
    target_pts, _ = trimesh.sample.sample_surface(unwrap_mesh(dtc_mesh), 20000)

    s_min, s_max = source_pts.min(axis=0), source_pts.max(axis=0)
    t_min, t_max = target_pts.min(axis=0), target_pts.max(axis=0)

    diag_source = np.linalg.norm(s_max - s_min)
    diag_target = np.linalg.norm(t_max - t_min)
    
    scale_factor = diag_target / diag_source

    c_source = (s_max + s_min) / 2.0
    c_target = (t_max + t_min) / 2.0

    source_normalized_pts = (source_pts - c_source) * scale_factor + c_target

    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_normalized_pts)
    
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_pts)

    voxel_size = diag_target * 0.03 

    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)
    
    source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))

    distance_threshold = voxel_size * 1.5
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, False, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, voxel_size * 0.5, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())


    T_norm = np.eye(4)
    T_norm[:3, :3] *= scale_factor
    T_norm[:3, 3] = c_target - scale_factor * c_source

    T_rigid = result_icp.transformation
    
    final_transform = T_rigid @ T_norm
    return final_transform




def align_meshes(dtc_mesh, pred_mesh, n_samples=10000):
    """Align pred_mesh to dtc_mesh using:"""

    dtc = unwrap_mesh(dtc_mesh)
    pred = unwrap_mesh(pred_mesh)

   
    dtc_pts, _ = trimesh.sample.sample_surface(dtc, n_samples)
    pred_pts, _ = trimesh.sample.sample_surface(pred, n_samples)


    c_dtc = dtc.bounds.mean(axis=0)
    c_pred = pred.bounds.mean(axis=0)

    
    diag_dtc = np.linalg.norm(dtc.bounds[1] - dtc.bounds[0])
    diag_pred = np.linalg.norm(pred.bounds[1] - pred.bounds[0])
    scale = diag_dtc / diag_pred


    dtc_centered = dtc_pts - dtc_pts.mean(axis=0)
    pred_centered = pred_pts - pred_pts.mean(axis=0)

    _, _, Vt_dtc = np.linalg.svd(dtc_centered, full_matrices=False)
    _, _, Vt_pred = np.linalg.svd(pred_centered, full_matrices=False)

    R_dtc = Vt_dtc.T
    R_pred = Vt_pred.T


    R = R_dtc @ R_pred.T

    if np.linalg.det(R) < 0:
        R[:, -1] *= -1


    t = c_dtc - scale * (R @ c_pred)


    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t

    return T



def align_meshes_similarity(
    pred_mesh,
    dtc_mesh,
    n_samples=10000,
):
    """Align pred_mesh to gt_mesh."""

    pred_mesh = unwrap_mesh(pred_mesh)
    gt_mesh = unwrap_mesh(dtc_mesh)
    src, _ = trimesh.sample.sample_surface(pred_mesh, n_samples)
    tgt, _ = trimesh.sample.sample_surface(gt_mesh, n_samples)

    src_center = src.mean(axis=0)
    tgt_center = tgt.mean(axis=0)

    src0 = src - src_center
    tgt0 = tgt - tgt_center

    src_radius = np.sqrt(np.mean(np.sum(src0**2, axis=1)))
    tgt_radius = np.sqrt(np.mean(np.sum(tgt0**2, axis=1)))

    init_scale = tgt_radius / src_radius

    tgt_tree = cKDTree(tgt)

    def objective(params):
        rx, ry, rz, tx, ty, tz, log_s = params

        s = np.exp(log_s)

        R = Rotation.from_rotvec([rx, ry, rz]).as_matrix()

        transformed = (
            s * (src @ R.T)
            + np.array([tx, ty, tz])
        )

        tree_src = cKDTree(transformed)

        d1, _ = tgt_tree.query(transformed, k=1)
        d2, _ = tree_src.query(tgt, k=1)

        chamfer = d1.mean() + d2.mean()

        return chamfer

    x0 = np.array([
        0.0,
        0.0,
        0.0,
        *(tgt_center - init_scale * src_center),
        np.log(init_scale),
    ])

    result = minimize(
        objective,
        x0,
        method="Powell",
        options=dict(
            maxiter=300,
            xtol=1e-4,
            ftol=1e-4,
        ),
    )

    rx, ry, rz, tx, ty, tz, log_s = result.x

    scale = np.exp(log_s)

    R = Rotation.from_rotvec(
        [rx, ry, rz]
    ).as_matrix()

    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = [tx, ty, tz]

    return T, scale



def transf3d(pts3d, transf4x4):
    """Args:"""
    device = pts3d.device
    dtype = pts3d.dtype
    transf4x4 = transf4x4.to(device)
    ones1d = torch.ones(size=list(pts3d.shape[:-1]) + [1]).to(
        device=device,
        dtype=dtype,
    )
    pts4d = torch.concatenate([pts3d, ones1d], dim=-1)
    pts4d = pts4d.reshape(list(pts4d.shape) + [1])
    pts4d_transf = torch.bmm(
        transf4x4.reshape(-1, 4, 4),
        pts4d.reshape(-1, 4, 1),
    ).reshape(pts4d.shape)
    dim_coords3d = pts4d_transf.dim() - 2
    pts3d_transf = pts4d_transf.index_select(
        dim=dim_coords3d,
        index=torch.LongTensor([0, 1, 2]).to(device=device),
    )
    pts3d_transf = pts3d_transf.squeeze(dim=-1)
    return pts3d_transf

def transf3d_broadcast(pts3d, transf4x4):
    shape_first_dims = torch.broadcast_shapes(pts3d.shape[:-1], transf4x4.shape[:-2])
    return transf3d(
        pts3d.expand(*shape_first_dims, 3),
        transf4x4.expand(*shape_first_dims, 4, 4),
    )




def save_image(fn, x : np.ndarray) -> np.ndarray:
    try:
        if os.path.splitext(fn)[1] == ".png":
            imageio.imwrite(fn, np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8), compress_level=3)
        else:
            imageio.imwrite(fn, np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8))
    except:
        print("WARNING: FAILED to save image %s" % fn)
        

import numpy as np




def normalize_matrix(M):
    R = M[:3, :3]
    s = R.norm(dim=0).mean()
    M = M.clone()
    M[:3, :3] /= s
    M[:3, 3]  /= s
    return M

def push_camera_back(E, d):
    R = E[:3, :3]
    t = E[:3, 3]

    C = -R.T @ t

    fwd = R.T @ torch.tensor([0.0, 0.0, 1.0], device=E.device)

    C_new = C - d * fwd

    t_new = -R @ C_new

    E_new = E.clone()
    E_new[:3, 3] = t_new
    return E_new





def unwrap_mesh(mesh):
    
    if isinstance(mesh, trimesh.Trimesh):
        return mesh
    
    if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
        verts = np.asarray(mesh.vertices.detach().cpu().numpy())
        faces = np.asarray(mesh.faces.detach().cpu().numpy())
        return trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    
    raise TypeError(f"Unsupported mesh type: {type(mesh)}")


def pca_rotation(mesh):
    verts = mesh.vertices - mesh.vertices.mean(axis=0)
    cov = np.cov(verts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    return eigvecs

R_opengl_to_trellis = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
])

import numpy as np

def kabsch_rotation(A, B):
    """A, B: (N, 3) arrays of corresponding points"""
    Ac = A - A.mean(axis=0)
    Bc = B - B.mean(axis=0)

    H = Ac.T @ Bc

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return R




theta = np.radians(-90)
R_fix = np.array([
    [1, 0,            0,           0],
    [0, np.cos(theta), -np.sin(theta), 0],
    [0, np.sin(theta),  np.cos(theta), 0],
    [0, 0,            0,           1]
], dtype=np.float32)

def get_global_rotations():
    def rot_x(theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.eye(4, dtype=np.float32)
        R[1,1], R[1,2] =  c, -s
        R[2,1], R[2,2] =  s,  c
        return R

    def rot_z(theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.eye(4, dtype=np.float32)
        R[0,0], R[0,1] =  c, -s
        R[1,0], R[1,1] =  s,  c
        return R
    
    def rot_y(theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.eye(4, dtype=np.float32)
        R[0,0], R[0,2] =  c,  s
        R[2,0], R[2,2] = -s,  c
        return R


    return {
        "x90":   rot_x(np.pi/2),
        "x-90":  rot_x(-np.pi/2),
        "y90":   rot_y(np.pi/2),
        "y-90":  rot_y(-np.pi/2),
        "z90":   rot_z(np.pi/2),
        "z-90":  rot_z(-np.pi/2),
    }
    
def normalize_custom(verts):
    vmin = verts.min(dim=0).values
    vmax = verts.max(dim=0).values
    centroid = (vmin + vmax) * 0.5
    extents = vmax - vmin
    scale = extents.norm()

    verts_norm = (verts - centroid) / scale
    return verts_norm, centroid, scale











def normalize(mesh):
    mesh = mesh.copy()
    mesh.apply_translation(-mesh.centroid)
    scale = 1.0 / mesh.scale
    mesh.apply_scale(scale)
    return mesh

    
    

def compute_icp_alignment_transform(dtc_mesh, pred_mesh, simplify_target=10000, icp_iterations=50):
    """Compute robust ICP-based alignment transformation from predicted mesh to ground truth mesh."""
    device = pred_mesh.device if hasattr(pred_mesh, 'device') else torch.device('cpu')
    
    if hasattr(pred_mesh, 'simplify'):
        pred_mesh.simplify(simplify_target)
    
    dtc_trimesh = unwrap_mesh(dtc_mesh)
    pred_trimesh = unwrap_mesh(pred_mesh)
    
    dtc_verts = dtc_trimesh.vertices.copy()
    pred_verts = pred_trimesh.vertices.copy()
    
    dtc_centroid = dtc_verts.mean(axis=0)
    pred_centroid = pred_verts.mean(axis=0)
    
    dtc_bbox_min, dtc_bbox_max = dtc_verts.min(axis=0), dtc_verts.max(axis=0)
    pred_bbox_min, pred_bbox_max = pred_verts.min(axis=0), pred_verts.max(axis=0)
    
    dtc_diag = np.linalg.norm(dtc_bbox_max - dtc_bbox_min)
    pred_diag = np.linalg.norm(pred_bbox_max - pred_bbox_min)
    
    scale = dtc_diag / (pred_diag + 1e-8)
    
    print(f"[Alignment] DTC diag: {dtc_diag:.4f}, Pred diag: {pred_diag:.4f}, Scale: {scale:.4f}")
    print(f"[Alignment] DTC centroid: {dtc_centroid}, Pred centroid: {pred_centroid}")
    
    dtc_norm_verts = (dtc_verts - dtc_centroid) / (dtc_diag / 2.0)
    pred_norm_verts = (pred_verts - pred_centroid) / (pred_diag / 2.0)
    
    try:
        T_icp, _, _ = trimesh.registration.icp(
            dtc_norm_verts,
            pred_norm_verts,
            scale=False,
            max_iterations=icp_iterations,
            threshold=1e-6
        )
        print(f"[Alignment] ICP converged successfully")
        print(f"[Alignment] ICP transform:\n{T_icp}")
    except Exception as e:
        print(f"[Alignment] ICP failed: {e}. Using identity.")
        T_icp = np.eye(4)
    
    
    T_denorm_pred = np.eye(4)
    T_denorm_pred[:3, :3] = (pred_diag / 2.0) * np.eye(3)
    T_denorm_pred[:3, 3] = pred_centroid
    
    T_norm_dtc = np.eye(4)
    T_norm_dtc[:3, :3] = (2.0 / dtc_diag) * np.eye(3)
    T_norm_dtc[:3, 3] = -dtc_centroid * (2.0 / dtc_diag)
    
    
    T_norm_pred = np.eye(4)
    T_norm_pred[:3, :3] = (2.0 / pred_diag) * np.eye(3)
    T_norm_pred[:3, 3] = -pred_centroid * (2.0 / pred_diag)
    
    T_denorm_dtc = np.eye(4)
    T_denorm_dtc[:3, :3] = (dtc_diag / 2.0) * np.eye(3)
    T_denorm_dtc[:3, 3] = dtc_centroid
    
    T_scale = np.eye(4)
    T_scale[:3, :3] *= scale
    
    T_icp_result = T_icp @ T_norm_pred
    
    R_icp = T_icp_result[:3, :3]
    t_icp = T_icp_result[:3, 3]
    
    t_scaled = scale * t_icp + dtc_centroid - scale * pred_centroid
    
    T_final = np.eye(4)
    T_final[:3, :3] = R_icp
    T_final[:3, 3] = t_scaled
    
    print(f"[Alignment] Final transformation matrix:\n{T_final}")
    print(f"[Alignment] Rotation part preserved from ICP, translation scaled")
    
    return torch.tensor(T_final, dtype=torch.float32, device=device)


def get_normalization_matrix(dtc_mesh, pred_mesh):
    """DEPRECATED: Use compute_icp_alignment_transform instead."""
    return compute_icp_alignment_transform(dtc_mesh, pred_mesh)





import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window
def psnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2)
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)







from scipy.spatial import cKDTree

def align_meshes_pca(dtc_mesh, pred_mesh):
    """Aligns two identical shapes globally using Principal Component Analysis (PCA)."""
    source_mesh = unwrap_mesh(dtc_mesh)
    target_mesh = unwrap_mesh(pred_mesh)
    src_pts = source_mesh.vertices
    tgt_pts = target_mesh.vertices
    
    src_center = np.mean(src_pts, axis=0)
    tgt_center = np.mean(tgt_pts, axis=0)
    
    src_centered = src_pts - src_center
    tgt_centered = tgt_pts - tgt_center
    
    _, _, Vt_src = np.linalg.svd(src_centered, full_matrices=False)
    _, _, Vt_tgt = np.linalg.svd(tgt_centered, full_matrices=False)
    
    R_src = Vt_src.T
    R_tgt = Vt_tgt.T
    
    best_error = float('inf')
    best_R = np.eye(3)
    
    tgt_tree = cKDTree(tgt_centered)
    
    sign_flips = np.array(np.meshgrid([1, -1], [1, -1], [1, -1])).T.reshape(-1, 3)
    
    for flip in sign_flips:
        R_src_flipped = R_src * flip
        
        R_cand = R_tgt @ R_src_flipped.T
        
        if np.linalg.det(R_cand) < 0:
            continue
            
        test_pts = src_centered @ R_cand.T
        distances, _ = tgt_tree.query(test_pts, k=1)
        mean_error = np.mean(distances)
        
        if mean_error < best_error:
            best_error = mean_error
            best_R = R_cand

    T = np.eye(4)
    T[:3, :3] = best_R
    T[:3, 3] = tgt_center - best_R @ src_center
    
    return T




def align_meshes_ransac_icp(pred_mesh, dtc_mesh, voxel_size=0.05):
   
    source_mesh = unwrap_mesh(pred_mesh)
    target_mesh = unwrap_mesh(dtc_mesh)
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_mesh.vertices)
    
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_mesh.vertices)
    
    
    
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)
    
    source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))

    distance_threshold = voxel_size * 1.5
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, False, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(True), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.3),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

    rough_alignment = result_ransac.transformation

    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, voxel_size * 0.4, rough_alignment,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

 
    return result_icp.transformation








'''

coords, data = o_voxel.io.read("ovoxel_helmet.vxz")
position = (coords / RES - 0.5).cuda()
base_color = (data['base_color'] / 255).cuda()

# Render
renderer = o_voxel.rasterize.VoxelRenderer(
    rendering_options={"resolution": 512, "ssaa": 2}
)
output = renderer.render(
    position=position,          # Voxel centers
    attrs=base_color,           # Color/Opacity etc.
    voxel_size=1.0/RES,
    extrinsics=extr,
    intrinsics=intr
)

'''

'''
k, i guess you could try to 
1) get the o-voxel representation from the mesh, 
2) use the o-voxel parameters for optimization, 
3) get the mesh from the o-voxel representation to render
[2:06 PM]you would have to repeat 3) for each iteration




mesh = asset.to_mesh()
vertices = torch.from_numpy(mesh.vertices).float()
faces = torch.from_numpy(mesh.faces).long()
voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
    vertices, faces,
    grid_size=RES,                              # Resolution
    aabb=[[-0.5,-0.5,-0.5],[0.5,0.5,0.5]],      # Axis-aligned bounding box
    face_weight=1.0,                            # Face term weight in QEF
    boundary_weight=0.2,                        # Boundary term weight in QEF
    regularization_weight=1e-2,                 # Regularization term weight in QEF
    timing=True
)
## sort to ensure align between geometry and material voxelization
vid = o_voxel.serialize.encode_seq(voxel_indices)
mapping = torch.argsort(vid)
voxel_indices = voxel_indices[mapping]
dual_vertices = dual_vertices[mapping]
intersected = intersected[mapping]




# 2. Material Voxelization (Volumetric Attributes)
# Returns: dict containing 'base_color', 'metallic', 'roughness', etc.
voxel_indices_mat, attributes = o_voxel.convert.textured_mesh_to_volumetric_attr(
    asset,
    grid_size=RES,
    aabb=[[-0.5,-0.5,-0.5],[0.5,0.5,0.5]],
    timing=True
)
## sort to ensure align between geometry and material voxelization
vid_mat = o_voxel.serialize.encode_seq(voxel_indices_mat)
mapping_mat = torch.argsort(vid_mat)
attributes = {k: v[mapping_mat] for k, v in attributes.items()}

# Save to compressed .vxz format
## packing
dual_vertices = dual_vertices * RES - voxel_indices
dual_vertices = (torch.clamp(dual_vertices, 0, 1) * 255).type(torch.uint8)
intersected = (intersected[:, 0:1] + 2 * intersected[:, 1:2] + 4 * intersected[:, 2:3]).type(torch.uint8)
attributes['dual_vertices'] = dual_vertices
attributes['intersected'] = intersected
o_voxel.io.write("ovoxel_helmet.vxz", voxel_indices, attributes)




dual_vertices = data['dual_vertices']
intersected = data['intersected']
base_color = data['base_color']
## ... other attributes omitted for brevity

# Depack
dual_vertices = dual_vertices / 255
intersected = torch.cat([
    intersected % 2,
    intersected // 2 % 2,
    intersected // 4 % 2,
], dim=-1).bool()

# Extract Mesh
# O-Voxel connects dual vertices to form quads, optionally splitting them 
# based on geometric features.
rec_verts, rec_faces = o_voxel.convert.flexible_dual_grid_to_mesh(
    coords.cuda(), 
    dual_vertices.cuda(), 
    intersected.cuda(), 
    split_weight=None, # Auto-split based on min angle if None
    grid_size=RES,
    aabb=[[-0.5,-0.5,-0.5],[0.5,0.5,0.5]],
)

'''



'''

coords, attributes = o_voxel.io.read("your_file.vxz")  
  
# Unpack dual_vertices and intersected  
dual_vertices = attributes['dual_vertices'] / 255.0  # Back to float [0,1]  
intersected_packed = attributes['intersected']  
  
# Unpack intersected bits  
intersected = torch.stack([  
    intersected_packed % 2,  
    intersected_packed // 2 % 2,  
    intersected_packed // 4 % 2  
], dim=-1).bool()  
  
# Create the 7-channel h.feats format  
# Channels 0-3: dual_vertices (need to apply inverse sigmoid)  
# Channels 3-6: intersected_logits (need to convert from bool to logits)  
# Channel 6: quad_lerp (initialize to 0 or small value)  
  
# For dual_vertices, apply inverse of the sigmoid+margin scaling  
# Original: (1 + 2 * margin) * sigmoid(feats[..., 0:3]) - margin  
# Inverse: sigmoid_inv((vertices + margin) / (1 + 2 * margin))  
voxel_margin = 0.5  # Default from FlexiDualGridVaeDecoder  
dual_vertices_feats = torch.logit(  
    (dual_vertices + voxel_margin) / (1 + 2 * voxel_margin)  
)  
  
# For intersected, convert bool to logits (use reasonable values)  
intersected_logits = torch.where(  
    intersected,   
    torch.tensor(2.0),  # Positive logits for True  
    torch.tensor(-2.0)  # Negative logits for False  
).float()  
  
# For quad_lerp, initialize with softplus inverse of small positive values  
quad_lerp = F.softplus_inverse(torch.ones_like(dual_vertices[:, :1]) * 0.1)  
  
# Concatenate to create h.feats  
h_feats = torch.cat([  
    dual_vertices_feats,  # channels 0-3  
    intersected_logits,   # channels 3-6  
    quad_lerp            # channel 6  
], dim=-1)  
  
# Create SparseTensor h  
h = sp.SparseTensor(  
    feats=h_feats,  
    coords=torch.cat([torch.zeros_like(coords[:, 0:1]), coords], dim=-1)  
)  
  
# Make h.feats trainable for optimization  
h_feats = torch.nn.Parameter(h_feats)  
h_opt = h.replace(feats=h_feats)




'''




''''
Ovoxel to latent to mesh

import torch
import trimesh
import o_voxel
from trellis2 import models
from trellis2.modules.sparse import SparseTensor
from trellis2.representations import MeshWithVoxel


def load_shape_models():
    shape_enc = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16")
    shape_dec = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16")
    for model in [shape_enc, shape_dec]:
        model.cuda().eval()
    
    return shape_enc, shape_dec


def normalize_mesh(mesh):
    """Normalize mesh vertices to unit cube."""
    vertices = torch.from_numpy(mesh.vertices).float()
    faces = torch.from_numpy(mesh.faces).long()
    
    center = (vertices.min(0)[0] + vertices.max(0)[0]) / 2
    scale = 0.99999 / (vertices.max(0)[0] - vertices.min(0)[0]).max()
    vertices = (vertices - center) * scale
    
    return vertices, faces


def encode_decode_shape(encoder, decoder, vertices, faces, resolution=512):
    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices, faces, grid_size=resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    )
    
    coords = torch.cat([torch.zeros(len(voxel_indices), 1, dtype=torch.int32), voxel_indices], dim=1)
    vertices_sparse = SparseTensor(dual_vertices.cuda(), coords.cuda())
    intersected_sparse = SparseTensor(intersected.float().cuda(), coords.cuda())
    
    with torch.no_grad():
        latent = encoder(vertices_sparse, intersected_sparse)
        decoder.set_resolution(resolution)
        meshes, subs = decoder(latent, return_subs=True)
    
    return meshes[0], subs


def postprocess_with_default_texture(mesh, subs, resolution):
    pbr_attr_layout = {
        'base_color': slice(0, 3),
        'metallic': slice(3, 4),
        'roughness': slice(4, 5),
        'alpha': slice(5, 6),
    }
    
    mesh.fill_holes()
    
    # Create default white material
    sub = subs[-1]
    num_voxels = sub.coords.shape[0]
    default_attrs = torch.ones(num_voxels, 6, device=sub.coords.device, dtype=torch.float32)
    default_attrs[:, 3] = 0.0  # metallic
    default_attrs[:, 4] = 0.5  # roughness
    default_attrs[:, 5] = 1.0  # alpha
    
    mesh_with_voxel = MeshWithVoxel(
        mesh.vertices, mesh.faces,
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1 / resolution,
        coords=sub.coords[:, 1:],
        attrs=default_attrs,
        voxel_shape=torch.Size([*sub.shape, *sub.spatial_shape]),
        layout=pbr_attr_layout
    )
    
    mesh_with_voxel.simplify(16777216)
    
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh_with_voxel.vertices,
        faces=mesh_with_voxel.faces,
        attr_volume=mesh_with_voxel.attrs,
        coords=mesh_with_voxel.coords,
        attr_layout=mesh_with_voxel.layout,
        voxel_size=mesh_with_voxel.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=1000000,
        texture_size=4096,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True
    )
    return glb


def save_mesh(mesh, output_path):
    trimesh.Trimesh(
        vertices=mesh.vertices.cpu().numpy(),
        faces=mesh.faces.cpu().numpy(),
    ).export(output_path)


def main():
    shape_enc, shape_dec = load_shape_models()
    
    mesh_path = "/common_data/workspace/ariel/repos/Hunyuan3D-2.1/assets/cpt_america_out.glb"
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = list(mesh.geometry.values())[0]
    vertices, faces = normalize_mesh(mesh)
    
    resolution = 1024
    shape_mesh, subs = encode_decode_shape(shape_enc, shape_dec, vertices, faces, resolution=resolution)
    save_mesh(shape_mesh, "reconstructed_raw_shape.glb")

    glb_mesh = postprocess_with_default_texture(shape_mesh, subs, resolution)
    glb_mesh.export("reconstructed_mesh_postprocessed.glb")
    


if __name__ == "__main__":
    main()



'''




'''


   
    
import torch
import torch.nn.functional as F

def tsdf_edge_topology_loss(
    rep,                       # MeshExtractResult with attributes: tsdf_v (N,3), tsdf_s (N,) and res (int)
    max_neighbors: int = 8,
    grid_res: int | None = None,
    neighbor_radius: float | None = None,
    truncation_factor: float = 3.0,
    loss_type: str = "bce",    # "bce" or "hinge"
    hinge_margin: float = 0.0,
    device: torch.device | None = None,
    weight: float = 1.0,
):
    """
    Compute a topology-aware loss from rep.tsdf_v and rep.tsdf_s.

    - Builds adjacency by connecting each tsdf vertex to nearby vertices within `neighbor_radius`.
      If neighbor_radius is None, it defaults to 1.1 * (1 / grid_res) if grid_res provided, else 1.1 * median NN distance.
    - Keeps only candidate edges near the surface band (|s| < trunc), where trunc = truncation_factor / res.
    - loss_type "bce": binary_cross_entropy_with_logits(s_i, (s_j > 0).float()) + symmetric term.
    - loss_type "hinge": relu(s_i * s_j + margin) encourages opposite signs (s_i * s_j < -margin).
    - Returns scalar loss (tensor).
    """
    if device is None:
        device = rep.tsdf_v.device

    v = rep.tsdf_v  # (N,3)
    s = rep.tsdf_s.flatten()  # (N,)
    N = v.shape[0]
    if N < 2:
        return torch.tensor(0.0, device=device, dtype=s.dtype)

    # grid_res fallback
    if grid_res is None:
        grid_res = getattr(rep, "res", None)

    # compute pairwise distances (efficiently in chunks if needed)
    # We'll compute full distance matrix for simplicity; for very large N you should use sparse kNN.
    dists = torch.cdist(v, v)  # (N,N)
    dists.fill_diagonal_(float('inf'))

    # determine neighbor radius
    if neighbor_radius is None:
        if grid_res is not None:
            neighbor_radius = 1.1 * (1.0 / float(grid_res))
        else:
            # fallback: use median nearest neighbor distance
            nn_dist, _ = torch.min(dists, dim=1)
            neighbor_radius = 1.1 * torch.median(nn_dist).item()
            if neighbor_radius == 0:
                neighbor_radius = 1e-3

    # adjacency mask: connect if distance <= neighbor_radius
    adj_mask = dists <= neighbor_radius

    # optionally limit neighbors per vertex to max_neighbors (keep closest)
    if max_neighbors is not None and max_neighbors > 0:
        # for each row, keep only the smallest max_neighbors distances
        topk = torch.topk(dists, k=min(max_neighbors, N-1), largest=False)
        kth = topk.values[:, -1].unsqueeze(1)  # threshold per row
        adj_mask = adj_mask & (dists <= kth)

    # extract unique undirected edges (i < j)
    idx_i, idx_j = torch.nonzero(adj_mask, as_tuple=True)
    mask_upper = idx_i < idx_j
    idx_i = idx_i[mask_upper]
    idx_j = idx_j[mask_upper]
    if idx_i.numel() == 0:
        return torch.tensor(0.0, device=device, dtype=s.dtype)

    s_i = s[idx_i]
    s_j = s[idx_j]

    # truncation band: only consider edges where at least one endpoint is near surface
    if grid_res is not None:
        trunc = truncation_factor / float(grid_res)
    else:
        # fallback trunc based on median absolute value
        trunc = truncation_factor * torch.median(s.abs()).item() if s.numel() > 0 else 1e-3
        trunc = max(trunc, 1e-4)

    candidate_mask = (s_i.abs() < trunc) | (s_j.abs() < trunc)
    if candidate_mask.sum() == 0:
        return torch.tensor(0.0, device=device, dtype=s.dtype)

    s_i = s_i[candidate_mask]
    s_j = s_j[candidate_mask]

    if loss_type == "bce":
        # treat s as logits for inside/outside (positive => inside)
        target_j_pos = (s_j > 0).float()
        target_i_pos = (s_i > 0).float()
        loss_ij = F.binary_cross_entropy_with_logits(s_i, target_j_pos)
        loss_ji = F.binary_cross_entropy_with_logits(s_j, target_i_pos)
        loss = (loss_ij + loss_ji) * 0.5
    elif loss_type == "hinge":
        # encourage s_i * s_j to be negative (opposite signs)
        prod = s_i * s_j
        # want prod < -margin -> penalize prod + margin when > 0
        loss = F.relu(prod + hinge_margin).mean()
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    return loss * float(weight)


# Helper: TSDF projection/regression loss from the document (drop-in)
def tsdf_projection_loss(rep, depth_map: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor):
    """
    Reimplementation of the _tsdf_reg_loss from your document.
    Projects rep.tsdf_v into camera, samples GT depth, forms pseudo-sdf and regresses rep.tsdf_s to it.
    """
    with torch.no_grad():
        projected_pts, pts_depth = utils3d.torch.project_cv(
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            points=rep.tsdf_v,
        )
        projected_pts = (projected_pts - 0.5) * 2.0
        depth_map_res = depth_map.shape[-1]
        gt_depth = torch.nn.functional.grid_sample(
            depth_map.reshape(1, 1, depth_map_res, depth_map_res),
            projected_pts.reshape(1, 1, -1, 2),
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )
        pseudo_sdf = gt_depth.flatten() - pts_depth.flatten()
        delta = 1.0 / rep.res * 3.0
        trunc_mask = pseudo_sdf > -delta

    gt_tsdf = pseudo_sdf[trunc_mask]
    tsdf = rep.tsdf_s.flatten()[trunc_mask]
    gt_tsdf = torch.clamp(gt_tsdf, -delta, delta)
    return torch.mean((tsdf - gt_tsdf) ** 2)

'''
