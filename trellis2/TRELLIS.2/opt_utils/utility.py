import os
import numpy as np
import imageio
import torch
import trimesh

def transf3d(pts3d, transf4x4):
    """Apply a 4x4 homogeneous transform to points of shape [..., 3]."""
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

_glfw_initialized = False
def init_glfw():
    global _glfw_initialized
    try:
        import glfw
        glfw.ERROR_REPORTING = 'raise'
        glfw.default_window_hints()
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        test = glfw.create_window(8, 8, "Test", None, None)
    except glfw.GLFWError as e:
        if e.error_code == glfw.NOT_INITIALIZED:
            glfw.init()
            _glfw_initialized = True


_glfw_window = None
def display_image(image, title=None):
    import OpenGL.GL as gl
    import glfw

    image = np.asarray(image[..., 0:3]) if image.shape[-1] == 4 else np.asarray(image)
    height, width, channels = image.shape

    init_glfw()
    if title is None:
        title = 'Debug window'
    global _glfw_window
    if _glfw_window is None:
        glfw.default_window_hints()
        _glfw_window = glfw.create_window(width, height, title, None, None)
        glfw.make_context_current(_glfw_window)
        glfw.show_window(_glfw_window)
        glfw.swap_interval(0)
    else:
        glfw.make_context_current(_glfw_window)
        glfw.set_window_title(_glfw_window, title)
        glfw.set_window_size(_glfw_window, width, height)

    glfw.poll_events()
    gl.glClearColor(0, 0, 0, 1)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glWindowPos2f(0, 0)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl_format = {3: gl.GL_RGB, 2: gl.GL_RG, 1: gl.GL_LUMINANCE}[channels]
    gl_dtype = {'uint8': gl.GL_UNSIGNED_BYTE, 'float32': gl.GL_FLOAT}[image.dtype.name]
    gl.glDrawPixels(width, height, gl_format, gl_dtype, image[::-1])
    glfw.swap_buffers(_glfw_window)
    if glfw.window_should_close(_glfw_window):
        return False
    return True


def save_image(fn, x : np.ndarray) -> np.ndarray:
    try:
        if os.path.splitext(fn)[1] == ".png":
            imageio.imwrite(fn, np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8), compress_level=3)
        else:
            imageio.imwrite(fn, np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8))
    except:
        print("WARNING: FAILED to save image %s" % fn)
        

import numpy as np

def opengl_projection_from_intrinsics(fx, fy, cx, cy, width, height, near=0.01, far=100.0):
    proj = np.zeros((4, 4), dtype=np.float32)

    proj[0, 0] = 2 * fx / width
    proj[1, 1] = 2 * fy / height

    proj[0, 2] = 1 - 2 * cx / width
    proj[1, 2] = 2 * cy / height - 1

    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -(2 * far * near) / (far - near)

    proj[3, 2] = -1

    return proj


def _parse_frame(self, cfg, idx):
        fovy   = util.fovx_to_fovy(cfg['camera_angle_x'], self.aspect) 
        proj   = util.perspective(fovy, self.aspect, self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        img    = _load_img(os.path.join(self.base_dir, cfg['frames'][idx]['file_path']))
    
        mv = torch.tensor(cfg['frames'][idx]['transform_matrix'], dtype=torch.float32)
        object_matrix = torch.tensor(cfg['object_pose'], dtype=torch.float32)
        
        
        mv = mv @ object_matrix

        flip_zy = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float32))
        mv = flip_zy @ mv

        flip_obj = torch.diag(torch.tensor([1.0, -1.0, 1.0, 1.0], dtype=torch.float32)) 
        mv = mv @ flip_obj
        
        scale = 5.0
        S = torch.diag(torch.tensor([1/scale, 1/scale, 1/scale, 1.0], dtype=torch.float32))
        mv = mv @ S

        
        campos = torch.linalg.inv(mv)[:3, 3]   
        mvp    = proj @ mv
        
       
        return img[None, ...], mv[None, ...], mvp[None, ...], campos[None, ...]

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
    """Optimal rotation R (3x3) aligning corresponding points A -> B, both (N, 3) (Kabsch)."""
    Ac = A - A.mean(axis=0)
    Bc = B - B.mean(axis=0)

    H = Ac.T @ Bc

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return R


def convert_to_trellis_axes(mesh):
    if hasattr(mesh, "vertices"):
        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
    else:
        verts = np.asarray(mesh)
        faces = None

    verts_trellis = verts @ R_opengl_to_trellis.T

    if faces is not None:
        return trimesh.Trimesh(vertices=verts_trellis, faces=faces, process=False)
    else:
        return verts_trellis

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


def compute_simple_alignment_transform(dtc_mesh, pred_mesh):
    """Compute simple center-and-scale alignment (no ICP)."""
    device = pred_mesh.device if hasattr(pred_mesh, 'device') else torch.device('cpu')
    
    if hasattr(pred_mesh, 'simplify'):
        try:
            pred_mesh.simplify(10000)
        except:
            pass
    
    dtc_trimesh = unwrap_mesh(dtc_mesh)
    pred_trimesh = unwrap_mesh(pred_mesh)
    
    dtc_verts = dtc_trimesh.vertices.copy()
    pred_verts = pred_trimesh.vertices.copy()
    
    dtc_centroid = dtc_verts.mean(axis=0)
    pred_centroid = pred_verts.mean(axis=0)
    
    dtc_bbox_diag = np.linalg.norm(dtc_verts.max(axis=0) - dtc_verts.min(axis=0))
    pred_bbox_diag = np.linalg.norm(pred_verts.max(axis=0) - pred_verts.min(axis=0))
    
    scale = dtc_bbox_diag / (pred_bbox_diag + 1e-8)
    
    print(f"[Simple Alignment] DTC centroid: {dtc_centroid}, scale: {dtc_bbox_diag:.4f}")
    print(f"[Simple Alignment] Pred centroid: {pred_centroid}, scale: {pred_bbox_diag:.4f}")
    print(f"[Simple Alignment] Scale factor: {scale:.4f}")
    
    T = np.eye(4)
    T[:3, :3] *= scale
    T[:3, 3] = dtc_centroid - scale * pred_centroid
    
    print(f"[Simple Alignment] Transformation:\n{T}")
    
    return torch.tensor(T, dtype=torch.float32, device=device)
 

import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from lpips import LPIPS

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


loss_fn_vgg = None
def lpips(img1, img2, value_range=(0, 1)):
    global loss_fn_vgg
    if loss_fn_vgg is None:
        loss_fn_vgg = LPIPS(net='vgg').cuda().eval()
    img1 = (img1 - value_range[0]) / (value_range[1] - value_range[0]) * 2 - 1
    img2 = (img2 - value_range[0]) / (value_range[1] - value_range[0]) * 2 - 1
    return loss_fn_vgg(img1, img2).mean()


def normal_angle(pred, gt):
    pred = pred * 2.0 - 1.0
    gt = gt * 2.0 - 1.0
    norms = pred.norm(dim=-1) * gt.norm(dim=-1)
    cos_sim = (pred * gt).sum(-1) / (norms + 1e-9)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    ang = torch.rad2deg(torch.acos(cos_sim[norms > 1e-9])).mean()
    if ang.isnan():
        return -1
    return ang


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


import open3d as o3d

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


from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


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

def bake_extrinsic_intrinsic(
    M_cv: torch.Tensor,
    intr_norm: torch.Tensor,
    object_pose: torch.Tensor = None,
    scale: float = 1.0,
):
    """Bake OpenCV->OpenGL flips, object pose, and world scale directly into"""

    device = M_cv.device
    dtype = M_cv.dtype

    extr = M_cv.clone()

    if object_pose is not None:
        extr = extr @ object_pose.to(device=device, dtype=dtype)

    flip_zy = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=dtype, device=device))
    extr = flip_zy @ extr

    flip_obj = torch.diag(torch.tensor([1.0, -1.0, 1.0, 1.0], dtype=dtype, device=device))
    extr = extr @ flip_obj

    S = torch.diag(torch.tensor([1.0/scale, 1.0/scale, 1.0/scale, 1.0], dtype=dtype, device=device))
    extr = extr @ S

    baked_intr = intr_norm.clone()

    baked_intr[1, 1] *= -1.0


    return extr, baked_intr

