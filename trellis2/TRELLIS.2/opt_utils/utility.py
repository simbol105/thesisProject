
import os
import numpy as np
import imageio
import torch
import trimesh

def transf3d(pts3d, transf4x4):
    """
    Args:
        pts3d (torch.Tensor): ...x3
        transf4x4 (torch.Tensor): ...x4x4

    Returns:
        pts3d_transf (torch.Tensor): ...x3
    """
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
        test = glfw.create_window(8, 8, "Test", None, None) # Create a window and see if not initialized yet
    except glfw.GLFWError as e:
        if e.error_code == glfw.NOT_INITIALIZED:
            glfw.init()
            _glfw_initialized = True


_glfw_window = None
def display_image(image, title=None):
    # Import OpenGL
    import OpenGL.GL as gl
    import glfw

    # Zoom image if requested.
    image = np.asarray(image[..., 0:3]) if image.shape[-1] == 4 else np.asarray(image)
    height, width, channels = image.shape

    # Initialize window.
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

    # Update window.
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

#----------------------------------------------------------------------------
# Image save/load helper.
#----------------------------------------------------------------------------

def save_image(fn, x : np.ndarray) -> np.ndarray:
    try:
        if os.path.splitext(fn)[1] == ".png":
            imageio.imwrite(fn, np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8), compress_level=3) # Low compression for faster saving
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

        # Load image data and modelview matrix
        img    = _load_img(os.path.join(self.base_dir, cfg['frames'][idx]['file_path']))
    
        # mv = torch.linalg.inv(torch.tensor(cfg['frames'][idx]['transform_matrix'], dtype=torch.float32))
        mv = torch.tensor(cfg['frames'][idx]['transform_matrix'], dtype=torch.float32)
        object_matrix = torch.tensor(cfg['object_pose'], dtype=torch.float32)
        
        
   
        mv = mv @ object_matrix

   # Convert OpenCV (+Z forward) -> OpenGL (–Z forward) 
        flip_zy = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float32))
        mv = flip_zy @ mv

        flip_obj = torch.diag(torch.tensor([1.0, -1.0, 1.0, 1.0], dtype=torch.float32)) 
        mv = mv @ flip_obj
        
        scale = 5.0
        S = torch.diag(torch.tensor([1/scale, 1/scale, 1/scale, 1.0], dtype=torch.float32))
        mv = mv @ S

        # mv = mv @ util.rotate_y(np.pi/2)
        

        campos = torch.linalg.inv(mv)[:3, 3]   
        mvp    = proj @ mv
        
       

        return img[None, ...], mv[None, ...], mvp[None, ...], campos[None, ...] # Add batch dimension   

def normalize_matrix(M):
    R = M[:3, :3]
    s = R.norm(dim=0).mean()
    M = M.clone()
    M[:3, :3] /= s
    M[:3, 3]  /= s
    return M

def push_camera_back(E, d):
    # E is 4x4 world→camera extrinsic
    R = E[:3, :3]
    t = E[:3, 3]

    # camera center in world space
    C = -R.T @ t

    # camera forward direction in world space (OpenCV: +Z)
    fwd = R.T @ torch.tensor([0.0, 0.0, 1.0], device=E.device)

    # move camera backward by distance d
    C_new = C - d * fwd

    # recompute translation
    t_new = -R @ C_new

    E_new = E.clone()
    E_new[:3, 3] = t_new
    return E_new




# import trimesh
# import numpy as np

def unwrap_mesh(mesh):
    
    if isinstance(mesh, trimesh.Trimesh):
        return mesh
    
    # Trellis2 MeshWithVoxel: has vertices + faces
    if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
        verts = np.asarray(mesh.vertices.detach().cpu().numpy())
        faces = np.asarray(mesh.faces.detach().cpu().numpy())
        return trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    
    raise TypeError(f"Unsupported mesh type: {type(mesh)}")


def pca_rotation(mesh):
    verts = mesh.vertices - mesh.vertices.mean(axis=0)
    cov = np.cov(verts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    idx = np.argsort(eigvals)[::-1]  # descending
    eigvecs = eigvecs[:, idx]
    return eigvecs

R_opengl_to_trellis = np.array([
    [1, 0, 0],   # X stays X
    [0, 0, 1],   # Y -> Z
    [0, 1, 0],   # Z -> Y
])

import numpy as np

def kabsch_rotation(A, B):
    """
    A, B: (N, 3) arrays of corresponding points
    returns R: (3, 3) optimal rotation aligning A -> B
    """
    # Center
    Ac = A - A.mean(axis=0)
    Bc = B - B.mean(axis=0)

    # Covariance
    H = Ac.T @ Bc

    # SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Fix reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return R


def convert_to_trellis_axes(mesh):
    # Accept either a Trimesh or raw vertices
    if hasattr(mesh, "vertices"):
        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
    else:
        verts = np.asarray(mesh)
        faces = None

    # Apply axis conversion
    verts_trellis = verts @ R_opengl_to_trellis.T

    # Return same type as input
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
    # verts: torch tensor [N, 3]
    vmin = verts.min(dim=0).values
    vmax = verts.max(dim=0).values
    centroid = (vmin + vmax) * 0.5
    extents = vmax - vmin
    scale = extents.norm()  # diagonal of bounding box

    verts_norm = (verts - centroid) / scale
    return verts_norm, centroid, scale

# def get_normalization_matrix(dtc_mesh, pred_mesh):
#     # unwrap to get verts/faces as torch tensors
#     pred_mesh.simplify(target=100000)
#     dtc = unwrap_mesh(dtc_mesh)
#     pred = unwrap_mesh(pred_mesh)

#     # --- Normalize BEFORE converting to trimesh ---
#     A_verts, A_centroid, A_scale = normalize_custom(torch.tensor(dtc.vertices))
#     B_verts, B_centroid, B_scale = normalize_custom(torch.tensor(pred.vertices))

#     # Convert to trimesh
#     A = trimesh.Trimesh(vertices=A_verts.numpy(), faces=dtc.faces)
#     B = trimesh.Trimesh(vertices=B_verts.numpy(), faces=pred.faces)

#     # --- ICP on normalized meshes ---
#     T_init, _ = trimesh.registration.mesh_other(
#         A, B,
#         scale=False,
#         icp_first=10,     # coarse alignment
#         samples=5000
#     )

#     # T_norm, _ = trimesh.registration.mesh_other(
#     #     A, B,
#     #     scale=False,
#     #     samples=10000,
#     #     initial=T_init
#     # )
#     T_norm, _, _ = trimesh.registration.icp(
#         A.vertices,
#         B.vertices,
#         initial=T_init,   # now allowed
#         scale=False,
#         max_iterations=50
#     )

#     # --- Build un-normalization transforms ---
#     S_A = np.eye(4); S_A[:3, :3] *= A_scale.item()
#     S_B = np.eye(4); S_B[:3, :3] *= B_scale.item()

#     T_A = np.eye(4); T_A[:3, 3] = A_centroid.numpy()
#     T_B = np.eye(4); T_B[:3, 3] = B_centroid.numpy()

#     # Final transform in original units
#     T_final = T_A @ S_A @ T_norm @ np.linalg.inv(S_B) @ np.linalg.inv(T_B)

#     return torch.tensor(T_final, dtype=torch.float32, device=pred_mesh.device)


def normalize(mesh):
    mesh = mesh.copy()
    mesh.apply_translation(-mesh.centroid)
    scale = 1.0 / mesh.scale
    mesh.apply_scale(scale)
    return mesh

# def get_normalization_matrix(dtc_mesh, pred_mesh):
#     pred_mesh.simplify(50000)
#     device = pred_mesh.device
#     dtc = unwrap_mesh(dtc_mesh)
#     pred = unwrap_mesh(pred_mesh)
#     A_verts, A_centroid, A_scale = normalize_custom(torch.tensor(dtc.vertices))
#     B_verts, B_centroid, B_scale = normalize_custom(torch.tensor(pred.vertices))
#     A = trimesh.Trimesh(vertices=A_verts.numpy(), faces=dtc.faces)
#     B = trimesh.Trimesh(vertices=B_verts.numpy(), faces=pred.faces)
#     # Coarse alignment
#     T_init, _ = trimesh.registration.mesh_other(A, B, scale=True, samples=10000)
#     T_norm, _, _ = trimesh.registration.icp(
#         A.vertices, 
#         B.vertices, 
#         initial=T_init,
#         scale=True,
#         max_iterations=50
#     )
#     S_A = np.eye(4); S_A[:3, :3] *= A_scale.item()
#     S_B = np.eye(4); S_B[:3, :3] *= B_scale.item()
#     T_A = np.eye(4); T_A[:3, 3] = A_centroid.numpy()
#     T_B = np.eye(4); T_B[:3, 3] = B_centroid.numpy()
#     T_final = T_A @ S_A @ T_norm @ np.linalg.inv(S_B) @ np.linalg.inv(T_B)
    
#     return torch.tensor(T_final, dtype=torch.float32, device=device)
    

def compute_icp_alignment_transform(dtc_mesh, pred_mesh, simplify_target=10000, icp_iterations=50):
    """
    Compute robust ICP-based alignment transformation from predicted mesh to ground truth mesh.
    
    This function aligns two meshes by:
    1. Simplifying the predicted mesh
    2. Computing centroids and scaling
    3. Using ICP registration on normalized meshes
    4. Properly denormalizing the result
    
    Args:
        dtc_mesh: Ground truth mesh (trimesh.Trimesh or MeshWithVoxel)
        pred_mesh: Predicted mesh to be aligned (trimesh.Trimesh or MeshWithVoxel)
        simplify_target: Target vertex count for simplification
        icp_iterations: Number of ICP iterations
        
    Returns:
        T_final: 4x4 transformation matrix that aligns pred_mesh to dtc_mesh space (torch.Tensor)
    """
    device = pred_mesh.device if hasattr(pred_mesh, 'device') else torch.device('cpu')
    
    # Simplify predicted mesh first
    if hasattr(pred_mesh, 'simplify'):
        pred_mesh.simplify(simplify_target)
    
    # Unwrap mesh objects
    dtc_trimesh = unwrap_mesh(dtc_mesh)
    pred_trimesh = unwrap_mesh(pred_mesh)
    
    # Work with copies to avoid modifying originals
    dtc_verts = dtc_trimesh.vertices.copy()
    pred_verts = pred_trimesh.vertices.copy()
    
    # Compute centroids
    dtc_centroid = dtc_verts.mean(axis=0)
    pred_centroid = pred_verts.mean(axis=0)
    
    # Compute scale from bounding box diagonals
    dtc_bbox_min, dtc_bbox_max = dtc_verts.min(axis=0), dtc_verts.max(axis=0)
    pred_bbox_min, pred_bbox_max = pred_verts.min(axis=0), pred_verts.max(axis=0)
    
    dtc_diag = np.linalg.norm(dtc_bbox_max - dtc_bbox_min)
    pred_diag = np.linalg.norm(pred_bbox_max - pred_bbox_min)
    
    scale = dtc_diag / (pred_diag + 1e-8)
    
    print(f"[Alignment] DTC diag: {dtc_diag:.4f}, Pred diag: {pred_diag:.4f}, Scale: {scale:.4f}")
    print(f"[Alignment] DTC centroid: {dtc_centroid}, Pred centroid: {pred_centroid}")
    
    # Normalize vertices to unit scale, centered at origin
    dtc_norm_verts = (dtc_verts - dtc_centroid) / (dtc_diag / 2.0)
    pred_norm_verts = (pred_verts - pred_centroid) / (pred_diag / 2.0)
    
    # Use ICP for alignment in normalized space
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
    
    # Construct transformation in original space
    # T_final transforms points: p_dtc_space = T_final @ p_pred_space
    # Derivation:
    #   1. Center pred at origin: p' = p - pred_centroid
    #   2. Normalize pred scale: p'' = p' / (pred_diag/2)
    #   3. Apply ICP in normalized space: p''' = T_icp @ p''
    #   4. Denormalize: p'''' = p''' * (dtc_diag/2)
    #   5. Re-center at dtc: p_final = p'''' + dtc_centroid
    
    # Compose transforms as 4x4 matrices:
    # T_denorm_pred: undo normalization for pred (center and scale)
    T_denorm_pred = np.eye(4)
    T_denorm_pred[:3, :3] = (pred_diag / 2.0) * np.eye(3)
    T_denorm_pred[:3, 3] = pred_centroid
    
    # T_norm_dtc: normalize dtc mesh (inverse of above)
    T_norm_dtc = np.eye(4)
    T_norm_dtc[:3, :3] = (2.0 / dtc_diag) * np.eye(3)
    T_norm_dtc[:3, 3] = -dtc_centroid * (2.0 / dtc_diag)
    
    # Compose: denorm_pred -> icp -> norm_dtc -> scale
    # p_out = inv(T_norm_dtc) @ T_icp @ T_norm_pred @ p_in
    # which is: p_out = denorm_dtc @ T_icp @ norm_pred @ p_in
    
    T_norm_pred = np.eye(4)
    T_norm_pred[:3, :3] = (2.0 / pred_diag) * np.eye(3)
    T_norm_pred[:3, 3] = -pred_centroid * (2.0 / pred_diag)
    
    T_denorm_dtc = np.eye(4)
    T_denorm_dtc[:3, :3] = (dtc_diag / 2.0) * np.eye(3)
    T_denorm_dtc[:3, 3] = dtc_centroid
    
    # Apply scale to account for different mesh sizes
    T_scale = np.eye(4)
    T_scale[:3, :3] *= scale
    
    # Compose transformations - IMPORTANT: only scale the translation, not rotation
    # First apply ICP in normalized space
    T_icp_result = T_icp @ T_norm_pred  # Normalize and register
    
    # Extract rotation and translation from ICP result
    R_icp = T_icp_result[:3, :3]
    t_icp = T_icp_result[:3, 3]
    
    # Apply denormalization: scale and translate
    # p_final = denorm_dtc(scale * denorm_pred_inv(p_icp))
    t_scaled = scale * t_icp + dtc_centroid - scale * pred_centroid
    
    # Build final matrix with proper rotation
    T_final = np.eye(4)
    T_final[:3, :3] = R_icp  # Keep rotation from ICP
    T_final[:3, 3] = t_scaled  # Scale translation only
    
    print(f"[Alignment] Final transformation matrix:\n{T_final}")
    print(f"[Alignment] Rotation part preserved from ICP, translation scaled")
    
    return torch.tensor(T_final, dtype=torch.float32, device=device)


# Legacy function kept for backward compatibility
def get_normalization_matrix(dtc_mesh, pred_mesh):
    """
    DEPRECATED: Use compute_icp_alignment_transform instead.
    This function is kept for backward compatibility.
    """
    return compute_icp_alignment_transform(dtc_mesh, pred_mesh)


def compute_simple_alignment_transform(dtc_mesh, pred_mesh):
    """
    Compute simple center-and-scale alignment (no ICP).
    Use this as a fallback if ICP-based alignment produces bad results.
    
    Args:
        dtc_mesh: Ground truth mesh (trimesh.Trimesh or MeshWithVoxel)
        pred_mesh: Predicted mesh to be aligned (trimesh.Trimesh or MeshWithVoxel)
        
    Returns:
        T_final: 4x4 transformation matrix (torch.Tensor)
    """
    device = pred_mesh.device if hasattr(pred_mesh, 'device') else torch.device('cpu')
    
    # Simplify predicted mesh if possible
    if hasattr(pred_mesh, 'simplify'):
        try:
            pred_mesh.simplify(10000)
        except:
            pass
    
    # Unwrap mesh objects
    dtc_trimesh = unwrap_mesh(dtc_mesh)
    pred_trimesh = unwrap_mesh(pred_mesh)
    
    # Get vertices
    dtc_verts = dtc_trimesh.vertices.copy()
    pred_verts = pred_trimesh.vertices.copy()
    
    # Compute centroids
    dtc_centroid = dtc_verts.mean(axis=0)
    pred_centroid = pred_verts.mean(axis=0)
    
    # Compute scales from bounding boxes
    dtc_bbox_diag = np.linalg.norm(dtc_verts.max(axis=0) - dtc_verts.min(axis=0))
    pred_bbox_diag = np.linalg.norm(pred_verts.max(axis=0) - pred_verts.min(axis=0))
    
    scale = dtc_bbox_diag / (pred_bbox_diag + 1e-8)
    
    print(f"[Simple Alignment] DTC centroid: {dtc_centroid}, scale: {dtc_bbox_diag:.4f}")
    print(f"[Simple Alignment] Pred centroid: {pred_centroid}, scale: {pred_bbox_diag:.4f}")
    print(f"[Simple Alignment] Scale factor: {scale:.4f}")
    
    # Build transformation: scale then translate
    T = np.eye(4)
    T[:3, :3] *= scale
    T[:3, 3] = dtc_centroid - scale * pred_centroid
    
    print(f"[Simple Alignment] Transformation:\n{T}")
    
    return torch.tensor(T, dtype=torch.float32, device=device)
 



# Loss Utils 

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
    # normalize to [-1, 1]
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


# Get initial structure from DmTeT, 
# Update dmtet directly with the loss  
# If it works, use the logic to update the trellis latent optimization process. 
# 







from scipy.spatial import cKDTree

def align_meshes_pca(dtc_mesh, pred_mesh):
    """
    Aligns two identical shapes globally using Principal Component Analysis (PCA).
    No initial guess is required.
    """
    # 1. Get vertex arrays
    source_mesh = unwrap_mesh(dtc_mesh)
    target_mesh = unwrap_mesh(pred_mesh)
    src_pts = source_mesh.vertices
    tgt_pts = target_mesh.vertices
    
    # 2. Compute centers of mass and center the points
    src_center = np.mean(src_pts, axis=0)
    tgt_center = np.mean(tgt_pts, axis=0)
    
    src_centered = src_pts - src_center
    tgt_centered = tgt_pts - tgt_center
    
    # 3. Compute Covariance Matrices and Singular Value Decomposition (SVD)
    # This gives us the principal geometric axes of both shapes
    _, _, Vt_src = np.linalg.svd(src_centered, full_matrices=False)
    _, _, Vt_tgt = np.linalg.svd(tgt_centered, full_matrices=False)
    
    R_src = Vt_src.T
    R_tgt = Vt_tgt.T
    
    # 4. Handle PCA sign/flip ambiguity
    # Test all 8 possible sign flips for the axes to find the correct orientation
    best_error = float('inf')
    best_R = np.eye(3)
    
    # Build a KD-Tree of the target for rapid distance checking
    tgt_tree = cKDTree(tgt_centered)
    
    # Generate all sign combinations for 3D space: [[1,1,1], [1,1,-1], ...]
    sign_flips = np.array(np.meshgrid([1, -1], [1, -1], [1, -1])).T.reshape(-1, 3)
    
    for flip in sign_flips:
        # Apply the flip to the source axes
        R_src_flipped = R_src * flip
        
        # Candidate rotation matrix to map source axes to target axes
        R_cand = R_tgt @ R_src_flipped.T
        
        # Enforce a valid rotation matrix (determinant must be +1)
        if np.linalg.det(R_cand) < 0:
            continue
            
        # Test how close the points match under this rotation
        test_pts = src_centered @ R_cand.T
        distances, _ = tgt_tree.query(test_pts, k=1)
        mean_error = np.mean(distances)
        
        if mean_error < best_error:
            best_error = mean_error
            best_R = R_cand

    # 5. Build final 4x4 Transformation Matrix
    # T maps: src -> center -> rotate -> translate to target
    T = np.eye(4)
    T[:3, :3] = best_R
    T[:3, 3] = tgt_center - best_R @ src_center
    
    return T


import open3d as o3d

def align_meshes_ransac_icp(pred_mesh, dtc_mesh, voxel_size=0.05):
   
    source_mesh = unwrap_mesh(pred_mesh)
    target_mesh = unwrap_mesh(dtc_mesh)
    # 1. Convert Trimesh structures into Open3D Point Clouds
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_mesh.vertices)
    
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_mesh.vertices)
    
    # scale both
    # source_extent = source.get_max_bound() - source.get_min_bound()
    # scale_factor = max(source_extent) 
    # target.scale(scale_factor, center=(0, 0, 0))
    
    
    # 2. Downsample and estimate geometric surface normals
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)
    
    source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    # 3. Extract FPFH local shape descriptors
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))

    # 4. Run RANSAC Global Registration (The global rough alignment step)
    distance_threshold = voxel_size * 1.5
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, False, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(True), 3, # False
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.3), # 0.9
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

    rough_alignment = result_ransac.transformation

    # 5. Run Point-to-Plane ICP to fine-tune the final position
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, voxel_size * 0.4, rough_alignment,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()) # False

 
    return result_icp.transformation # 4 x 4 matrix 



from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


def align_meshes_similarity(
    pred_mesh,
    dtc_mesh,
    n_samples=10000,
):
    """
    Align pred_mesh to gt_mesh.

    Returns
    -------
    T : (4,4)
        Similarity transform matrix mapping
        pred -> gt

    scale : float
        Uniform scale factor
    """

    # --------------------------------------------------
    # Sample surfaces
    # --------------------------------------------------
    pred_mesh = unwrap_mesh(pred_mesh)
    gt_mesh = unwrap_mesh(dtc_mesh)
    src, _ = trimesh.sample.sample_surface(pred_mesh, n_samples)
    tgt, _ = trimesh.sample.sample_surface(gt_mesh, n_samples)

    # --------------------------------------------------
    # Initial centering
    # --------------------------------------------------
    src_center = src.mean(axis=0)
    tgt_center = tgt.mean(axis=0)

    src0 = src - src_center
    tgt0 = tgt - tgt_center

    # --------------------------------------------------
    # Initial scale estimate
    # --------------------------------------------------
    src_radius = np.sqrt(np.mean(np.sum(src0**2, axis=1)))
    tgt_radius = np.sqrt(np.mean(np.sum(tgt0**2, axis=1)))

    init_scale = tgt_radius / src_radius

    tgt_tree = cKDTree(tgt)

    # --------------------------------------------------
    # Chamfer objective
    # params:
    # [rx ry rz tx ty tz log_scale]
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Initialization
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Build homogeneous transform
    # x_gt = scale * R @ x_pred + t
    # --------------------------------------------------
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
    """
    Bake OpenCV->OpenGL flips, object pose, and world scale directly into
    extrinsic and intrinsic matrices so the renderer can use its own projection.

    Args:
        M_cv: [4,4] OpenCV world->camera matrix
        intr_norm: [3,3] normalized intrinsics (fx/w, fy/h, cx/w, cy/h)
        object_pose: [4,4] object->world matrix or None
        scale: world scale factor (same as nvdiffrecmc)

    Returns:
        baked_extr: [4,4] world->camera matrix (OpenGL-ready)
        baked_intr: [3,3] baked intrinsics matching renderer's projection
    """

    device = M_cv.device
    dtype = M_cv.dtype

    # -------------------------------
    # 1. Start with world->camera (OpenCV)
    # -------------------------------
    extr = M_cv.clone()

    # -------------------------------
    # 2. Apply object pose (object->world)
    # -------------------------------
    if object_pose is not None:
        extr = extr @ object_pose.to(device=device, dtype=dtype)

    # -------------------------------
    # 3. Apply OpenCV -> OpenGL axis flip
    # -------------------------------
    flip_zy = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=dtype, device=device))
    extr = flip_zy @ extr

    # -------------------------------
    # 4. Apply object coordinate flip (same as nvdiffrecmc)
    # -------------------------------
    flip_obj = torch.diag(torch.tensor([1.0, -1.0, 1.0, 1.0], dtype=dtype, device=device))
    extr = extr @ flip_obj

    # -------------------------------
    # 5. Apply world scale
    # -------------------------------
    S = torch.diag(torch.tensor([1.0/scale, 1.0/scale, 1.0/scale, 1.0], dtype=dtype, device=device))
    extr = extr @ S

    # -------------------------------
    # 6. Bake projection sign conventions into intrinsics
    #    (so renderer's projection matches nvdiffrecmc)
    # -------------------------------
    baked_intr = intr_norm.clone()

    # Y axis must be flipped (nvdiffrecmc uses negative Y)
    baked_intr[1, 1] *= -1.0

    # Depth sign baked into extrinsic (OpenGL expects -Z forward)
    # No change needed here; extr already has correct flips.

    return extr, baked_intr

# def bake_extrinsic_intrinsic(
#     M_cv: torch.Tensor,
#     intr_norm: torch.Tensor,
#     object_pose: torch.Tensor = None,
#     scale: float = 1.0,
# ):
#     """
#     Bake extrinsic/intrinsic for a renderer that expects camera->world extrinsics.

#     Args:
#         M_cv: [4,4] OpenCV world->camera matrix (cfg['frames'][idx]['transform_matrix'])
#         intr_norm: [3,3] normalized intrinsics (fx/w, fy/h, cx/w, cy/h)
#         object_pose: [4,4] object->world matrix or None
#         scale: world scale factor

#     Returns:
#         extr_baked: [4,4] camera->world matrix
#         intr_baked: [3,3] intrinsics (unchanged)
#     """

#     device = M_cv.device
#     dtype = M_cv.dtype

#     # 1. Start from world->camera (OpenCV)
#     W2C = M_cv

#     # 2. Convert to camera->world
#     C2W = torch.linalg.inv(W2C)

#     # 3. Apply world scale (same effect as scaling mv in nvdiffrecmc)
#     S = torch.diag(torch.tensor([scale, scale, scale, 1.0], dtype=dtype, device=device))
#     C2W = C2W @ S

#     # 4. If object_pose is given, assume mesh is in object space and
#     #    bake object->world into extrinsic as camera->object:
#     if object_pose is not None:
#         O2W = object_pose.to(device=device, dtype=dtype)
#         W2O = torch.linalg.inv(O2W)
#         C2O = C2W @ W2O
#         extr_baked = C2O
#     else:
#         extr_baked = C2W

#     intr_baked = intr_norm.clone()
#     return extr_baked, intr_baked

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




ndc_min = -1
            ndc_max = 1
            height = int(imgs_sizes[0])
            width = int(imgs_sizes[1])

            cams_persp4x4 = cams_intr4x4.clone()
            cams_persp4x4[:, 0, 2] = -cams_persp4x4[:, 0, 2]
            cams_persp4x4[:, 1, 2] = -cams_persp4x4[:, 1, 2]
            cams_persp4x4[:, 2, 2] = 0.0
            cams_persp4x4[:, 2, 3] = 1.0
            cams_persp4x4[:, 3, 2] = 1.0
            cams_persp4x4[:, 3, 3] = 0.0

            cams_persp4x4[:, 1, 2] = -(height + cams_persp4x4[:, 1, 2])
            # top = height / 2
            # bottom = -top
            # right = width / 2
            # left = -right

            top = 0
            bottom = max(height, 1)  # note: ensures no division by zero
            right = max(width, 1)  # note: ensures no division by zero
            left = 0

            zfar = 10000  # (cams_tform4x4_obj[..., 2, 3].max()) * 10.

            tx = -(right + left) / (right - left)
            ty = -(top + bottom) / (top - bottom)

            if ndc_min == -1 and ndc_max == 1:
                U = -2.0 * znear * zfar / (zfar - znear)
                V = -(zfar + znear) / (zfar - znear)
            elif ndc_min == 0 and ndc_max == 1:
                U = (znear * zfar) / (znear - zfar)
                V = zfar / (zfar - znear)
            elif ndc_min == 1 and ndc_max == 0:
                U = (znear * zfar) / (zfar - znear)
                V = znear / (zfar - znear)
            else:
                raise NotImplementedError(
                    "Perspective Projection does not support NDC range of "
                    f"[{ndc_min}, {ndc_max}]",
                )

            # The matrix is non differentiable, as NDC coordinates are a fixed standard set by the graphics api
            ndc_mat = torch.Tensor(
                [
                    [2.0 / (right - left), 0.0, 0.0, -tx],
                    [0.0, 2.0 / (top - bottom), 0.0, -ty],
                    [0.0, 0.0, U, V],
                    [0.0, 0.0, 0.0, -1.0],
                ],
            )[
                None,
            ].to(
                device=device,
            )

            # Squash matrices together to form complete perspective projection matrix which maps to NDC coordinates
            #
            from od3d.cv.visual.show import OPEN3D_CAM_TFORM_CAM

            cams_proj4x4 = (ndc_mat @ cams_persp4x4) @ (
                    OPEN3D_CAM_TFORM_CAM.clone()[None,].to(device=device)
                    @ cams_tform4x4_obj
            )

            verts, verts_counts = self.get_vert_mod_from_objs(
                mod=VERT_MODALITIES.PT3D,
                objs_ids=objects_ids,
                instance_deform=instance_deform,
                padded=True,
            )
            faces, faces_counts = self.get_face_mod_from_objs(
                mod=FACE_MODALITIES.VERTS_IN_SCENES_ID,
                objs_ids=objects_ids,
                padded=True,
            )
            from od3d.cv.select import cumsum_w0

            if objects_ids.dim() == 1:
                scenes_faces_counts = faces_counts
            elif objects_ids.dim() == 2:
                scenes_faces_counts = faces_counts.sum(dim=-1)
            else:
                raise NotImplementedError

            scenes_faces_counts_cumsum_w0 = cumsum_w0(scenes_faces_counts, dim=-1)

            ranges = torch.stack(
                [scenes_faces_counts_cumsum_w0, scenes_faces_counts],
                dim=1,
            )

            if objects_ids.dim() == 1:
                # 1x None for verts
                if obj_tform4x4_objs is not None:
                    cams_proj4x4 = cams_proj4x4[:, None] @ obj_tform4x4_objs
                else:
                    cams_proj4x4 = cams_proj4x4[:, None]
                verts_cam = (cams_proj4x4 @ add_homog_dim(verts, dim=-1)[..., None])[
                    ...,
                    0,
                ]
            elif objects_ids.dim() == 2:
                # 1x None for objects and 1x None for verts
                if obj_tform4x4_objs is not None:
                    if obj_tform4x4_objs.dim() == 3:
                        cams_proj4x4 = (
                            cams_proj4x4[:, None, None]
                            @ obj_tform4x4_objs[None, :, None]
                        )
                        logger.info(
                            "should not happen that obj_tform4x4_objs.dim() == 3",
                        )
                    elif obj_tform4x4_objs.dim() == 4:
                        cams_proj4x4 = (
                            cams_proj4x4[:, None, None] @ obj_tform4x4_objs[:, :, None]
                        )
                    else:
                        msg = f"not implemented for obj_tform4x4_objs.dim(): {obj_tform4x4_objs.dim()}"
                        raise NotImplementedError(msg)
                else:
                    cams_proj4x4 = cams_proj4x4[:, None, None]
                verts_cam = (cams_proj4x4 @ add_homog_dim(verts, dim=-1)[..., None])[
                    ...,
                    0,
                ]

            verts_cam = self.unpad_mod(objs_mod=verts_cam, objs_lengths=verts_counts)
            faces = self.unpad_mod(objs_mod=faces, objs_lengths=faces_counts)

            ranges = ranges.to(torch.int32)
            faces = faces.to(torch.int32)

            # verts_cam_new = verts_cam.clone()
            # faces_new = faces.clone()
            # ranges_new = ranges.clone()
            # TODO: doublecheck if the right faces are put together in the ranges?
            #
            # verts = self.get_verts_stacked_with_mesh_ids(mesh_ids=objects_ids[0], instance_deform=instance_deform).clone()
            # if detach_objects:
            #     verts = verts.detach()
            # if instance_deform is not None:
            #     for b, object_id in enumerate(objects_ids[0]):
            #         if not detach_deform:
            #             verts[b] += instance_deform.verts_deform[b, :len(verts[b])]
            #         else:
            #             verts[b] += instance_deform.verts_deform[b, :len(verts[b])].detach()
            #
            # verts_cam = (cams_proj4x4[:, None,] @ add_homog_dim(verts, dim=-1)[..., None])[..., 0]
            # faces = []
            # faces_count = 0
            # verts_count = 0
            # verts_count_acc_from_0 = []
            # ranges = []
            # for b, object_id in enumerate(objects_ids[0]):
            #     _faces = self.get_faces_with_mesh_id(object_id, clone=True)
            #     _faces += verts_count
            #     verts_count_acc_from_0.append(verts_count)
            #     faces.append(_faces)
            #     ranges.append(torch.LongTensor([faces_count, len(_faces)]))
            #     verts_count += len(verts_cam[b])
            #     faces_count += len(_faces)
            #
            # ranges = torch.stack(ranges, dim=0).to(device='cpu', dtype=torch.int32)
            # faces = torch.cat(faces, dim=0).to(device=device, dtype=torch.int32)
            # verts_cam = verts_cam.reshape(-1, 4)

            use_cuda = "cuda" in str(
                device,
            )  #  dr.RasterizeGLContext() if not use_cuda else
            self.glctx = (
                dr.RasterizeGLContext()
                if not use_cuda
                else dr.RasterizeCudaContext(device=device)
            )
            rast_out, _ = dr.rasterize(
                self.glctx,
                verts_cam,
                faces,
                ranges=ranges,
                resolution=[int(imgs_sizes[0]), int(imgs_sizes[1])],
            )

  # extr = torch.inverse(extr) @ cv_to_gl
            # extr = torch.eye(4, device=device)
            
            # modify the mesh vertices, advice from Leo
            # mesh_verts = transf3d_broadcast(
            #     pts3d=mesh.vertices.clone(),
            #     transf4x4=extr,
            # )
            # ext_buf = extr.clone()
            # transl = -(mesh_verts.max(dim=0)[0] + mesh_verts.min(dim=0)[0]) / 2.0
            # ext_buf[:3, 3] += transl
            # mesh_verts = transf3d_broadcast(
            #     pts3d=mesh.vertices.clone(),
            #     transf4x4=ext_buf,
            # )
            # scale = 1.0 / mesh_verts.abs().max()
            # ext_buf[:3, :] *= scale
            # extr = ext_buf
            # mesh_verts = transf3d_broadcast(pts3d=mesh.vertices.clone(), transf4x4=extr)
            
            # mesh.vertices = mesh_verts
            
            #         mesh_verts = mesh.vertices.clone()
#         verts = mesh_verts  # (N, 3)

# # Convert to homogeneous coordinates
#         ones = torch.ones((verts.shape[0], 1), device=verts.device, dtype=verts.dtype)
#         verts_h = torch.cat([verts, ones], dim=1)  # (N, 4)

# # Apply object pose
#         verts_world_h = (object_pose @ verts_h.T).T  # (N, 4)

# # Drop the homogeneous coordinate
#         vertices_world = verts_world_h[:, :3]
#         mesh.vertices = vertices_world.contiguous() 

'''

'''
        mesh = Meshes.read_from_ply_file(
                fpath=self.get_fpath_mesh(),
                device=device,
                scale=scale,
            )

            mesh_verts = transf3d_broadcast(
                pts3d=mesh.verts.clone(),
                transf4x4=tform_obj,
            )
            tform_obj_buf = tform_obj.clone()
            transl = -(mesh_verts.max(dim=0)[0] + mesh_verts.min(dim=0)[0]) / 2.0
            tform_obj_buf[:3, 3] += transl

            mesh_verts = transf3d_broadcast(
                pts3d=mesh.verts.clone(),
                transf4x4=tform_obj_buf,
            )
            scale = 1.0 / mesh_verts.abs().max()
            tform_obj_buf[:3, :] *= scale

            # mesh_verts = transf3d_broadcast(pts3d=mesh.verts.clone(), transf4x4=tform_obj_buf)
            tform_obj = tform_obj_buf
        
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


###############################3333333
optimizer = torch.optim.Adam([h_feats], lr=0.01)  
  
# Create decoder in train mode  
decoder = FlexiDualGridVaeDecoder(...)  
decoder.train()  
  
for iteration in range(num_iterations):  
    optimizer.zero_grad()  
      
    # Forward pass through decoder  
    vertices = h_opt.replace((1 + 2 * decoder.voxel_margin) *   
                           F.sigmoid(h_opt.feats[..., 0:3]) - decoder.voxel_margin)  
    intersected_logits = h_opt.replace(h_opt.feats[..., 3:6])  
    quad_lerp = h_opt.replace(F.softplus(h_opt.feats[..., 6:7]))  
      
    # Convert to mesh (this is differentiable for vertex positions)  
    mesh = Mesh(*flexible_dual_grid_to_mesh(  
        h_opt.coords[:, 1:],   
        vertices.feats,   
        intersected_logits.feats,   
        quad_lerp.feats,  
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],  
        grid_size=decoder.resolution,  
        train=True  
    ))  
      
    # Render and compute loss  
    rendered = renderer.render(mesh, camera_params)  
    loss = compute_loss(rendered, target)  
      
    # Backpropagate  
    loss.backward()  
    optimizer.step()  
      
    print(f"Iter {iteration}: Loss = {loss.item():.6f}")

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