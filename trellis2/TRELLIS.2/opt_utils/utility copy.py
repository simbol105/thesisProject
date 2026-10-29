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

# Image save/load helper.

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
    """Optimal rotation R (3x3) aligning corresponding points A -> B, both (N, 3) (Kabsch)."""
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


def normalize(mesh):
    mesh = mesh.copy()
    mesh.apply_translation(-mesh.centroid)
    scale = 1.0 / mesh.scale
    mesh.apply_scale(scale)
    return mesh

    
def compute_icp_alignment_transform(dtc_mesh, pred_mesh, simplify_target=50000, icp_iterations=100):
    """Compute precise ICP-based alignment transformation from predicted mesh to ground truth mesh."""
    device = pred_mesh.device if hasattr(pred_mesh, 'device') else torch.device('cpu')
    
    # Unwrap mesh objects
    dtc_trimesh = unwrap_mesh(dtc_mesh)
    pred_trimesh = unwrap_mesh(pred_mesh)
    
    # Simplify predicted mesh to reduce computation
    pred_trimesh = pred_trimesh.copy()
    if len(pred_trimesh.vertices) > simplify_target:
        pred_trimesh.simplify_mesh(target_count=simplify_target, aggressiveness=7.0, verbose=False)
    
    # Compute centroids and bounding box diagonals
    dtc_centroid = dtc_trimesh.bounds.mean(axis=0)
    pred_centroid = pred_trimesh.bounds.mean(axis=0)
    
    dtc_diag = np.linalg.norm(dtc_trimesh.bounds[1] - dtc_trimesh.bounds[0])
    pred_diag = np.linalg.norm(pred_trimesh.bounds[1] - pred_trimesh.bounds[0])
    
    # Scale factor to match bounding box sizes
    scale = dtc_diag / (pred_diag + 1e-6)
    
    # Normalize both meshes to [−0.5, 0.5]
    dtc_norm = dtc_trimesh.copy()
    dtc_norm.apply_translation(-dtc_centroid)
    dtc_norm.apply_scale(1.0 / (dtc_diag + 1e-6))
    
    pred_norm = pred_trimesh.copy()
    pred_norm.apply_translation(-pred_centroid)
    pred_norm.apply_scale(1.0 / (pred_diag + 1e-6))
    
    # Coarse alignment using mesh_other (fast registration)
    try:
        T_coarse, _ = trimesh.registration.mesh_other(
            dtc_norm, pred_norm,
            scale=False,
            icp_first=10,
            samples=5000
        )
    except Exception as e:
        print(f"Coarse alignment failed: {e}. Using identity matrix.")
        T_coarse = np.eye(4)
    
    # Fine alignment using ICP
    try:
        T_icp, _, _ = trimesh.registration.icp(
            dtc_norm.vertices,
            pred_norm.vertices,
            initial=T_coarse,
            scale=False,
            max_iterations=icp_iterations,
            threshold=1e-5
        )
    except Exception as e:
        print(f"ICP alignment failed: {e}. Using coarse alignment.")
        T_icp = T_coarse
    
    # Construct un-normalization transforms
    T_denorm_dtc = np.eye(4)
    T_denorm_dtc[:3, :3] = dtc_diag * np.eye(3)
    T_denorm_dtc[:3, 3] = dtc_centroid
    
    # T_denorm_pred: undo normalization of pred mesh (with scale)
    T_denorm_pred = np.eye(4)
    T_denorm_pred[:3, :3] = pred_diag * np.eye(3)  # Note: use pred_diag before scaling
    T_denorm_pred[:3, 3] = pred_centroid
    
    # Scale transform
    T_scale = np.eye(4)
    T_scale[:3, :3] *= scale
    
    T_denorm_pred_inv = np.linalg.inv(T_denorm_pred)
    T_final = T_denorm_dtc @ T_scale @ T_icp @ T_denorm_pred_inv
    
    return torch.tensor(T_final, dtype=torch.float32, device=device)


# Legacy function kept for backward compatibility
def get_normalization_matrix(dtc_mesh, pred_mesh):
    """DEPRECATED: Use compute_icp_alignment_transform instead."""
    return compute_icp_alignment_transform(dtc_mesh, pred_mesh)
 

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


# Get initial structure from DmTeT

