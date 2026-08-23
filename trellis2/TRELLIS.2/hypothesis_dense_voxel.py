import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["SPARSE_CONV_BACKEND"] = "spconv"

# from trellis2.modules.sparse import config  
# config.set_conv_backend('torchsparse')

# os.environ["ATTN_BACKEND"] = 'xformers'

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
# from torchmetrics.functional import structural_similarity_index_measure as ssim
import torch.nn as nn
import yaml
from PIL import Image
import imageio
import trimesh

import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.renderers import EnvMap
from trellis2.renderers.pbr_mesh_renderer import PbrMeshRenderer
from trellis2.renderers.mesh_renderer import MeshRenderer
from trellis2.representations import MeshWithVoxel, Mesh
from trellis2.utils import render_utils
from trellis2.modules.sparse import SparseTensor
from opt_utils.utility import display_image, save_image, transf3d_broadcast, transf3d, align_meshes_similarity, bake_extrinsic_intrinsic
from opt_utils.point_renderer import ovoxel_to_pointcloud, PointRenderer
import o_voxel
from o_voxel import postprocess


GENERATED_MESH_PATH = Path("/home/fr/fr_fr/fr_ba122/projects/recgen/out/mesh.obj")
RES = 64 # ovoxel resolution. 
TARGET_NUM_VERTICES = 30000
VOXEL_MARGIN = 0.5  

TEST_INDEX = 4
TEST_INTERVALS = [0, 1, 5, 10, 15, 20, 30, 35, 40, 45, 50] #  50, 100, 200,  500, 799, 999]

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

def load_image(path):
    image = Image.open(path)
    image = np.array(image)
    image = image.astype(np.uint8)
    return image

AXIS_SWAP = rot_z(np.pi/2)
# AXIS_SWAP = np.eye(4, dtype=np.float32)


def _load_json(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def _try_image_path(base_path_no_ext: Path) -> Path:
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        cand = base_path_no_ext.with_suffix(ext)
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Could not find image for: {base_path_no_ext}")


def _resize_rgb_to_tensor(img_path: Path, render_resolution: int, device: torch.device) -> torch.Tensor:
    img = Image.open(img_path).convert("RGBA")
    if img.size != (render_resolution, render_resolution):
        img = img.resize((render_resolution, render_resolution), Image.Resampling.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0  
    t = torch.from_numpy(arr).to(device=device, dtype=torch.float32)  
    t = t.permute(2, 0, 1).contiguous()  
    return t


def _scale_opencv_intrinsics(intr: dict, orig_wh: tuple[int, int]) -> torch.Tensor:
    fx, fy, cx, cy = float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"])
    orig_w, orig_h = orig_wh

    # Normalize parameters strictly to the [0, 1] range expected by TRELLIS2 intrinsics_to_projection
    fx_n = fx / float(orig_w)
    fy_n = fy / float(orig_h)
    cx_n = cx / float(orig_w)
    cy_n = cy / float(orig_h)

    return torch.tensor(
        [[fx_n, 0.0, cx_n], 
         [0.0, fy_n, cy_n], 
         [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


def scale_opencv_intrinsics_for_render(
    intr: dict,
    orig_wh: tuple[int, int],
    render_resolution: int,
) -> torch.Tensor:
    fx, fy, cx, cy = float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"])
    orig_w, orig_h = orig_wh

    # Assume square render: (render_resolution, render_resolution)
    render_w = render_resolution
    render_h = render_resolution

    # Scale intrinsics from original size to render size
    sx = render_w / orig_w
    sy = render_h / orig_h

    fx_r = fx * sx
    fy_r = fy * sy
    cx_r = cx * sx
    cy_r = cy * sy

    # Normalize to [0,1] for intrinsics_to_projection
    fx_n = fx_r / render_w
    fy_n = fy_r / render_h
    cx_n = cx_r / render_w
    cy_n = cy_r / render_h

    intr_normalized = torch.tensor(
        [[fx_n, 0.0, cx_n],
         [0.0, fy_n, cy_n],
         [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    return intr_normalized



def decode_vertex_features(feats: torch.Tensor, voxel_margin: float) -> torch.Tensor:
    return torch.tanh((feats - 0.5) * 2.0) * voxel_margin


@torch.no_grad()
def _freeze_module_params(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)

def freeze_pipeline(pipeline):
    for name, module in pipeline.models.items():
        if isinstance(module, torch.nn.Module):
            for p in module.parameters():
                p.requires_grad_(False)


def get_mesh_from_ovoxel(coords, attrs, resolution=512):
     
    # Extract geometry data from O-Voxel  
    dual_vertices = attrs['dual_vertices'] / 255.0  
    intersected = torch.cat([  
        attrs['intersected'] % 2,  
        attrs['intersected'] // 2 % 2,  
        attrs['intersected'] // 4 % 2,  
    ], dim=-1).bool()  
     
    vertices, faces = o_voxel.convert.flexible_dual_grid_to_mesh(  
        coords.cuda(),  
        dual_vertices.cuda(),  
        intersected.cuda(),  
        grid_size=resolution,  
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]  
    )   
    attr_tensor = torch.cat([attrs[k] for k in ['base_color', 'metallic', 'roughness', 'alpha']], dim=-1) / 255.0  
    
    mesh_with_voxel = MeshWithVoxel(  
        vertices=vertices.cpu(),  
        faces=faces.cpu(),  
        origin=[-0.5, -0.5, -0.5],  
        voxel_size=1/resolution,  
        coords=coords,  
        attrs=attr_tensor,  
        voxel_shape=torch.Size([1, resolution, resolution, resolution]),  
        layout={  
            'base_color': slice(0, 3),  
            'metallic': slice(3, 4),  
            'roughness': slice(4, 5),  
            'alpha': slice(5, 6)  
        }  
    )
    return mesh_with_voxel

def _decode_latent_differentiable(
    pipeline: Trellis2ImageTo3DPipeline,
    shape_slat_opt,
    tex_slat_opt,
    resolution: int,
) -> List[MeshWithVoxel]:
    """
    Decode shape and texture latents to mesh with voxel attributes.
    This is fully differentiable for gradient backpropagation.
    """
   
    assert shape_slat_opt.coords.shape[0] > 0
    assert not torch.isnan(shape_slat_opt.coords).any()
    assert not torch.isnan(shape_slat_opt.feats).any()

    meshes, subs = pipeline.decode_shape_slat(shape_slat_opt, resolution)
    tex_voxels = pipeline.decode_tex_slat(tex_slat_opt, subs)

    out_mesh: List[MeshWithVoxel] = []
    for m, v in zip(meshes, tex_voxels):

        out_mesh.append(
            MeshWithVoxel(
                m.vertices,
                m.faces,
                origin=[-0.5, -0.5, -0.5],
                voxel_size=1.0 / resolution,
                coords=v.coords[:, 1:],
                attrs=v.feats,
                voxel_shape=torch.Size([*v.shape, *v.spatial_shape]),
                layout=pipeline.pbr_attr_layout,
            )
        )
    return out_mesh

def _load_json(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def downsample_tensors(a: torch.Tensor, b: torch.Tensor):

    a = a.flatten().float()
    b = b.flatten().float()

    if a.numel() == b.numel():
        return a, b

    if a.numel() > b.numel():
        big, small = a, b
        replace_small = "b"
    else:
        big, small = b, a
        replace_small = "a"

    if small.numel() == 0:
        return a, b

    small_reshaped = small.view(1, 1, -1)
    small_up = F.interpolate(small_reshaped, size=big.numel(), mode="linear", align_corners=False)
    small_up = small_up.view(-1)

    if replace_small == "a":
        a = small_up
    else:
        b = small_up

    return a, b

# def match_sparse_voxels(pred_coords, pred_feats, gt_coords, gt_feats):
#     """
#     pred_coords: [N_pred, 3] int voxel coords
#     pred_feats:  [N_pred, C] float features (CUDA)
#     gt_coords:   [N_gt, 3] int voxel coords
#     gt_feats:    [N_gt, C] float GT features (CPU or CUDA)

#     Returns:
#         pred_matched: [K, C]
#         gt_matched:   [K, C]
#         pred_only_mask: [N_pred] boolean
#         gt_only_mask:   [N_gt] boolean
#     """

#     device = pred_feats.device

#     # --- Ensure EVERYTHING is on the same device ---
#     pred_coords = pred_coords.to(device)
#     gt_coords   = gt_coords.to(device)
#     gt_feats    = gt_feats.to(device)

#     # Convert coords to linear indices
#     pred_lin = o_voxel.serialize.encode_seq(pred_coords)   # [N_pred]
#     gt_lin   = o_voxel.serialize.encode_seq(gt_coords)     # [N_gt]

#     # Sort both
#     pred_sort = torch.argsort(pred_lin)
#     gt_sort   = torch.argsort(gt_lin)

#     pred_lin_sorted = pred_lin[pred_sort]
#     gt_lin_sorted   = gt_lin[gt_sort]

#     # Broadcasting equality (device-safe)
#     eq_matrix = pred_lin_sorted[:, None] == gt_lin_sorted[None, :]

#     # Find matched pairs
#     pred_idx, gt_idx = torch.nonzero(eq_matrix, as_tuple=True)

#     # Matched features
#     pred_matched = pred_feats[pred_sort[pred_idx]]
#     gt_matched   = gt_feats[gt_sort[gt_idx]]

#     # Voxels present only in prediction
#     pred_only_mask = torch.ones(pred_coords.shape[0], dtype=torch.bool, device=device)
#     pred_only_mask[pred_sort[pred_idx]] = False

#     # Voxels present only in GT
#     gt_only_mask = torch.ones(gt_coords.shape[0], dtype=torch.bool, device=device)
#     gt_only_mask[gt_sort[gt_idx]] = False

#     return pred_matched, gt_matched, pred_only_mask, gt_only_mask

# def match_sparse_voxels_nn(pred_coords, pred_feats, gt_coords, gt_feats):
#     device = pred_feats.device

#     pred_coords = pred_coords.to(device).float()
#     gt_coords   = gt_coords.to(device).float()
#     gt_feats    = gt_feats.to(device)

#     # Compute pairwise distances: [N_gt, N_pred]
#     dists = torch.cdist(gt_coords, pred_coords)

#     # For each GT voxel, find nearest predicted voxel
#     nn_pred_idx = torch.argmin(dists, dim=1)

#     pred_matched = pred_feats[nn_pred_idx]
#     gt_matched   = gt_feats

#     return pred_matched, gt_matched

def build_pred_grid(pred_coords, RES):
    grid = -torch.ones((RES, RES, RES), dtype=torch.long, device=pred_coords.device)
    x = pred_coords[:,0].long()
    y = pred_coords[:,1].long()
    z = pred_coords[:,2].long()
    grid[x, y, z] = torch.arange(pred_coords.shape[0], device=pred_coords.device)
    return grid


from sklearn.neighbors import KDTree

def match_sparse_voxels_nn(pred_coords, pred_feats, gt_coords, gt_feats):
    device = pred_feats.device

    # Move coords to CPU for KDTree (KDTree is CPU-only)
    pred_coords_cpu = pred_coords.detach().cpu().numpy()
    gt_coords_cpu   = gt_coords.detach().cpu().numpy()

    # Build KD-tree on predicted voxel coords
    tree = KDTree(pred_coords_cpu)

    # Query nearest predicted voxel for each GT voxel
    dist, nn_pred_idx_cpu = tree.query(gt_coords_cpu, k=1)

    # Convert indices back to torch
    nn_pred_idx = torch.from_numpy(nn_pred_idx_cpu.squeeze()).to(device)

    # Gather matched features
    pred_matched = pred_feats[nn_pred_idx]
    gt_matched   = gt_feats.to(device)

    return pred_matched, gt_matched






def pad_sparse_features(gt_features, pred_features):
    """Pad the smaller dense tensor to match the larger one."""
    if gt_features.shape == pred_features.shape:
        return gt_features, pred_features

    if len(gt_features.shape) != len(pred_features.shape):
        raise ValueError(
            f"Cannot pad tensors with different ranks: {gt_features.shape} vs {pred_features.shape}"
        )

    target_shape = tuple(max(a, b) for a, b in zip(gt_features.shape, pred_features.shape))

    def pad_to_shape(tensor: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        if tensor.shape == shape:
            return tensor
        out = tensor.new_zeros(shape)
        slices = tuple(slice(0, dim) for dim in tensor.shape)
        out[slices] = tensor
        return out

    return pad_to_shape(gt_features, target_shape), pad_to_shape(pred_features, target_shape)


def assign_vertex_colors(mesh_vertices, voxel_coords, voxel_colors, RES):
    # mesh_vertices: [N_vertices, 3]
    # voxel_coords:  [N_voxels, 3]
    # voxel_colors:  [N_voxels, 3]

    # Convert voxel coords to world positions
    voxel_pos = voxel_coords.float() / RES - 0.5  # same coordinate system as mesh

    # Compute nearest voxel for each vertex
    dists = torch.cdist(mesh_vertices, voxel_pos)
    nn_idx = torch.argmin(dists, dim=1)

    # Assign voxel color to vertex
    vertex_colors = voxel_colors[nn_idx]
    return vertex_colors

def build_dense_color_grid(voxel_coords, voxel_colors, RES):
    # voxel_colors: [N_voxels, 3]
    # voxel_coords: [N_voxels, 3] integer coords in [0, RES)

    device = voxel_colors.device

    dense = torch.zeros(1, 3, RES, RES, RES, device=device)

    # Fill dense grid
    x = voxel_coords[:, 0].long()
    y = voxel_coords[:, 1].long()
    z = voxel_coords[:, 2].long()

    dense[0, :, x, y, z] = voxel_colors.T
    return dense

def sample_vertex_colors(dense_grid, mesh_vertices_norm):
    # dense_grid: [1, 3, RES, RES, RES]
    # mesh_vertices_norm: [N_vertices, 3] in [-1,1]

    # grid_sample expects shape [N, D, H, W, 3]
    coords = mesh_vertices_norm.view(1, -1, 1, 1, 3)

    sampled = torch.nn.functional.grid_sample(
        dense_grid,
        coords,
        mode='bilinear',
        align_corners=False
    )

    # sampled: [1, 3, N_vertices, 1, 1]
    vertex_colors = sampled.squeeze().T  # → [N_vertices, 3]
    return vertex_colors


def normalize_vertices(mesh_vertices):
    # mesh_vertices: [N_vertices, 3] in [-0.5, 0.5]
    verts = mesh_vertices.clone()
    verts = (verts + 0.5) * 2 - 1   # map [-0.5,0.5] → [-1,1]
    return verts


# Main function
def refine_ovoxel(
    yaml_config_path: Path,
):
    config_yaml = yaml.safe_load(open(yaml_config_path, "r"))
    

    lr = float(config_yaml["lr"])
    num_iters = int(config_yaml["num_iters"])
    envmap_path = Path(config_yaml["envmap_path"])
    assert envmap_path.exists(), f"Envmap path does not exist: {envmap_path}"
    gt_mesh = Path(config_yaml.get("gt_mesh", ""))
    assert gt_mesh.exists(), f"Ground truth mesh path does not exist: {gt_mesh}"
    transforms_json_path = Path(config_yaml["transforms_json"])
    assert transforms_json_path.exists(), f"Transforms JSON path does not exist: {transforms_json_path}"
    output_glb_path = Path(config_yaml["output_path"])
   
    

    pipeline_type = str(config_yaml.get("pipeline_type", '512'))
    frames_per_iter = config_yaml.get("frames_per_iter", None)
    near = float(config_yaml.get("near", 0.01))
    far = float(config_yaml.get("far", 100.0))
    ssaa = int(config_yaml.get("ssaa", 1))
    lamda = float(config_yaml.get("lamda", 0.001))
    peel_layers = int(config_yaml.get("peel_layers", 8))
    loss_type = str(config_yaml.get("loss_type", "mse"))  # loss_fn
    latent_l2_weight = float(config_yaml.get("latent_l2_weight", 0.0))
    grad_clip_norm = float(config_yaml.get("grad_clip_norm", 0.0))  # 0 disables
    lambda_intersected = float(config_yaml.get("lambda_intersected", 0.1))
    lambda_vertice = float(config_yaml.get("lambda_vertice", 1e-2))
    lambda_geo = float(config_yaml.get("lambda_geo", 1e-1))
    lambda_mask = float(config_yaml.get("lambda_mask", 1.0))
    lambda_depth = float(config_yaml.get("lambda_depth", 10.0))
    lambda_normal = float(config_yaml.get("lambda_normal", 1.0))
    display_interval = int(config_yaml.get("display_interval", 20))
    out_dir = config_yaml.get("out_dir", "optimization_output")
    dmtet_res = int(config_yaml.get("dmtet_res", 64))
    test_out_dir = Path(out_dir) / "test"
    test_out_dir.mkdir(parents=True, exist_ok=True)

    test_transforms_json_path = Path(config_yaml.get("test_transforms_json", ""))
    assert test_transforms_json_path.exists(), f"Test Transforms JSON Path does not exist: {test_transforms_json_path}"
    test_transforms = _load_json(test_transforms_json_path)
    
    optimize_texture_only = bool(config_yaml.get("optimize_texture_only", False))  # Optimize texture latents only
    optimizer_type = str(config_yaml.get("optimizer_type", "adam")).lower()  # adam or sgd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transforms = _load_json(transforms_json_path)
    frames = transforms["frames"]
    base_object_pose = torch.tensor(transforms["object_pose"], dtype=torch.float32, device=device)
    base_dir = transforms_json_path.parent  
    image_basenames = [f["file_path"] for f in frames]
    image_paths: List[Path] = []
    for base_name in image_basenames:
        base_no_ext = base_dir / base_name
        image_paths.append(_try_image_path(base_no_ext))

    intr = transforms["intrinsic"]

    prompt_idx = int(config_yaml.get("prompt_idx", np.random.randint(0, len(image_paths))))
    # Load pipeline
    print("Loading TRELLIS.2 pipeline...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.low_vram = False
    pipeline.cuda()
    # freeze_pipeline(pipeline)  # Freeze pipeline weights, only optimize latents
    print("Pipeline loaded")
    
   
    
    # Run once on prompt image to extract latents
    print(f"Extracting latents from prompt image {prompt_idx}...")
    # prompt_img = Image.open(image_paths[prompt_idx]).convert("RGBA")
    
    # # Get initial mesh and latents from pipeline
    # mesh_ = pipeline.run(
    #     prompt_img,
    #     # return_latent=True,  # Request latents
    #     pipeline_type=pipeline_type,
    # )[0]
    
    mesh_ = trimesh.load_mesh(GENERATED_MESH_PATH)
    
    # mesh.visual = mesh.visual.to_color()
    mesh = Mesh(
        vertices=torch.from_numpy(np.array(mesh_.vertices)).float(),
        faces=torch.from_numpy(np.array(mesh_.faces)).long(),
        vertex_attrs=torch.from_numpy(np.array(mesh_.visual.vertex_colors)[:, :3] / 255.0).float()   
    )
    
    mesh.simplify(target = TARGET_NUM_VERTICES)
  
    
    vertices = torch.from_numpy(np.array(mesh.vertices)).float()
    faces = torch.from_numpy(np.array(mesh.faces)).float()
    print("Initial mesh vertices count after simplification: ", vertices.shape)
    print("Initial mesh faces count: \n", faces.shape)
    # convert to ovoxel features 
    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices=vertices,
        faces=faces,
        grid_size=[RES, RES, RES],  
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], 
        face_weight=1.0, 
        boundary_weight=0.2, 
        regularization_weight=1e-2, 
        timing=True
    )
    vid = o_voxel.serialize.encode_seq(voxel_indices)
    mapping = torch.argsort(vid)
    voxel_indices = voxel_indices[mapping]
    dual_vertices = dual_vertices[mapping]
    intersected = intersected[mapping]

    
    # # packing
    dual_vertices = dual_vertices * RES - voxel_indices
    dual_vertices = (torch.clamp(dual_vertices, 0, 1) * 255).type(torch.uint8)
    intersected = (intersected[:, 0:1] + 2 * intersected[:, 1:2] + 4 * intersected[:, 2:3]).type(torch.uint8)
    
    
    dual_vertices_opt = (dual_vertices.float() / 255.0).to(device)
    opt_intersected = torch.cat([
        intersected % 2,
        intersected // 2 % 2, 
        intersected // 4 % 2, 
    ], dim=-1).bool().to(device)
    print("Opt intersected shape: \n", opt_intersected.shape)
    voxel_margin = VOXEL_MARGIN
    quad_lerp = torch.nn.functional.softplus(torch.ones_like(dual_vertices[:, :1]) * 0.6).to(device)
   
    
    
    
    base_color = (torch.ones_like(dual_vertices_opt) * 0.5).to(device)
    coords_expanded = torch.cat([torch.zeros_like(voxel_indices[:, 0:1]), voxel_indices], dim=1).to(device)
    
    base_color_f = torch.nn.Parameter(base_color)
    intersected_f = torch.nn.Parameter(opt_intersected.float() / 7.0)
    quad_lerp_f = torch.nn.Parameter(quad_lerp)
    vertices_f = torch.nn.Parameter(dual_vertices_opt)

    vertices_sp = SparseTensor(
            feats=vertices_f,
            coords=coords_expanded, 
            shape=torch.Size([RES, RES, RES])
        )
    intersected_sp = vertices_sp.replace(
        intersected_f
    )
    # h_feats_init = torch.cat([
    #     dual_vertices_opt, # 0-3
    #     intersected_f, # 3-6
    #     quad_lerp, 
    #     base_color,
    #     # attributes_opt
    # ], dim=-1).to(device=device)
    # h_feats = torch.nn.Parameter(h_feats_init)
    
    
  
    optimizer = torch.optim.Adam([vertices_sp.feats, intersected_sp.feats, quad_lerp_f, base_color_f], lr = lr)
    
    renderer = MeshRenderer(
        rendering_options={
            "resolution": RES,
            "near": near,
            "far": far,
            "ssaa": ssaa,
        },
        device=device.type,
    )
    
    
    
    orig_img = Image.open(image_paths[prompt_idx])
    orig_w, orig_h = orig_img.size
    
    
    extrinsics = []
    for f in frames:
        extrinsics.append(torch.tensor(f["transform_matrix"], dtype=torch.float32))
        
    '''
    If we need to align mesh later, 
    It will be done here. 
    '''
    criterion = nn.MSELoss() 
    l1_criterion = nn.L1Loss(reduction='mean')
    extrinsics_t = torch.stack(extrinsics, dim=0).to(device=device)
    # ref_images   = torch.stack(
    #     [_resize_rgb_to_tensor(p, RES, device) for p in image_paths], dim=0
    # )
    
    N = len(image_paths)
    print(f"Loaded {N} reference images for optimization.")
    rng = np.random.default_rng(seed=int(config_yaml.get("seed", 0)))
    
    # optimization loop 
    
    intr_scaled = _scale_opencv_intrinsics(intr, (orig_w, orig_h)).to(device=device)

    
    # alignment computation. 
    mesh = Mesh( *o_voxel.convert.flexible_dual_grid_to_mesh(  
                coords=voxel_indices.to(device),   
                dual_vertices=dual_vertices_opt.to(device),   
                intersected_flag=opt_intersected.to(device),
                split_weight=None,  
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],  
                grid_size=RES,  
                train=False
            ), 
                    base_color.contiguous()
        )
    
    # NOTE: please provide gt mesh for rendering normal and mask
    dtc_mesh = trimesh.load_mesh(gt_mesh)
   
    dtc_mesh.visual = dtc_mesh.visual.to_color()
    dtc_mesh = Mesh(
        vertices=torch.from_numpy(np.array(dtc_mesh.vertices)).float(),
        faces=torch.from_numpy(np.array(dtc_mesh.faces)).long(),
        vertex_attrs=torch.from_numpy(np.array(dtc_mesh.visual.vertex_colors)[:, :3] / 255.0).float() 
    )
    # dtc_mesh.simplify(target = TARGET_NUM_VERTICES)
    print("GT mesh vertices count after simplification: ", dtc_mesh.vertices.shape)
    dtc_mesh.to(device=device)
    v = dtc_mesh.vertices
    f = dtc_mesh.faces
  
    if gt_mesh.exists():
        
        with torch.no_grad():
            # alignment = get_normalization_matrix(dtc_mesh, glb).to(device=device)
            alignment, _ = align_meshes_similarity(mesh, dtc_mesh) 
            alignment = torch.from_numpy(alignment @ AXIS_SWAP  ).to(device=device, dtype=torch.float32) # @ AXIS_SWAP
        
    else:
        print("No ground truth mesh provided, skipping alignment step.")
        alignment = torch.eye(4, device=device) 
    object_pose = base_object_pose @ alignment

    
    '''
    Derive the gt_intersected and vertices from the groundtruth for mesh optimization.
    '''
    gt_indices, gt_vertices, gt_intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices=v.detach().cpu(),
        faces=f.detach().cpu(),
        grid_size=[RES, RES, RES], 
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], 
        face_weight=1.0, 
        boundary_weight=0.2, 
        regularization_weight=1e-2, 
        timing=True
    )
    vid = o_voxel.serialize.encode_seq(gt_indices)
    mapping = torch.argsort(vid)
    gt_indices = gt_indices[mapping]
    gt_vertices = gt_vertices[mapping]
    gt_intersected = gt_intersected[mapping]

    gt_vertices = (gt_vertices.float() / 255.0).to(device)
    
    gt_intersected = (gt_intersected[:, 0:1] + 2 * gt_intersected[:, 1:2] + 4 * gt_intersected[:, 2:3]).type(torch.uint8)
    gt_opt_intersected = torch.cat([
            gt_intersected % 2,
            gt_intersected // 2 % 2, 
            gt_intersected// 4 % 2, 
        ], dim=-1).bool()
    # gt_color = torch.ones_like(gt_vertices) * 0.5; 
    # gt_quad_lerp = torch.nn.functional.softplus(torch.ones_like(gt_vertices[:, :1]) * 0.6)
    
    gt_coords_expanded = torch.cat([torch.zeros_like(gt_indices[:, 0:1]), gt_indices], dim=1).to(device=device)
    gt_vertices_sp = SparseTensor(
        feats=gt_vertices, 
        coords=gt_coords_expanded,
        shape=torch.Size([RES, RES, RES])
    )
    gt_intersected_sp = gt_vertices_sp.replace(
        gt_opt_intersected
    )
    gt_opt_intersected = gt_opt_intersected.to(device)
    gt_indices = gt_indices.to(device)
    gt_vertices = gt_vertices.to(device)


    # gt_h = SparseTensor(
    #     feats=gt_h_feats_init,
    #     coords=gt_coords_expanded,
    #     shape=torch.Size([1, gt_h_feats_init.shape[-1]]),
    #     spatial_shape=torch.Size([RES, RES, RES])
    # )
    
    # Get the dense representation
    # gt_vertices_dense = gt_vertices_sp.to_dense()
    # gt_intersected_dense = gt_intersected_sp.to_dense()


    print("Starting Optimization...")
    
    for i in range(num_iters + 1):
        indices = (
            np.arange(N)
            if frames_per_iter is None or frames_per_iter == N
            else np.atleast_1d(rng.choice(N, size=frames_per_iter, replace=False))
        )
        total_loss = torch.zeros((), device=device)
        
        optimizer.zero_grad()
        

      
        # vertices = decode_vertex_features(h_opt.feats[..., 0:3], voxel_margin)
        # intersected_logits = h_opt.feats[..., 3:6]
        # # intersected_flag = straight_through_threshold(intersected_logits).bool()
        # intersected_flag = intersected_logits > 0.2
        # color = torch.sigmoid(h_opt.feats[..., 7:10])
        # quad_lerp = F.softplus(h_opt.feats[..., 6:7])

        vertices_sp = vertices_sp.replace(feats=vertices_f)
        intersected_sp = intersected_sp.replace(feats=intersected_f)
        vertices = (1 + 2*voxel_margin) * F.sigmoid(vertices_sp.feats) - voxel_margin
        intersected_flag = torch.sigmoid(intersected_sp.feats) > 0.2
        color = torch.sigmoid(base_color_f)
        quad_lerp = F.softplus(quad_lerp_f)
        
        
        
        extracted_vertices, extracted_faces = o_voxel.convert.flexible_dual_grid_to_mesh(   
                vertices_sp.coords[:, 1:],  
                vertices,   
                # gt_intersected_dense,
                intersected_flag,
                quad_lerp, 
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],  
                grid_size=RES,  
                train=True
            )
        dense_grid = build_dense_color_grid(vertices_sp.coords[:,1:], color, RES)
        mesh_vertices_norm = normalize_vertices(extracted_vertices)
        vertex_colors = sample_vertex_colors(dense_grid, mesh_vertices_norm)

   

        
        # vertex_colors = assign_vertex_colors(
        #     mesh.vertices,
        #     vertices_sp.coords[:,1:],   # voxel coords
        #     color,                      # voxel colors
        #     RES
        # )

        # mesh = Mesh(mesh.vertices, mesh.faces, vertex_colors)

        
        # vertices, gt_vertices = downsample_tensors(vertices, gt_vertices)
        mesh = Mesh( extracted_vertices, extracted_faces,
                    vertex_attrs=vertex_colors.contiguous()
        )
        # loss_geom_anchor = torch.mean((h_opt.feats[..., 0:3] - dual_vertices_opt) ** 2)
        
        # Convert to dense representation to compute loss 
        # dense_pred = h_opt.to_dense()
        # pred_intersected_dense = dense_pred[..., 3:6]
        # pred_vertices_dense = dense_pred[..., 0:3]
        # vertices_dense = vertices_sp.to_dense()
        # intersected_dense = intersected_sp.to_dense()

        # # Pad the dense tensors so both prediction and GT share the same shape.
        # if intersected_dense.shape != gt_intersected_dense.shape:
        #     intersected_dense, gt_intersected_dense = pad_sparse_features(
        #         intersected_dense, gt_intersected_dense
        #     )
        # if vertices_dense.shape != gt_vertices_dense.shape:
        #     vertices_dense, gt_vertices_dense = pad_sparse_features(
        #         vertices_dense, gt_vertices_dense
        #     )

        # intersected_dense = intersected_dense.to(torch.float32)
        # gt_intersected_dense = gt_intersected_dense.to(torch.float32)

        # bce_loss = F.binary_cross_entropy_with_logits(
        #     intersected_dense.reshape(-1).to(device),
        #     gt_intersected_dense.reshape(-1).to(device)
        # )
        # # mask = gt_sparse_mask.bool() # boolean mask where GT has voxels
        # mse_loss = F.mse_loss(vertices_dense.to(device), gt_vertices_dense.to(device))
        pred_intersected_matched, gt_intersected_matched = \
            match_sparse_voxels_nn(
                pred_coords=vertices_sp.coords[:,1:], 
                pred_feats=intersected_f,
                gt_coords=gt_indices,
                gt_feats=gt_opt_intersected.float(),
               
            )

        bce_loss = F.binary_cross_entropy_with_logits(
            pred_intersected_matched,
            gt_intersected_matched
        )
        
        
        # begin 
        
        # KD-tree nearest neighbor
        gt_coords_cpu = gt_indices.detach().cpu().numpy()
        pred_coords_cpu = vertices_sp.coords[:,1:].detach().cpu().numpy()
        tree = KDTree(pred_coords_cpu)

        distances_cpu, nn_pred_idx_cpu = tree.query(gt_coords_cpu, k=1)

        # Move KD-tree outputs to CUDA
        distances = torch.from_numpy(distances_cpu.squeeze()).to(device)
        nn_pred_idx = torch.from_numpy(nn_pred_idx_cpu.squeeze()).to(device)

        # ============================
        # Topology Movement Losses
        # ============================

        # 1. Distance loss
        distance_loss = distances.mean()

        # 2. Missing-voxel penalty
        missing_mask = distances > 1.5   # threshold in voxel units

        if missing_mask.any():
            # IMPORTANT: mask must be on CUDA
            missing_gt = gt_opt_intersected.float()[missing_mask.to(device)]
            missing_pred = intersected_f[nn_pred_idx[missing_mask.to(device)]]

            missing_loss = F.binary_cross_entropy_with_logits(
                missing_pred,
                missing_gt
            )
        else:
            missing_loss = torch.zeros((), device=device)

        # 3. Extra-voxel penalty
        pred_only_mask = torch.ones(vertices_sp.coords.shape[0], dtype=torch.bool, device=device)
        pred_only_mask[nn_pred_idx] = False

        if pred_only_mask.any():
            extra_pred = intersected_f[pred_only_mask]
            extra_loss = extra_pred.sigmoid().mean()
        else:
            extra_loss = torch.zeros((), device=device)

        # Combine topology losses
        topology_loss = (
            1.0 * distance_loss +
            5.0 * missing_loss +
            2.0 * extra_loss
        )

        total_loss = total_loss + topology_loss

        # end


        
        pred_vertices_matched, gt_vertices_matched = \
            match_sparse_voxels_nn(
                pred_coords=vertices_sp.coords[:,1:], 
                pred_feats=vertices_f,
                gt_coords=gt_indices,
                gt_feats=gt_vertices,
                
            )

        mse_loss = F.mse_loss(pred_vertices_matched, gt_vertices_matched)


        
        geo_loss = bce_loss + mse_loss #+ pred_only_penalty + gt_only_penalty
        if i==5:
            print("color shape:", color.shape)
            print("mesh.vertices shape:", mesh.vertices.shape)
            print("mesh.vertex_attrs shape:", mesh.vertex_attrs.shape)

        
        
        print(f"Iteration {i} : Geometry Loss: {geo_loss.item():.3f}\n")
        for idx in indices:
        # render
        
            extr = extrinsics_t[idx] @ object_pose
            
            
            buffers = renderer.render(
                mesh=mesh,
                extrinsics=extr,
                intrinsics=intr_scaled, # 
                return_types=["attr", "mask", "normal", "depth"], # "attr", 
                # transformation=base_object_pose
            )
            # pred = buffers["attr"][:3]
            # gt = ref_images[idx] 
            pred_img = buffers["attr"][:3] 
            mask = buffers["mask"]
            normal = buffers["normal"]
            depth = buffers["depth"]
            
           
            
            # Compare only RGB (first 3 channels) 
            # gt_rgb = gt[:3]  # Drop alpha if present
            # pred = pred * mask
            
            # render groundtruth normal and mask
            gt_buffers = renderer.render(
                mesh=dtc_mesh.to(device=device),
                extrinsics = extrinsics_t[idx] @ base_object_pose,
                intrinsics = intr_scaled,
                return_types=["attr", "mask", "normal", "depth"]
            )
            gt_img = gt_buffers["attr"][:3]
            gt_normal = gt_buffers["normal"]
            gt_mask = gt_buffers["mask"]
            gt_depth = gt_buffers["depth"]

            rgb_loss = criterion(pred_img, gt_img)
            
            # losses
            mask_loss = l1_criterion(mask, gt_mask)
            normal_loss = l1_criterion(normal, gt_normal)
            depth_loss = l1_criterion(depth, gt_depth)

            # Image loss
            loss = rgb_loss + lambda_mask * mask_loss + lambda_depth * depth_loss + lambda_normal * normal_loss
            
            # img_loss = criterion(pred, gt_rgb)
            if mask.ndim == 2:
                mask = mask[None, ...]  
                gt_mask = gt_mask[None, ...]      
            elif mask.shape[0] != 1:
                mask = mask.permute(2, 0, 1)
                gt_mask = gt_mask.permute(2, 0, 1)
                
            if depth.ndim == 2:
                depth = depth[None, ...]  
                gt_depth = gt_depth[None, ...]      
            elif depth.shape[0] != 1:
                depth = depth.permute(2, 0, 1)
                gt_depth = gt_depth.permute(2, 0, 1)
            
            
          
            if display_interval and (idx % display_interval == 0 or idx == frames_per_iter - 1):
                normal_vis = (normal - normal.min()) / (normal.max() - normal.min() + 1e-6)
                H, W = normal_vis.shape[1], normal_vis.shape[2]
                alpha = torch.ones(1, H, W, device=normal_vis.device)
                normal_rgba = torch.cat([normal_vis, mask if mask is not None else alpha], dim=0)
                gt_normal_rgba = torch.cat([gt_normal, gt_mask if gt_mask is not None else alpha], dim = 0)
                
                mask_overlay = torch.cat([mask, gt_mask], dim=0).repeat(2, 1, 1)

                pred_rgba_vis = torch.cat([pred_img, mask], dim=0) if mask is not None else pred_img
                gt_img_vis = torch.cat([gt_img[:3], gt_mask], dim=0) if gt_mask is not None else gt_img
                # gt_vis = gt[:3] if gt.shape[0] >= 3 else gt
                # gt_rgba_vis = torch.cat([gt_vis, torch.ones(1, H, W, device=gt.device)], dim=0)
                
                # img = torch.cat([pred_rgba_vis, gt_rgba_vis, normal_rgba], dim=2)   
                
                mask_rgba = gt_mask.repeat(4, 1, 1)
                depth_rgba = gt_depth.repeat(4, 1, 1)
                
                img = torch.cat([mask_overlay, pred_rgba_vis ,gt_img_vis,  normal_rgba], dim = 2)
                
                img = img.permute(1, 2, 0)                     
                img = img.clamp(0.0, 1.0)                        
                img_np = img.detach().cpu().numpy()
                
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                save_image(f"{out_dir}/opt_{i:04d}_{idx:02d}.png", img_np)




            if i in TEST_INTERVALS:
                test_cam_extr = torch.tensor(test_transforms["frames"][TEST_INDEX]["transform_matrix"], dtype=torch.float32, device=device)
                test_pose = test_cam_extr @ object_pose
                test_buffer = renderer.render(
                    mesh=mesh, 
                    extrinsics=test_pose, 
                    intrinsics=intr_scaled, 
                    return_types=["attr","mask", "normal", "depth"]
                )
                test_normal = test_buffer["normal"]
                test_mask = test_buffer["mask"]
                test_depth = test_buffer["depth"]
                test_normal_vis = (test_normal - test_normal.min()) / (test_normal.max() - test_normal.min())
            
                test_depth_vis = test_depth.unsqueeze(0).repeat(3, 1, 1)
                test_img = torch.cat([ test_depth_vis, test_normal_vis ], dim=2)   
                test_img = test_img.permute(1, 2, 0)                     
                test_img = test_img.clamp(0.0, 1.0)                        
                test_img_np = test_img.detach().cpu().numpy()
                save_image(f"{test_out_dir}/test_{i:04d}.png", test_img_np)

           
            total_loss = total_loss + loss
            
        total_loss += lambda_geo  * geo_loss
        total_loss = total_loss / len(indices)
        # if latent_l2_weight > 0:
        #     total_loss = total_loss + latent_l2_weight * torch.mean(h_feats.pow(2))
        print(f"[iter {i}] loss={float(total_loss.detach().cpu()):.6f}")
        total_loss.backward()
        # if grad_clip_norm > 0:
        #     torch.nn.utils.clip_grad_norm_(h_feats, grad_clip_norm)
        optimizer.step()
        
        print("Vertices Grad Mean:", vertices_sp.feats.grad.mean().item())
        print("Color Grad Mean:", base_color_f.grad.mean().item())
        print("Quard lerp Grad Mean:", quad_lerp_f.grad.mean().item())
        print("Intersected flag Grad Mean:", intersected_sp.feats.grad.mean().item())

        
    print("Finished optimization...")
    # extract final mesh and save. here 

    with torch.no_grad():
        vertices_sp = vertices_sp.replace(feats=vertices_f)
        intersected_sp = intersected_sp.replace(feats=intersected_f)
        vertices = (1 + 2*voxel_margin) * F.sigmoid(vertices_sp.feats) - voxel_margin
        intersected_flag = torch.sigmoid(intersected_sp.feats) > 0.2
        color = torch.sigmoid(base_color_f)
        quad_lerp = F.softplus(quad_lerp_f)
        
        
        
        extracted_vertices, extracted_faces = o_voxel.convert.flexible_dual_grid_to_mesh(   
                vertices_sp.coords[:, 1:],  
                vertices,   
                # gt_intersected_dense,
                intersected_flag,
                quad_lerp, 
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],  
                grid_size=RES,  
                train=True
            )
        dense_grid = build_dense_color_grid(vertices_sp.coords[:,1:], color, RES)
        mesh_vertices_norm = normalize_vertices(extracted_vertices)
        vertex_colors = sample_vertex_colors(dense_grid, mesh_vertices_norm)
        final_mesh = Mesh( extracted_vertices, extracted_faces,
                    vertex_attrs=vertex_colors.contiguous()
        )
                
    output_glb_path = f"{test_out_dir}/opt.glb"
    trimesh_obj = trimesh.Trimesh(vertices = final_mesh.vertices.cpu().detach().numpy(), 
                                  faces = final_mesh.faces.cpu().detach().numpy(),vertex_colors=color.detach().cpu().numpy())
    trimesh_obj.export(output_glb_path)
    # print(f"Finish optimization, mesh exported to {output_glb_path}")

    





def main():
    parser = argparse.ArgumentParser(description="Refine Trellis latents, using image loss point renderer. ")
    
    parser.add_argument("--yaml-config", type=Path, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    refine_ovoxel(
        yaml_config_path=args.yaml_config,
    )


if __name__ == "__main__":
    main()
