import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["ATTN_BACKEND"] = 'xformers'

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
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
from opt_utils.utility import display_image, save_image, transf3d_broadcast, transf3d, get_normalization_matrix
from dmtet.dmtet import DmTetGeometry
import o_voxel
from o_voxel import postprocess



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


def _scale_opencv_intrinsics(intr: Dict, orig_wh: Tuple[int, int], render_resolution: int) -> torch.Tensor:
    fx, fy, cx, cy = float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"])
    orig_w, orig_h = orig_wh

    fx_n = fx / float(orig_w)
    fy_n = fy / float(orig_h)
    cx_n = cx / float(orig_w)
    cy_n = cy / float(orig_h)

    intr_normalized = torch.tensor(
        [[fx_n, 0.0, cx_n], [0.0, fy_n, cy_n], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    return intr_normalized

@torch.no_grad()
def _freeze_module_params(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)

def freeze_pipeline(pipeline):
    for name, module in pipeline.models.items():
        if isinstance(module, torch.nn.Module):
            for p in module.parameters():
                p.requires_grad_(False)


def copy_mesh(mesh: MeshWithVoxel) -> MeshWithVoxel:
    """Create a shallow copy of a mesh, keeping the same data pointers."""
    return MeshWithVoxel(
        vertices=mesh.vertices,
        faces=mesh.faces,
        origin=mesh.origin,
        voxel_size=mesh.voxel_size,
        coords=mesh.coords,
        attrs=mesh.attrs,
        voxel_shape=mesh.voxel_shape,
        layout=mesh.layout,
    )

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

def _decode_latent_differentiable(
    pipeline: Trellis2ImageTo3DPipeline,
    shape_slat_opt,
    tex_slat_opt,
    resolution: int,
) -> List[MeshWithVoxel]:
   
   
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

def softplus_inverse(y):
    return y + torch.log(-torch.expm1(-y))



def refine_latents(
    yaml_config_path: Path,
):
    config_yaml = yaml.safe_load(open(yaml_config_path, "r"))
    

    lr = float(config_yaml["lr"])
    num_iters = int(config_yaml["num_iters"])
    envmap_path = Path(config_yaml["envmap_path"])
    assert envmap_path.exists(), f"Envmap path does not exist: {envmap_path}"
    gt_mesh = Path(config_yaml.get("gt_mesh", ""))
    assert gt_mesh.exists(), f"DTC mesh path does not exist: {gt_mesh}"
    transforms_json_path = Path(config_yaml["transforms_json"])
    assert transforms_json_path.exists(), f"Transforms JSON path does not exist: {transforms_json_path}"
    output_glb_path = Path(config_yaml.get("output_path", "optimization_output/output_mesh.glb"))
    if not output_glb_path.exists():
        output_glb_path.mkdir(parents=True, exist_ok=True)

    pipeline_type = str(config_yaml.get("pipeline_type", '512'))
    frames_per_iter = config_yaml.get("frames_per_iter", None)
    near = float(config_yaml.get("near", 0.01))
    far = float(config_yaml.get("far", 100.0))
    ssaa = int(config_yaml.get("ssaa", 1))
    lamda  = float(config_yaml.get("lamda", 0.001))
    peel_layers = int(config_yaml.get("peel_layers", 8))
    loss_type = str(config_yaml.get("loss_type", "mse"))  # loss_fn
    latent_l2_weight = float(config_yaml.get("latent_l2_weight", 0.0))
    grad_clip_norm = float(config_yaml.get("grad_clip_norm", 0.0))  # 0 disables
    display_interval = int(config_yaml.get("display_interval", 20))
    out_dir = config_yaml.get("out_dir", "optimization_output")
    dmtet_res = int(config_yaml.get("dmtet_res", 64))
    
    optimize_texture_only = bool(config_yaml.get("optimize_texture_only", False))  # NEW
    optimizer_type = str(config_yaml.get("optimizer_type", "adam")).lower()  # NEW

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

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.low_vram = False
    pipeline.cuda()


    # Build envmap and renderer
    envmap_img = cv2.imread(str(envmap_path), cv2.IMREAD_UNCHANGED)
    assert envmap_img is not None, f"Failed to read envmap: {envmap_path}"
    envmap_img = cv2.cvtColor(envmap_img, cv2.COLOR_BGR2RGB)
    env_map = EnvMap(torch.tensor(envmap_img, dtype=torch.float32, device=device))

    # Run once on  prompt image to obtain latents and render resolution 
    prompt_img = Image.open(image_paths[prompt_idx]).convert("RGBA")
    
    mesh_= pipeline.run(
        prompt_img,
        # return_latent=True,
        pipeline_type=pipeline_type,
    )[0]
    
    render_resolution = 512 
   

    # Scale intrinsics to render_resolution.
    orig_img = Image.open(image_paths[prompt_idx])
    orig_w, orig_h = orig_img.size
    intr_scaled = _scale_opencv_intrinsics(intr, (orig_w, orig_h), render_resolution).to(device=device)

    # camera extrinsics
    extrinsics = []
    for f in frames:
        extrinsics.append(torch.tensor(f["transform_matrix"], dtype=torch.float32))
    
    # Compute alignment transform for the mesh 
    if gt_mesh.exists():
        dtc_mesh = trimesh.load_mesh(gt_mesh) 
        with torch.no_grad():
            object_transforms_fixed = get_normalization_matrix(dtc_mesh, mesh_)
           
    else:
        print("No ground truth mesh provided, skipping alignment step.")
        object_transforms_fixed = torch.eye(4, device=device)
        
    object_pose = base_object_pose @ object_transforms_fixed
    criterion = nn.MSELoss(reduction='mean') 
    extrinsics_t = torch.stack(extrinsics, dim=0).to(device=device)
    ref_images   = torch.stack(
        [_resize_rgb_to_tensor(p, render_resolution, device) for p in image_paths], dim=0
    )
    N = len(image_paths)
    print(f"Loaded {N} reference images for optimization.")
    rng = np.random.default_rng(seed=int(config_yaml.get("seed", 0)))
    
    renderer = MeshRenderer(
        rendering_options={
            "resolution": render_resolution,
            "near": near,
            "far": far,
            "ssaa": ssaa,
        },
        device=device.type,
    )
    
    #LEO
    # 1. Convert the predicted MeshWithVoxel to O-Voxel format
    # I won't save explicitely, but the representation will be used for the optimization later. 
    
    # export the mesh to glb 
    glb = o_voxel.postprocess.to_glb(
        vertices            =   mesh_.vertices,
        faces               =   mesh_.faces,
        attr_volume         =   mesh_.attrs,
        coords              =   mesh_.coords,
        attr_layout         =   mesh_.layout,
        voxel_size          =   mesh_.voxel_size,
        aabb                =   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target   =   1000000,
        texture_size        =   1024,
        remesh              =   True,
        remesh_band         =   1,
        remesh_project      =   0,
        verbose             =   True
    )
    
    mesh = glb
    vertices = mesh_.vertices.float().cpu()
    faces = mesh_.faces.long().cpu()
    # vertices = torch.tensor(mesh_.vertices).float()
    # faces = torch.tensor(mesh_.faces).long()
    

    
    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices=torch.tensor(mesh.vertices, dtype=torch.float32),
        faces=torch.tensor(mesh.faces, dtype=torch.float32), 
        grid_size=render_resolution, 
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], 
        face_weight=1.0, 
        boundary_weight=0.2, 
        regularization_weight=1e-2, 
        timing=True
    )

    
    # # sorting 
    vid = o_voxel.serialize.encode_seq(voxel_indices)
    mapping = torch.argsort(vid)
    voxel_indices = voxel_indices[mapping]
    dual_vertices = dual_vertices[mapping]
    intersected = intersected[mapping]
    
    # # Material voxelization
    # # mesh_vox = trimesh.Trimesh(vertices=vertices.cpu().numpy(), faces=faces.cpu().numpy())
    # # tex = trimesh.visual.texture.TextureVisuals.from_color(mesh_vox.visual.vertex_colors)
    # # mesh_vox.visual = tex
    voxel_indices_mat, attributes = o_voxel.convert.textured_mesh_to_volumetric_attr(
        mesh, 
        grid_size=render_resolution, 
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], 
        timing=True
    )
    # ## sorting 
    vid_mat = o_voxel.serialize.encode_seq(voxel_indices_mat)
    mapping_mat = torch.argsort(vid_mat)
    attributes = {k: v[mapping_mat] for k, v in attributes.items()}
 
    # # packing
    dual_vertices = dual_vertices * render_resolution - voxel_indices
    dual_vertices = (torch.clamp(dual_vertices, 0, 1) * 255).type(torch.uint8)
    intersected = (intersected[:, 0:1] + 2 * intersected[:, 1:2] + 4 * intersected[:, 2:3]).type(torch.uint8)
    attributes['dual_vertices'] = dual_vertices 
    attributes['intersected'] = intersected 
    # # o_voxel.io.write("ovoxel_helmet.vxz", voxel_indices, attributes)
   
    # # 2. Extract and convert the optimization parameters from the O-Voxel representation 
    dual_vertices_opt = attributes['dual_vertices'].float() / 255.0
    gt_intersected = torch.cat([
        intersected % 2,
        intersected // 2 % 2, 
        intersected // 4 % 2, 
    ], dim=-1).bool()
    
    voxel_margin = 0.5 
   
    
    quad_lerp = torch.nn.functional.softplus(torch.ones_like(dual_vertices[:, :1]) * 0.1)
    
    # # Material attributes, base_color, metallic, rougness, alpha, normal, emissive. 

    
    base_color = attributes['base_color'].float() / 255.0

    intersected_f = intersected.float() / 7.0
    
    h_feats = torch.cat([
        dual_vertices_opt, # 0-3
        intersected_f, # 3-6
        quad_lerp, 
        base_color
    ], dim=-1).to(device=device)
    
    # # print("H feats: \n", h_feats.shape)
    
    # # Use a sparse tensor to hold the optimization parameters with gradients enabled
    h_feats = torch.nn.Parameter(h_feats)
    h = SparseTensor(
        feats= h_feats, 
        coords=torch.cat([torch.zeros_like(voxel_indices[:, 0:1]), voxel_indices], dim=-1) 
    ).to(device=device)
    
    # print("H feats: \n", h_feats.shape)
    # print("dual_vertices_opts: \n", dual_vertices_opt.shape)
    # print("Intersected_f: \n", intersected_f.shape)
    # print("Quard lerp: \n", quad_lerp.shape)
    # print("Base color: \n", base_color.shape)

    #========================DEBUG=====================================================
   
    
    
    # extr_ = extrinsics_t[0] @ object_pose
    
    # v_ = dual_vertices / 255.0
    
    # packed_mesh = Mesh(*o_voxel.convert.flexible_dual_grid_to_mesh(  
    #     voxel_indices.to(device=device),   
    #     v_.to(device=device),   
    #     gt_intersected.to(device=device),
    #     (dual_vertices[:, :1] * 0.1).to(device=device),  
    #     aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],  
    #     grid_size=render_resolution,  
    #     train=True  
    # ), base_color.contiguous())
    
    # buff = renderer.render(
    #     mesh=packed_mesh, 
    #     extrinsics=extr_,
    #     intrinsics=intr_scaled,
    #     return_types=["mask", "normal"],
    # )
    # packed_normal = buff['normal']
    # packed_mask = buff['mask']
    # packed_img = torch.cat([packed_normal, packed_mask.repeat(3, 1, 1)], dim=2)
    # packed_img = packed_img.permute(1, 2, 0)
    # packed_img = packed_img.clamp(0.0, 1.0)
    # packed_img = packed_img.detach().cpu().numpy()
    # save_image(f"{out_dir}/rot_packed_normal_mask.png", packed_img)
    
    
    
    
    #=================================================INITIAL MESH EXTRACTION=======================================================================

    
    # buffers = renderer.render(
    #     mesh=mesh_,
    #     extrinsics=extr_,
    #     intrinsics=intr_scaled,
    #     return_types=[ "mask", "normal"],
    #     # return_types=["base_color"],
    #     # return_types=["attr", "mask", "normal"], 
    #     # transformation = object_pose
      
    # )
    # normal = buffers["normal"]      # (3, H, W)
    # mask   = buffers["mask"][None]  # (1, H, W)
    
    # img = torch.cat([normal, mask.repeat(3,1,1)], dim=2)  # (4, H, W)
    # img = img.permute(1, 2, 0)  # (H, W, 4)
    # img = img.clamp(0.0, 1.0)
    # img_np = img.detach().cpu().numpy()

    # save_image(f"{out_dir}/rot_initial_normal_mask.png", img_np)


    #======================================================================================
#----------------------------------------------OPTIMIZATION DEBUG--------------------------------------------------------------------------------------



    # optimizer = torch.optim.Adam([h_feats, color], lr = lr)
    optimizer = torch.optim.Adam([h_feats], lr = lr)
    
    for i in range (num_iters+1): 
        optimizer.zero_grad()
      
    
        h_opt = h.replace(feats = h_feats)
        vertices = (1 + 2 * voxel_margin) *   torch.nn.functional.sigmoid(h_opt.feats[..., 0:3]) - voxel_margin
        intersected_logit  = torch.cat([
            h_opt.feats[..., 	3:4] % 2,
            h_opt.feats[..., 	3:4] // 2 % 2, 
            h_opt.feats[..., 	3:4] // 4 % 2, 
        ], dim=-1).bool()
      
    
        # color = h_opt.feats[..., 5:8]#.transpose(0, 1) 
        color = torch.sigmoid(h_opt.feats[..., 5:8])
        # quad_lerp = softplus_inverse(h_opt.feats[..., 4:5])
        quad_lerp = torch.nn.functional.softplus(h_opt.feats[..., 4:5])
        # quad_lerp = h_opt.feats[..., 4:5]
        
        mesh_verts, mesh_faces = o_voxel.convert.flexible_dual_grid_to_mesh(  
            h_opt.coords[:, 1:],   
            vertices,   
            # gt_intersected,
            intersected_logit,
            quad_lerp,  
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],  
            grid_size=render_resolution,  
            train=False 
        )  
        #, vertex_attrs=color.contiguous() 
        # L = (mesh_faces.shape[0] // 4) 
        # original_vertex_count = vertices.shape[0]  
        
     

        # quad_indices = mesh_faces[:L*4].reshape(-1, 4)[:, :4]  
        
        

        # quad_colors = color[quad_indices]  
        # mean_color_02 = (quad_colors[:, 0] + quad_colors[:, 2]) / 2  
        # mean_color_13 = (quad_colors[:, 1] + quad_colors[:, 3]) / 2  
         
        # split_weight_ws = quad_lerp[quad_indices]  
        # split_weight_ws_02 = split_weight_ws[:, 0] * split_weight_ws[:, 2]  
        # split_weight_ws_13 = split_weight_ws[:, 1] * split_weight_ws[:, 3]  
        
        # mid_colors = (  
        #     split_weight_ws_02 * mean_color_02 +  
        #     split_weight_ws_13 * mean_color_13  
        # ) / ((split_weight_ws_02 + split_weight_ws_13)  + 1e-8)
        
        # extended_colors = torch.cat([color, mid_colors], dim=0) 
        
        
        
        
        
        mesh = Mesh(mesh_verts, mesh_faces, vertex_attrs=color.contiguous())
        
        indices = (
            np.arange(N)
            if frames_per_iter == N
            else np.atleast_1d(rng.choice(N, size=frames_per_iter, replace=False))
        )
        total_loss = torch.zeros((), device=device)

        for idx in indices: 
            extr = extrinsics_t[idx] #@ object_pose
            buffer = renderer.render(
                mesh=mesh,
                extrinsics=extr,
                intrinsics=intr_scaled,
                return_types=["attr", "mask", "normal"],
                transformation=object_pose 
            )
            
            pred = buffer["attr"] #[:3]    
            gt   = ref_images[idx]
            gt_rgb = gt [:3]
            mask = buffer["mask"]
            normal = buffer["normal"]
        
            loss = criterion(pred, gt_rgb)
            mask_loss = criterion(mask, gt[3])
            loss += mask_loss
            total_loss += loss
            
            # Visualization
            if display_interval and (idx % display_interval == 0 or idx == num_iters -1):
                normal_vis = (normal - normal.min()) / (normal.max() - normal.min())
                H, W = normal_vis.shape[1], normal_vis.shape[2]
                alpha = torch.ones(1, H, W, device=normal_vis.device)
                if mask is not None:
                    if mask.ndim == 2:
                        mask = mask.unsqueeze(0)  # (1, H, W)
                else:
                    mask = alpha

            
                normal_rgba = torch.cat([normal_vis, mask ], dim=0)
                pred_rgba_vis = torch.cat([pred, mask], dim=0) if mask is not None else pred
                img = torch.cat([pred_rgba_vis, gt, normal_rgba], dim=2)     
                img = img.permute(1, 2, 0)                     
                img = img.clamp(0.0, 1.0)                        
                img_np = img.detach().cpu().numpy()

                save_image(f"{out_dir}/opt_{i:04d}_{idx:02d}.png", img_np)

        total_loss = total_loss / len(indices)
        print(f"[iter {i}] loss={float(total_loss.detach().cpu()):.6f}")

        total_loss.backward()
        optimizer.step()
    
    
    
    
    
# -----------------------------------------------------------------------------------------------------------------------------------------------------

    # if optimizer_type  == "sgd":
    #     print(f"Using SGD Optimizer with lr={lr}")
    #     optimizer = torch.optim.SGD([h_feats], lr=lr)
    # else:
    #     print(f"Using Adam Optimizer with lr={lr}")
    #     optimizer = torch.optim.Adam([h_feats], lr=lr)


# Main optimization loop 
    # for it in range(num_iters):
    #     indices = (
    #         np.arange(N)
    #         if frames_per_iter == N
    #         else np.atleast_1d(rng.choice(N, size=frames_per_iter, replace=False))
    #     )
        
    #     h_opt = h.replace(feats = h_feats)
        
    #     vertices = (1 + 2 * voxel_margin) *   torch.nn.functional.sigmoid(h_opt.feats[..., 0:3]) - voxel_margin
    #     intersected_logit = (h_opt.feats[..., 	3:4] > 0)

        

    #     quad_lerp = torch.nn.functional.softplus(h_opt.feats[..., 4:5])
        
    #     base_color = h_opt.feats[..., 5:8]
        
        
    #     mesh = Mesh(*o_voxel.convert.flexible_dual_grid_to_mesh(  
    #         h_opt.coords[:, 1:],   
    #         vertices,   
    #         gt_intersected,
    #         quad_lerp,  
    #         aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],  
    #         grid_size=render_resolution,  
    #         train=True  
    #     ),base_color.contiguous()) 
    

    #     assert mesh is not None, f"The mesh is not valid"
    #     optimizer.zero_grad(set_to_none=True)
        
    #     total_loss = torch.zeros((), device=device)
        
              
    #     for idx in indices:
            
            
    #         extr = extrinsics_t[idx] #@ object_pose
    #         # extr = R_yup_to_zup @ extr
           
           
    #         buffers = renderer.render(
    #             mesh=mesh,
    #             extrinsics=extr,
    #             intrinsics=intr_scaled,
    #             return_types=["attr", "mask", "normal"],
    #             # return_types=["base_color"],
    #             # return_types=["attr", "mask", "normal"], 
    #             transformation = object_pose
    #             # envmap=env_map,
    #         )
           

      
    #         pred = buffers["attr"][:3] 
           

            
    #         gt   = ref_images[idx]  
    #         # mask = buffers["mask"]
    #         # normal = buffers["normal"]

     
    #         # if mask.ndim == 2:
    #         #     mask = mask[None, ...]        
    #         # elif mask.shape[0] != 1:
    #         #     mask = mask.permute(2, 0, 1)
            
    #         # Compare only RGB (first 3 channels) 
    #         gt_rgb = gt[:3]  # Drop alpha if present
           

    #         # diff = pred - gt_rgb
    #         img_loss =  criterion(pred, gt_rgb)
    #         # mask_loss = criterion(mask, gt[3:4] if gt.shape[0] >=4 else torch.ones_like(mask)) # if gt has alpha mask, add mask loss
    #         # sdf_loss = dmtet_mesh.get_sdf_loss()
    #         per_pix = img_loss #+ mask_loss #+ sdf_loss

             

            
    #         if display_interval and (idx % display_interval == 0 or idx == num_iters -1):
    #             # normal_vis = (normal - normal.min()) / (normal.max() - normal.min())
    #             # H, W = normal_vis.shape[1], normal_vis.shape[2]
    #             # alpha = torch.ones(1, H, W, device=normal_vis.device)
    #             # normal_rgba = torch.cat([normal_vis, mask if mask is not None else alpha], dim=0)
                
    #             # Build display image with RGB channels only
    #             # pred_rgba_vis = torch.cat([pred, mask], dim=0) if mask is not None else pred
    #             # gt_vis = gt[:3] if gt.shape[0] >= 3 else gt  # RGB only
    #             # gt_rgba_vis = torch.cat([gt_vis, torch.ones(1, H, W, device=gt.device)], dim=0)
                
    #             # img = torch.cat([pred_rgba_vis, gt, normal_rgba], dim=2)
               
    #             img = torch.cat([pred, gt[:3]], dim=2)
    #             img = img.permute(1, 2, 0)                     
    #             img = img.clamp(0.0, 1.0)                        
    #             img_np = img.detach().cpu().numpy()
                

    #             save_image(f"{out_dir}/opt_{it:04d}_{idx:02d}.png", img_np)
    #             # display_image(img_np, title=f"Iter {it} - Frame {idx}")

     
    #         loss = per_pix

    #         total_loss = total_loss + loss
        
      
        
    #     total_loss = total_loss / len(indices)

     

    #     print(f"[iter {it}] loss={float(total_loss.detach().cpu()):.6f}")

    #     total_loss.backward()

    #     optimizer.step()


    # # Decode final mesh for export
    # with torch.no_grad():
    #     h_opt = h.replace(feats = h_feats) 
    #     vertices = (1 + 2 * voxel_margin) *   torch.nn.functional.sigmoid(h_opt.feats[..., 0:3]) - voxel_margin
    #     intersected_logit = (h_opt.feats[..., 	3:4] > 0)
    #     quad_lerp = torch.nn.functional.softplus(h_opt.feats[..., 4:5])
    #     base_color = h_opt.feats[..., 5:8]
    #     final_mesh = Mesh(*o_voxel.convert.flexible_dual_grid_to_mesh(  
    #         h_opt.coords[:, 1:],   
    #         vertices,   
    #         intersected_logit,
    #         quad_lerp,  
    #         aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],  
    #         grid_size=render_resolution,  
    #         train=True  
    #     ),base_color.contiguous()) 
        

    # if output_glb_path is not None:
        
    #     # import imageio
        
    #     video = render_utils.make_pbr_vis_frames(render_utils.render_video(final_mesh, envmap=env_map))
    #     imageio.mimsave("mesh_opt_video.mp4", video, fps=15)

    #     # glb = postprocess.to_glb(
    #     #     vertices=final_mesh.vertices,
    #     #     faces=final_mesh.faces,
    #     #     attr_volume=final_mesh.attrs,
    #     #     coords=final_mesh.coords,
    #     #     attr_layout=final_mesh.layout,
    #     #     voxel_size=final_mesh.voxel_size,
    #     #     aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    #     #     decimation_target=int(config_yaml.get("decimation_target", 1_000_000)),
    #     #     texture_size=int(config_yaml.get("texture_size", 1024)),
    #     #     remesh=bool(config_yaml.get("remesh", True)),
    #     #     remesh_band=int(config_yaml.get("remesh_band", 1)),
    #     #     remesh_project=int(config_yaml.get("remesh_project", 0)),
    #     #     verbose=True,
    #     # )
    #     glb = o_voxel.postprocess.to_glb(
    #     vertices            =   final_mesh.vertices,
    #     faces               =   final_mesh.faces,
    #     attr_volume         =   final_mesh.attrs,
    #     coords              =   final_mesh.coords,
    #     attr_layout         =   final_mesh.layout,
    #     voxel_size          =   final_mesh.voxel_size,
    #     aabb                =   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    #     decimation_target   =   1000000,
    #     texture_size        =   1024,
    #     remesh              =   True,
    #     remesh_band         =   1,
    #     remesh_project      =   0,
    #     verbose             =   True
    # )
    #     output_glb_path.parent.mkdir(parents=True, exist_ok=True)
    #     glb_path = output_glb_path / "mesh_opt.glb"
    #     glb.export(str(glb_path), extension_webp=False)
    #     print(f"Exported: {glb_path}")

    # return final_mesh


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--yaml-config", type=Path, required=True)
    args = parser.parse_args()

    refine_latents(
        yaml_config_path=args.yaml_config,
    )


if __name__ == "__main__":
    main()

