import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
from trellis2.representations import MeshWithVoxel
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
    loss_type = str(config_yaml.get("loss_type", "mse"))
    latent_l2_weight = float(config_yaml.get("latent_l2_weight", 0.0))
    grad_clip_norm = float(config_yaml.get("grad_clip_norm", 0.0))
    display_interval = int(config_yaml.get("display_interval", 20))
    out_dir = config_yaml.get("out_dir", "optimization_output")
    dmtet_res = int(config_yaml.get("dmtet_res", 64))
    
    optimize_texture_only = bool(config_yaml.get("optimize_texture_only", False))
    optimizer_type = str(config_yaml.get("optimizer_type", "adam")).lower()

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


    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.low_vram = False
    pipeline.cuda()


    envmap_img = cv2.imread(str(envmap_path), cv2.IMREAD_UNCHANGED)
    assert envmap_img is not None, f"Failed to read envmap: {envmap_path}"
    envmap_img = cv2.cvtColor(envmap_img, cv2.COLOR_BGR2RGB)
    env_map = EnvMap(torch.tensor(envmap_img, dtype=torch.float32, device=device))

    prompt_img = Image.open(image_paths[prompt_idx]).convert("RGBA")
    mesh= pipeline.run(
        prompt_img,
        pipeline_type=pipeline_type,
    )[0]
    render_resolution = 512 
   

    orig_img = Image.open(image_paths[prompt_idx])
    orig_w, orig_h = orig_img.size
    intr_scaled = _scale_opencv_intrinsics(intr, (orig_w, orig_h), render_resolution).to(device=device)

    extrinsics = []
    for f in frames:
        extrinsics.append(torch.tensor(f["transform_matrix"], dtype=torch.float32))
    
    renderer = MeshRenderer(
        rendering_options={
            "resolution": render_resolution,
            "near": near,
            "far": far,
            "ssaa": ssaa,
        },
        device=device.type,
    )


    vertices = mesh.vertices.clone().detach().to(device).requires_grad_(True)
    faces = mesh.faces.clone().detach().to(device)
    mesh_origin = mesh.origin
    voxel_size = mesh.voxel_size
    coords = mesh.coords.clone().detach().to(device)
    attrs = mesh.attrs.clone().detach().to(device).requires_grad_(True)
    voxel_shape = mesh.voxel_shape
    layout = mesh.layout

    opt_vert = nn.Parameter(vertices)
    opt_attrs = nn.Parameter(attrs)


    extrinsics_t = torch.stack(extrinsics, dim=0).to(device=device)
    ref_images   = torch.stack(
        [_resize_rgb_to_tensor(p, render_resolution, device) for p in image_paths], dim=0
    )
    N = len(image_paths)
    print(f"Loaded {N} reference images for optimization.")
    rng = np.random.default_rng(seed=int(config_yaml.get("seed", 0)))

   
    if gt_mesh.exists():
        dtc_mesh = trimesh.load_mesh(gt_mesh)
        with torch.no_grad():
            object_transforms_fixed = get_normalization_matrix(dtc_mesh, mesh)
           
    else:
        print("No ground truth mesh provided, skipping alignment step.")
        object_transforms_fixed = torch.eye(4, device=device)
    
            
    object_pose = base_object_pose @ object_transforms_fixed
    criterion = nn.MSELoss(reduction='mean') 

    
    if optimizer_type  == "sgd":
        print(f"Using SGD Optimizer with lr={lr}")
        optimizer = torch.optim.SGD([opt_vert, opt_attrs], lr=lr)
    else:
        print(f"Using Adam Optimizer with lr={lr}")
        optimizer = torch.optim.Adam([opt_vert, opt_attrs], lr=lr)

    for it in range(num_iters):
        indices = (
            np.arange(N)
            if frames_per_iter == N
            else np.atleast_1d(rng.choice(N, size=frames_per_iter, replace=False))
        )
        
           
        optimizer.zero_grad(set_to_none=True)
       
        
        mesh = MeshWithVoxel(
            vertices=opt_vert,
            faces=faces,
            origin=mesh_origin,
            voxel_size=voxel_size,
            coords=coords,
            attrs=opt_attrs,
            voxel_shape=voxel_shape,
            layout=layout,
        )


        total_loss = torch.zeros((), device=device)
        
              
        for idx in indices:
            
            
            extr = extrinsics_t[idx] @ object_pose
           
           
            buffers = renderer.render(
                mesh=mesh,
                extrinsics=extr,
                intrinsics=intr_scaled,
                return_types=["attr", "mask", "normal"], 
            )

            
            pred = buffers["base_color"]
            
            
            gt   = ref_images[idx]  
            mask = buffers["mask"]
            normal = buffers["normal"]

            
            if mask.ndim == 2:
                mask = mask[None, ...]        
            elif mask.shape[0] != 1:
                mask = mask.permute(2, 0, 1)
            
            gt_rgb = gt[:3]
        

            img_loss =  criterion(pred, gt_rgb)
            mask_loss = criterion(mask, gt[3:4] if gt.shape[0] >=4 else torch.ones_like(mask))
            per_pix = img_loss + mask_loss

             
            if display_interval and (idx % display_interval == 0 or idx == num_iters -1):
                normal_vis = (normal - normal.min()) / (normal.max() - normal.min())
                H, W = normal_vis.shape[1], normal_vis.shape[2]
                alpha = torch.ones(1, H, W, device=normal_vis.device)
                normal_rgba = torch.cat([normal_vis, mask if mask is not None else alpha], dim=0)
                
                pred_rgba_vis = torch.cat([pred, mask], dim=0) if mask is not None else pred
                gt_vis = gt[:3] if gt.shape[0] >= 3 else gt
                gt_rgba_vis = torch.cat([gt_vis, torch.ones(1, H, W, device=gt.device)], dim=0)
                
                img = torch.cat([pred_rgba_vis, gt, normal_rgba], dim=2)   
                img = img.permute(1, 2, 0)                     
                img = img.clamp(0.0, 1.0)                        
                img_np = img.detach().cpu().numpy()
                

                save_image(f"{out_dir}/opt_{it:04d}_{idx:02d}.png", img_np)

            loss = per_pix

            total_loss = total_loss + loss
        
      
        total_loss = total_loss / len(indices)

        
        print(f"[iter {it}] loss={float(total_loss.detach().cpu()):.6f}")

        total_loss.backward()
       

        optimizer.step()


    with torch.no_grad():
        final_mesh = MeshWithVoxel(
            vertices=opt_vert.detach(),
            faces=faces,
            origin=mesh_origin,
            voxel_size=voxel_size,
            coords=coords,
            attrs=opt_attrs.detach(),
            voxel_shape=voxel_shape,
            layout=layout,
        )
        

    if output_glb_path is not None:
        
        
        video = render_utils.make_pbr_vis_frames(render_utils.render_video(final_mesh, envmap=env_map))
        imageio.mimsave("mesh_opt_video.mp4", video, fps=15)

        glb = postprocess.to_glb(
            vertices=final_mesh.vertices,
            faces=final_mesh.faces,
            attr_volume=final_mesh.attrs,
            coords=final_mesh.coords,
            attr_layout=final_mesh.layout,
            voxel_size=final_mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=int(config_yaml.get("decimation_target", 1_000_000)),
            texture_size=int(config_yaml.get("texture_size", 1024)),
            remesh=bool(config_yaml.get("remesh", True)),
            remesh_band=int(config_yaml.get("remesh_band", 1)),
            remesh_project=int(config_yaml.get("remesh_project", 0)),
            verbose=True,
        )
        output_glb_path.parent.mkdir(parents=True, exist_ok=True)
        glb_path = output_glb_path / "mesh_opt.glb"
        glb.export(str(glb_path), extension_webp=False)
        print(f"Exported: {glb_path}")

    return final_mesh


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--yaml-config", type=Path, required=True)
    args = parser.parse_args()

    refine_latents(
        yaml_config_path=args.yaml_config,
    )


if __name__ == "__main__":
    main()

