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
from trellis2.representations import MeshWithVoxel, Mesh
from trellis2.utils import render_utils
from trellis2.modules.sparse import SparseTensor
from opt_utils.utility import display_image, save_image, transf3d_broadcast, transf3d, get_normalization_matrix
from dmtet.dmtet import DmTetGeometry
import o_voxel
from o_voxel import postprocess


render_resolution = 512 
model_name = "trellis_init"

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
    out_dir = config_yaml.get("out_dir", "optimization_output/hypothesis")
    dmtet_res = int(config_yaml.get("dmtet_res", 64))
    if not Path(out_dir).exists(): 
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    output_glb_path = Path(f"{out_dir}/mesh/{model_name}.glb")

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


    prompt_img = Image.open(image_paths[prompt_idx])# .convert("RGBA")
    w, h = prompt_img.size 
    mesh_= pipeline.run(
        prompt_img,
    )[0]
    attr = mesh_.attrs.clone().detach().to(device).requires_grad_(True)
    attrs = nn.Parameter(attr)
   
    intr_scaled = _scale_opencv_intrinsics(intr, (w, h), render_resolution).to(device=device)

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
        
    object_pose = base_object_pose  @ object_transforms_fixed
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

    # compute sdf from initial trelli2 mesh
    dmtet = DmTetGeometry(
        res=128, 
        device = device.__str__(), 
        init_mesh=mesh_,
        scale=1.0
    )
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam([attrs], lr=lr)
        
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD([attrs], lr=lr, momentum=0.9)
    else:
        print(f"Invalid optimizer type: {optimizer_type}. Defaulting to Adam. ")
        optimizer = torch.optim.Adam([attrs], lr=lr)

    dmtet_optimizer = dmtet.getOptimizer(lr)
    for idx in range(num_iters): 
        
        mesh = dmtet.get_mesh(vertex_attr = attrs)
        total_loss =  torch.zeros((), device=device)
        optimizer.zero_grad()
        dmtet_optimizer.zero_grad()

        indices = (
            np.arange(N)
            if frames_per_iter == N
            else np.atleast_1d(rng.choice(N, size=frames_per_iter, replace=False))
        )

        for i in indices: 
            cam_extr = extrinsics_t[i]

            buffer = renderer.render(mesh=mesh, extrinsics=cam_extr, 
                                     intrinsics=intr_scaled, return_types=["attr","mask", "normal"],
                                     transformation=object_pose)
            
            img = buffer["attr"][:3]
            mask = buffer["mask"]
            normal = buffer["normal"]

            ref_img = ref_images[i]
            loss = criterion(img, ref_img[:3])
            total_loss += loss 

            if display_interval and (idx % display_interval == 0 or idx == num_iters -1):
                normal_vis = (normal - normal.min()) / (normal.max() - normal.min())
                H, W = normal_vis.shape[1], normal_vis.shape[2]
                alpha = torch.ones(1, H, W, device=normal_vis.device)
                if mask is not None:
                    if mask.dim() == 2:
                        mask_ch = mask.unsqueeze(0)
                    else:
                        mask_ch = mask
                else:
                    mask_ch = alpha

                normal_rgba = torch.cat([normal_vis, mask_ch], dim=0)

                pred_rgba_vis = torch.cat([img, mask_ch], dim=0) if mask is not None else img  # RGB only
                
                img = torch.cat([pred_rgba_vis, ref_img, normal_rgba], dim=2)   
                img = img.permute(1, 2, 0)                     
                img = img.clamp(0.0, 1.0)                        
                img_np = img.detach().cpu().numpy()
                save_image(f"{out_dir}/opt_{idx:04d}_{i:02d}.png", img_np)
            
        total_loss = total_loss / len(indices)
        print(f"[iter {idx}] loss={float(total_loss.detach().cpu()):.6f}")
        total_loss.backward()
        optimizer.step()
    
    # Export final mesh
    final_mesh = dmtet.get_mesh(vertex_attr = attrs)

    
    output_glb_path.parent.mkdir(parents=True, exist_ok=True)
    color = attrs.detach().cpu().numpy()
    trimesh_obj = trimesh.Trimesh(vertices = final_mesh.vertices.cpu().detach().numpy(), 
                                  faces = final_mesh.faces.cpu().detach().numpy(),vertex_colors=color)
    trimesh_obj.export(output_glb_path)


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--yaml-config", type=Path, required=True)
    args = parser.parse_args()

    refine_latents(
        yaml_config_path=args.yaml_config,
    )


if __name__ == "__main__":
    main()

