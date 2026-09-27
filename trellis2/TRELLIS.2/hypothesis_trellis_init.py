import os

from matplotlib import colors
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
from opt_utils.utility import display_image, save_image, align_meshes_pca, get_normalization_matrix, compute_icp_alignment_transform, align_meshes_ransac_icp, align_meshes_similarity
from dmtet.dmtet import DmTetGeometry
import o_voxel
from o_voxel import postprocess
import time

GENERATED_MESH_PATH = Path("/home/fr/fr_fr/fr_ba122/projects/recgen/out/mesh.obj")


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

render_resolution = 512 
model_name = "trellis_init"
TEST_INDEX = 4
TEST_INTERVALS = [0, 1, 50,100, 200,  500, 799, 999]
angle = 90
AXIS_SWAP = rot_z(np.pi/2)
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
    test_transforms_json_path = Path(config_yaml.get("test_transforms_json", ""))
    assert test_transforms_json_path.exists(), f"Test Transforms JSON Path does not exist: {test_transforms_json_path}"

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
    out_dir = config_yaml.get("out_dir", "optimization_output/hypothesis")
    
    
    dmtet_res = int(config_yaml.get("dmtet_res", 64))
    if not Path(out_dir).exists(): 
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    test_out_dir = Path(out_dir) / "test"
    test_out_dir.mkdir(parents=True, exist_ok=True)
    output_glb_path = Path(f"{out_dir}/mesh/{model_name}.glb")

    
    optimizer_type = str(config_yaml.get("optimizer_type", "adam")).lower()  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transforms = _load_json(transforms_json_path)
    test_transforms = _load_json(test_transforms_json_path)
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



    ''' 
    prompt_img = Image.open(image_paths[prompt_idx])
    w, h = prompt_img.size 
    mesh_generation_time_ = time.time()
    mesh_= pipeline.run(
        prompt_img,
        # pipeline_type=pipeline_type,
    )[0]
    # mesh_.fill_holes()
    trimesh.Trimesh(vertices=mesh_.vertices.cpu().numpy(), 
                    faces=mesh_.faces.cpu().numpy()).export(f"{out_dir}/initialization_mesh.obj")
    mesh_generation_time = time.time() - mesh_generation_time_
    attr = mesh_.attrs.clone().detach().to(device).requires_grad_(True)
    attrs = nn.Parameter(attr)
   '''
    
    w, h = 2000, 2000
    intr_scaled = _scale_opencv_intrinsics(intr, (w, h), render_resolution).to(device=device)
    mesh_ = trimesh.load_mesh(GENERATED_MESH_PATH)
   
    extrinsics = []
    for f in frames:
        extrinsics.append(torch.tensor(f["transform_matrix"], dtype=torch.float32))
    
    
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

    if gt_mesh.exists():
        dtc_mesh = trimesh.load_mesh(gt_mesh)
        mesh_alignment, _ = align_meshes_similarity(mesh_, dtc_mesh)
        print("Alignment matrix for mesh alignment: \n", mesh_alignment)
        mesh_alignment = mesh_alignment @ AXIS_SWAP
        mesh_.apply_transform(mesh_alignment)

    if hasattr(mesh_.visual, "to_color"):
        mesh_.visual = mesh_.visual.to_color()
        

    vertices = np.asarray(mesh_.vertices).copy()
    faces = np.asarray(mesh_.faces)
    colors = np.asarray(mesh_.visual.vertex_colors[:, :3]) / 255.0

    alignment_mesh = Mesh(
        vertices=torch.from_numpy(vertices).to(device),
        faces=torch.from_numpy(faces).to(device),
        vertex_attrs=torch.from_numpy(colors).to(device).to(torch.float32)
    )

    extrr = torch.tensor(test_transforms["frames"][TEST_INDEX]["transform_matrix"], dtype=torch.float32, device=device)
    buf_align = renderer.render(mesh=alignment_mesh, 
                                extrinsics=extrr @ base_object_pose,
                                intrinsics=intr_scaled, return_types=["attr", "normal"])
    rgb_align = buf_align["attr"][:3]
    cv2.imwrite(f"{test_out_dir}/alignment_check.png", (rgb_align.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    colors = np.asarray(mesh_.visual.vertex_colors) / 255.0
    attr = torch.tensor(colors, dtype=torch.float32, device=device).requires_grad_(True)

    attrs = nn.Parameter(attr)
    sdf_init_time_ = time.time()

    dmtet = DmTetGeometry(
        res=128, 
        device = device.__str__(), 
        init_mesh=mesh_,
        scale=1.0
    )
    initial_mesh = dmtet.get_mesh(vertex_attr = attr)
    sdf_init_time = time.time() - sdf_init_time_
    
    object_transforms_fixed, _ = align_meshes_similarity(initial_mesh, dtc_mesh)
    print("Alignment matrix for the rendering part: \n", object_transforms_fixed)
    trimesh.Trimesh(vertices=initial_mesh.vertices.detach().cpu().numpy(), 
                    faces=initial_mesh.faces.detach().cpu().numpy()).export(f"{out_dir}/extracted_sdf_mesh.obj")
        
    
        
    object_pose = base_object_pose  @ torch.tensor(object_transforms_fixed, dtype=torch.float32, device=device)
    
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam([attrs], lr=lr)
        
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD([attrs], lr=lr, momentum=0.9)
    else:
        print(f"Invalid optimizer type: {optimizer_type}. Defaulting to Adam. ")
        optimizer = torch.optim.Adam([attrs], lr=lr)

    dmtet_optimizer = dmtet.getOptimizer(lr)
    
    optimization_time_ = time.time()
    for idx in range(num_iters): 
        
        mesh = dmtet.get_mesh(vertex_attr = attrs)
        total_loss =  torch.zeros((), device=device)
        optimizer.zero_grad()
        dmtet_optimizer.zero_grad()
      



        vmin = mesh.vertices.min(dim=0).values
        vmax = mesh.vertices.max(dim=0).values
        extents = vmax - vmin


        indices = (
            np.arange(N)
            if frames_per_iter == N
            else np.atleast_1d(rng.choice(N, size=frames_per_iter, replace=False))
        )

        for i in indices: 
            cam_extr = extrinsics_t[i]
            pose = cam_extr @ object_pose 

            buffer = renderer.render(mesh=mesh, extrinsics=pose, 
                                     intrinsics=intr_scaled, return_types=["attr","mask", "normal"]
                                     )
            
            img = buffer["attr"][:3]
            mask = buffer["mask"]
            normal = buffer["normal"]

            ref_img = ref_images[i]
            loss = criterion(img, ref_img[:3])
            mask_loss = criterion(ref_img[3], mask)
            sdf_loss = dmtet.get_sdf_loss()
            loss += mask_loss
            loss += sdf_loss
            total_loss += loss 

            if display_interval and (i % display_interval == 0 or i == num_iters -1):
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

                pred_rgba_vis = buffer["attr"][:4]   
                img = torch.cat([pred_rgba_vis, ref_img, normal_rgba], dim=2)   
                img = img.permute(1, 2, 0)                     
                img = img.clamp(0.0, 1.0)                        
                img_np = img.detach().cpu().numpy()
                save_image(f"{out_dir}/opt_{idx:04d}_{i:02d}.png", img_np)
                
        if idx in TEST_INTERVALS:
            test_cam_extr = torch.tensor(test_transforms["frames"][TEST_INDEX]["transform_matrix"], dtype=torch.float32, device=device)
            test_pose = test_cam_extr @ object_pose
            test_buffer = renderer.render(
                mesh=mesh, 
                extrinsics=test_pose, 
                intrinsics=intr_scaled, return_types=["attr","mask", "normal", "depth"]
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
            save_image(f"{test_out_dir}/test_{idx:04d}.png", test_img_np)
            

            
        total_loss = total_loss / len(indices)
        print(f"[iter {idx}] loss={float(total_loss.detach().cpu()):.6f}")
        total_loss.backward()


        optimizer.step()
        dmtet_optimizer.step()
    optimization_time = time.time() - optimization_time_
    final_mesh = dmtet.get_mesh(vertex_attr = attrs)
    output_glb_path.parent.mkdir(parents=True, exist_ok=True)
    color = attrs.detach().cpu().numpy()
    trimesh_obj = trimesh.Trimesh(vertices = final_mesh.vertices.cpu().detach().numpy(), 
                                  faces = final_mesh.faces.cpu().detach().numpy(),vertex_colors=color)
    trimesh_obj.export(output_glb_path)
    Path(f"{out_dir}/timings").mkdir(parents=True, exist_ok=True)
    with open(f"{out_dir}/timings/timing.txt", "a") as f:
        f.write(f"SDF calculation time: {sdf_init_time:.2f} seconds\n")
        f.write(f"Total optimization time for  {num_iters} iterations: {optimization_time:.2f} seconds\n")


        

    

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--yaml-config", type=Path, required=True)
    args = parser.parse_args()

    refine_latents(
        yaml_config_path=args.yaml_config,
    )


if __name__ == "__main__":
    main()

