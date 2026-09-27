import os
os.environ['ATTN_BACKEND'] = 'flash-attn'    
os.environ['SPCONV_ALGO'] = 'native' 
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))


from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
from trellis.renderers.mesh_renderer import MeshRenderer
from opt_utils.utility import display_image, save_image, align_meshes_similarity
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import imageio
from PIL import Image


import argparse
import yaml
import json
from typing import List, Dict, Tuple
import trimesh

RES = 512
TEST_INDEX = 4
TEST_INTERVALS = [0, 1, 50, 100, 200,  500, 799, 999]

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

AXIS_SWAP = rot_z(-np.pi/2)

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

def ensure_nchw(x):
    """Normalize tensor to NCHW format."""
    if x is None:
        return None
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)

    if x.dim() == 4:
        B, a, b, c = x.shape
        if c <= 4 and a > c:
            return x.permute(0, 3, 1, 2)
        if a <= 4 and a <= 16:
            return x
        if c <= 16:
            return x.permute(0, 3, 1, 2)
        return x
    elif x.dim() == 3:
        a, b, c = x.shape
        if c <= 4 and a > c:
            return x.permute(2, 0, 1).unsqueeze(0)
        if a <= 4 and b > 16 and c > 16:
            return x.unsqueeze(0)
        return x.unsqueeze(1)
    elif x.dim() == 2:
        return x.unsqueeze(0).unsqueeze(0)

    return x

def refine_latent(config_path: Path):
    config_yaml = yaml.safe_load(open(config_path, 'r'))

    lr = float(config_yaml['lr'])
    num_iters = int(config_yaml['num_iters'])
    transforms_json_path = Path(config_yaml['transforms_json'])
    frames_per_iter = int(config_yaml['frames_per_iter'])
    near = float(config_yaml.get('near', 0.01))
    far = float(config_yaml.get('far', 100.0))
    ssaa = int(config_yaml.get('ssaa', 1))
    display_interval = int(config_yaml.get('display_interval', 10))
    seed = int(config_yaml.get("seed", 1))
    gt_mesh = Path(config_yaml.get("gt_mesh", ""))
    out_dir = config_yaml.get("out_dir", "optimization_output")
    Path(out_dir).mkdir(parents=True, exist_ok=True)


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

    print("Loading Trellis2 ImageTo3DPipeline...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()

    prompt_img = Image.open(image_paths[prompt_idx])

    outputs = pipeline.run(
        prompt_img,
        seed=seed,
        return_latent=True,
    )
    orig_w, orig_h = prompt_img.size
    intr_scaled = _scale_opencv_intrinsics(intr, (orig_w, orig_h), RES).to(device=device)

    extrinsics = []
    for f in frames:
        extrinsics.append(torch.tensor(f["transform_matrix"], dtype=torch.float32))
    
    renderer = MeshRenderer(
        rendering_options={
                    "resolution": RES,
                    "near": near,
                    "far": far,
                    "ssaa": ssaa,
                },
                device=device.type,
    )
    

    slat = outputs['latent']
    mesh = outputs['mesh'][0]
    slat_opt = nn.Parameter(slat.feats.clone())
    slat_opt.requires_grad_(True)

    if config_yaml.get("optimizer", "") == "sgd":
        optimizer = torch.optim.SGD([slat_opt], lr=lr)
    else: 
        optimizer = torch.optim.Adam([slat_opt], lr=lr)
    extrinsics_t = torch.stack(extrinsics, dim=0).to(device=device)
    ref_images = torch.stack(
        [_resize_rgb_to_tensor(p, RES, device) for p in image_paths], dim=0
    )
    N = len(image_paths)
    rng = np.random.default_rng(seed=seed)

    if gt_mesh.exists():
        dtc_mesh = trimesh.load_mesh(gt_mesh)
        with torch.no_grad():
            alignment, _ = align_meshes_similarity(mesh, dtc_mesh) 
            alignment = torch.from_numpy(alignment @ AXIS_SWAP  ).to(device=device, dtype=torch.float32)
    else:
        print("No ground truth mesh provided, skipping alignment step.")
        alignment = torch.eye(4, device=device) 
    object_pose = base_object_pose @ alignment
    print(" Starting latent optimization...")

    l1_criterion = nn.L1Loss(reduction="mean")
    l2_criterion = nn.MSELoss(reduction="mean")
    gt_mask = None
    
    decoder = pipeline.models['slat_decoder_mesh']
    print(pipeline.models.keys())

    for i in range(num_iters + 1):
        indices = (
            np.arange(N)
            if frames_per_iter is None or frames_per_iter == N
            else np.atleast_1d(rng.choice(N, size=frames_per_iter, replace=False))
        )
        optimizer.zero_grad()
        slat_ = slat.replace(feats=slat_opt)
        mesh = decoder(slat_)[0]
        print(f"[Iter {i:03d}] after decode: slat_opt.requires_grad={slat_opt.requires_grad}, "\
      f"mesh.vertices.requires_grad={getattr(mesh.vertices, 'requires_grad', None)}")
        total_loss = torch.zeros((), device=device)

        for idx in indices: 
            ref_rgb = ref_images[idx][:3]
            extr = extrinsics_t[idx ] @ object_pose
            buffers = renderer.render(
                mesh=mesh, 
                extrinsics=extr,
                intrinsics=intr_scaled,
                return_types=["color"]
            )

            pred_rgb = buffers["color"]
           
            
            if pred_rgb.ndim == 4:
                pred_rgb = pred_rgb.squeeze(0)
                pred_rgb = pred_rgb.permute(2, 0, 1)
                
            rgb_loss = l2_criterion(pred_rgb, ref_rgb)

            loss = rgb_loss

                

            if display_interval and (idx % display_interval == 0 or idx == frames_per_iter - 1):
    
                pred_rgb    = ensure_nchw(pred_rgb)
                ref_rgb     = ensure_nchw(ref_rgb)
                gt_mask     = ensure_nchw(gt_mask) if gt_mask is not None else None






                img = torch.cat([pred_rgb, ref_rgb], dim=3)

                img = img[0].permute(1, 2, 0).clamp(0.0, 1.0)

                img_np = img.detach().cpu().numpy()
                save_image(f"{out_dir}/opt_{i:04d}_{idx:02d}.png", img_np)



            total_loss += loss
        total_loss = total_loss / len(indices)
        print(f"[Iter {i:03d}] Loss: {float(total_loss.detach().cpu()):.4f}")

        print("pred_rgb.requires_grad:", pred_rgb.requires_grad)
        print("rgb_loss.requires_grad:", rgb_loss.requires_grad)
        print("rgb_loss.grad_fn:", rgb_loss.grad_fn)
        print("total_loss.requires_grad:", total_loss.requires_grad)
        print("total_loss.grad_fn:", total_loss.grad_fn)

      

        total_loss.backward()
        optimizer.step()

    print("Latent optimization completed!")

            
                        
    





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters to refine latent")
    parser.add_argument("--yaml-config", type=Path, required=True, help="Path to the YAML config file" )
    args = parser.parse_args()
    refine_latent(Path(args.yaml_config))
