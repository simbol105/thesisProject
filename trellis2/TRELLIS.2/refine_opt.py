
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

import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.renderers import EnvMap
from trellis2.renderers.pbr_mesh_renderer import PbrMeshRenderer
from trellis2.representations import MeshWithVoxel
from trellis2.utils import render_utils


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
    img = Image.open(img_path).convert("RGB")
    if img.size != (render_resolution, render_resolution):
        img = img.resize((render_resolution, render_resolution), Image.Resampling.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
    t = t.permute(2, 0, 1).contiguous()
    return t


def _scale_opencv_intrinsics(intr: Dict, orig_wh: Tuple[int, int], render_resolution: int) -> torch.Tensor:
    fx, fy, cx, cy = float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"])
    orig_w, orig_h = orig_wh

    sx = render_resolution / float(orig_w)
    sy = render_resolution / float(orig_h)

    fx_s, fy_s = fx * sx, fy * sy
    cx_s, cy_s = cx * sx, cy * sy

    intr_scaled = torch.tensor(
        [[fx_s, 0.0, cx_s], [0.0, fy_s, cy_s], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    return intr_scaled


@torch.no_grad()
def _freeze_module_params(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


def _decode_latent_differentiable(
    pipeline: Trellis2ImageTo3DPipeline,
    shape_slat_opt,
    tex_slat_opt,
    resolution: int,
) -> List[MeshWithVoxel]:
   
    print(
        "shape_slat_opt coords:", 
        shape_slat_opt.coords.shape, 
        "feats:", 
        shape_slat_opt.feats.shape,
    )
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
    transforms_json_path: Path,
    yaml_config_path: Path,
    output_glb_path: Optional[Path] = None,
):
    config_yaml = yaml.safe_load(open(yaml_config_path, "r"))
    transforms = _load_json(transforms_json_path)

    lr = float(config_yaml["lr"])
    num_iters = int(config_yaml["num_iters"])
    envmap_path = Path(config_yaml["envmap_path"])
    assert envmap_path.exists(), f"Envmap path does not exist: {envmap_path}"

    pipeline_type = str(config_yaml.get("pipeline_type", "512"))
    frames_per_iter_cfg = config_yaml.get("frames_per_iter", None)
    near = float(config_yaml.get("near", 0.01))
    far = float(config_yaml.get("far", 100.0))
    ssaa = int(config_yaml.get("ssaa", 1))
    peel_layers = int(config_yaml.get("peel_layers", 8))
    loss_type = str(config_yaml.get("loss_type", "mse"))
    latent_l2_weight = float(config_yaml.get("latent_l2_weight", 0.0))
    grad_clip_norm = float(config_yaml.get("grad_clip_norm", 0.0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(True)

    frames = transforms["frames"]
    base_dir = transforms_json_path.parent
    image_basenames = [f["file_path"] for f in frames]
    image_paths: List[Path] = [_try_image_path(base_dir / b) for b in image_basenames]

    intr = transforms["intrinsic"]
    prompt_idx = int(config_yaml.get("prompt_idx", np.random.randint(0, len(image_paths))))

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.low_vram = False
    pipeline.cuda()

    envmap_img = cv2.imread(str(envmap_path), cv2.IMREAD_UNCHANGED)
    assert envmap_img is not None, f"Failed to read envmap: {envmap_path}"
    envmap_img = cv2.cvtColor(envmap_img, cv2.COLOR_BGR2RGB)
    env_map = EnvMap(torch.tensor(envmap_img, dtype=torch.float32, device=device))

    prompt_img = Image.open(image_paths[prompt_idx]).convert("RGB")
    _, (shape_slat, tex_slat, res) = pipeline.run(
        prompt_img,
        return_latent=True,
        pipeline_type=pipeline_type,
    )
    
    shape_slat = shape_slat.replace(
        coords=shape_slat.coords.detach(),
        feats=shape_slat.feats.detach(),
    )
    tex_slat = tex_slat.replace(
        coords=tex_slat.coords.detach(),
        feats=tex_slat.feats.detach(),
    )
    render_resolution = 256

    orig_img = Image.open(image_paths[prompt_idx]).convert("RGB")
    orig_w, orig_h = orig_img.size
    intr_scaled = _scale_opencv_intrinsics(intr, (orig_w, orig_h), render_resolution).to(device=device)

    extrinsics = [torch.tensor(f["transform_matrix"], dtype=torch.float32) for f in frames]
    extrinsics_t = torch.stack(extrinsics, dim=0).to(device=device)

    shape_coords = shape_slat.coords.detach()
    tex_coords   = tex_slat.coords.detach()

    shape_feats_init = shape_slat.feats.detach().clone().to(device)
    tex_feats_init   = tex_slat.feats.detach().clone().to(device)

    shape_feats = nn.Parameter(shape_feats_init.clone())
    tex_feats   = nn.Parameter(tex_feats_init.clone())

    optimizer = torch.optim.Adam([shape_feats, tex_feats], lr=lr)

    N = len(image_paths)
    if frames_per_iter_cfg is None:
        frames_per_iter = max(1, min(N, N // 4))
    else:
        frames_per_iter = max(1, min(int(frames_per_iter_cfg), N))
    rng = np.random.default_rng(seed=int(config_yaml.get("seed", 0)))

    for it in range(num_iters):
        optimizer.zero_grad(set_to_none=True)

        shape_slat_opt = shape_slat.replace(feats=shape_feats, coords=shape_coords)
        tex_slat_opt   = tex_slat.replace(feats=tex_feats, coords=tex_coords)

        mesh = _decode_latent_differentiable(
            pipeline, shape_slat_opt, tex_slat_opt, render_resolution
        )[0]


        if frames_per_iter == N:
            indices = np.arange(N)
        else:
            indices = rng.choice(N, size=frames_per_iter, replace=False)

        total_loss = 0.0
        
        renderer = PbrMeshRenderer(
        rendering_options={
            "resolution": render_resolution,
            "near": near,
            "far": far,
            "ssaa": ssaa,
            "peel_layers": peel_layers,
        },
        device=device.type,
    )
        
        for idx in indices:
            extr = extrinsics_t[int(idx)]

            buffers = renderer.render(
                mesh=mesh,
                extrinsics=extr,
                intrinsics=intr_scaled,
                envmap=env_map,
            )
            pred = buffers["shaded"]
            gt   = _resize_rgb_to_tensor(
                image_paths[int(idx)], render_resolution, device=device
            )

            if loss_type == "l1":
                diff = pred - gt
                per_pix = diff.abs()
            else:
                diff = pred - gt
                per_pix = diff.pow(2)

            mask = buffers.get("mask", None)
            if mask is not None:
                per_pix = per_pix * mask.unsqueeze(0)
                denom = mask.sum().clamp_min(1.0)
                loss = per_pix.sum() / denom
            else:
                loss = per_pix.mean()

            total_loss = total_loss + loss

        if latent_l2_weight > 0:
            l2 = (shape_feats - shape_feats_init).pow(2).mean() + \
                 (tex_feats   - tex_feats_init).pow(2).mean()
            total_loss = total_loss + latent_l2_weight * l2

        print(f"[iter {it}] loss={float(total_loss.detach().cpu()):.6f}")

        total_loss.backward()

        if grad_clip_norm and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_([shape_feats, tex_feats], grad_clip_norm)

        optimizer.step()

        with torch.no_grad():
            for p, p_init in [(shape_feats, shape_feats_init),
                              (tex_feats,   tex_feats_init)]:
                if torch.isnan(p).any() or torch.isinf(p).any():
                    p.copy_(p_init)
                p.clamp_(-10.0, 10.0)


    with torch.no_grad():
        shape_slat_opt = shape_slat.replace(feats=shape_feats, coords=shape_coords)
        tex_slat_opt = tex_slat.replace(feats=tex_feats, coords=tex_coords)
        final_mesh = _decode_latent_differentiable(
            pipeline, shape_slat_opt, tex_slat_opt, render_resolution
        )[0]

    if output_glb_path is not None:
        import o_voxel
        from o_voxel import postprocess
        import imageio
        from utils3d import render_utils

        video = render_utils.make_pbr_vis_frames(
            render_utils.render_video(final_mesh, envmap=env_map)
        )
        imageio.mimsave("opt.mp4", video, fps=15)

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
        glb.export(str(output_glb_path), extension_webp=False)
        print(f"Exported: {output_glb_path}")

    return (
        final_mesh,
        shape_slat.replace(feats=shape_feats, coords=shape_coords),
        tex_slat.replace(feats=tex_feats, coords=tex_coords),
    )



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transforms-json", type=Path, required=True)
    parser.add_argument("--yaml-config", type=Path, required=True)
    parser.add_argument("--output-glb", type=Path, default=None)
    args = parser.parse_args()

    refine_latents(
        transforms_json_path=args.transforms_json,
        yaml_config_path=args.yaml_config,
        output_glb_path=args.output_glb,
    )


if __name__ == "__main__":
    main()
