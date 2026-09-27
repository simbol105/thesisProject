"""Optimize a SAM3D SLAT against posed reference images."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
import trimesh
import yaml
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR / "notebook"))
from inference import Inference, load_image


def load_transforms(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def find_image(base_dir: Path, name: str) -> Path:
    candidate = base_dir / name
    if candidate.suffix:
        candidates = [candidate]
    else:
        candidates = [candidate.with_suffix(ext) for ext in (".png", ".jpg", ".jpeg", ".webp")]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find reference image for {name}")


def resize_image(path: Path, resolution: int, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGBA").resize(
        (resolution, resolution), Image.Resampling.BILINEAR
    )
    tensor = torch.from_numpy(np.asarray(image).astype(np.float32) / 255.0)
    return tensor[..., :3].permute(2, 0, 1).to(device)


def scaled_intrinsics(
    intrinsics: Dict[str, float], image_size: Tuple[int, int]
) -> torch.Tensor:
    width = float(intrinsics.get("width", intrinsics.get("w", image_size[0])))
    height = float(intrinsics.get("height", intrinsics.get("h", image_size[1])))
    return torch.tensor(
        [
            [float(intrinsics["fx"]) / width, 0.0, float(intrinsics["cx"]) / width],
            [0.0, float(intrinsics["fy"]) / height, float(intrinsics["cy"]) / height],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def intrinsics_to_projection(
    intrinsics: torch.Tensor, near: float, far: float
) -> torch.Tensor:
    """Convert normalized OpenCV intrinsics to an OpenGL projection matrix."""
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    projection = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    projection[0, 0] = 2 * fx
    projection[1, 1] = 2 * fy
    projection[0, 2] = 2 * cx - 1
    projection[1, 2] = -2 * cy + 1
    projection[2, 2] = far / (far - near)
    projection[2, 3] = near * far / (near - far)
    projection[3, 2] = 1.0
    return projection


class NVDiffRasterMeshRenderer:
    """Differentiable vertex-color mesh renderer backed by nvdiffrast."""

    def __init__(self, resolution: int, near: float, far: float, ssaa: int, device: torch.device):
        try:
            import nvdiffrast.torch as dr
        except ImportError as error:
            raise ImportError(
                "nvdiffrast is required for NVDiffRasterMeshRenderer. "
                "Install it in the SAM3D environment."
            ) from error
        if device.type != "cuda":
            raise RuntimeError("nvdiffrast RasterizeCudaContext requires a CUDA device")
        self.dr = dr
        self.resolution = resolution
        self.near = near
        self.far = far
        self.ssaa = ssaa
        self.device = device
        self.glctx = dr.RasterizeCudaContext(device=device)

    def render(self, mesh, extrinsics: torch.Tensor, intrinsics: torch.Tensor, return_types):
        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
            empty_color = torch.zeros(
                (3, self.resolution, self.resolution), device=self.device
            )
            return {render_type: empty_color for render_type in return_types}

        vertices = mesh.vertices.unsqueeze(0)
        vertices_homo = torch.cat(
            [vertices, torch.ones_like(vertices[..., :1])], dim=-1
        )
        projection = intrinsics_to_projection(intrinsics, self.near, self.far)
        full_projection = (projection @ extrinsics).unsqueeze(0)
        vertices_camera = torch.bmm(
            vertices_homo, extrinsics.unsqueeze(0).transpose(-1, -2)
        )
        vertices_clip = torch.bmm(
            vertices_homo, full_projection.transpose(-1, -2)
        )
        faces = mesh.faces.to(device=self.device, dtype=torch.int32)
        rast, _ = self.dr.rasterize(
            self.glctx,
            vertices_clip,
            faces,
            (self.resolution * self.ssaa, self.resolution * self.ssaa),
        )

        rendered = {}
        for render_type in return_types:
            if render_type == "color":
                vertex_colors = mesh.vertex_attrs[:, :3].contiguous()
                image = self.dr.interpolate(vertex_colors.unsqueeze(0), rast, faces)[0]
            elif render_type == "mask":
                image = (rast[..., -1:] > 0).float()
            elif render_type == "depth":
                image = self.dr.interpolate(
                    vertices_camera[..., 2:3].contiguous(), rast, faces
                )[0]
            else:
                raise ValueError(f"Unsupported render type: {render_type}")

            image = self.dr.antialias(image, rast, vertices_clip, faces)
            if self.ssaa > 1:
                image = torch.nn.functional.interpolate(
                    image.permute(0, 3, 1, 2),
                    (self.resolution, self.resolution),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
            else:
                image = image.permute(0, 3, 1, 2)
            rendered[render_type] = image.squeeze(0)
        return rendered


def mesh_to_trimesh(mesh) -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=mesh.vertices.detach().cpu().numpy(),
        faces=mesh.faces.detach().cpu().numpy(),
        process=False,
    )


def load_alignment_function(config: dict):
    trellis_root = Path(
        config.get(
            "trellis_root",
            "/home/fr/fr_fr/fr_ba122/projects/trellis1/TRELLIS",
        )
    ).expanduser().resolve()
    if not trellis_root.is_dir():
        raise FileNotFoundError(f"TRELLIS root does not exist: {trellis_root}")
    sys.path.insert(0, str(trellis_root))
    try:
        from opt_utils.utility import align_meshes_fixed_scale_ransac
    except ImportError as error:
        raise ImportError(
            "Could not import TRELLIS alignment function from "
            f"{trellis_root / 'opt_utils' / 'utility.py'}"
        ) from error
    return align_meshes_fixed_scale_ransac


def optimize(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    transforms_path = Path(config["transforms_json"]).expanduser().resolve()
    output_dir = Path(config["out_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    transforms = load_transforms(transforms_path)
    frames = transforms["frames"]
    base_object_pose = np.array(transforms["object_pose"], dtype=np.float32)
    image_paths = [find_image(transforms_path.parent, frame["file_path"]) for frame in frames]
    prompt_idx = int(config.get("prompt_idx", 0))
    resolution = int(config.get("resolution", 512))
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(config["checkpoint_path"]).expanduser().resolve()
    inference = Inference(str(checkpoint_path), compile=False)
    pipeline = inference._pipeline
    prompt_image = load_image(str(image_paths[prompt_idx]))
    output = pipeline.run(
        prompt_image,
        None,
        seed=int(config.get("seed", 42)),
        with_mesh_postprocess=False,
        with_texture_baking=False,
        with_layout_postprocess=False,
        use_vertex_color=True,
        decode_formats=["mesh", "gaussian"],
        return_slat=True,
    )
    slat = output["slat"]
    print("slat feats shape: ", slat.feats.shape)
    slat_opt = nn.Parameter(slat.feats.detach().float().clone())
    slat_opt.requires_grad_(True)
    optimizer_cls = torch.optim.SGD if config.get("optimizer", "adam").lower() == "sgd" else torch.optim.Adam
    optimizer = optimizer_cls([slat_opt], lr=float(config.get("lr", 1e-2)))
    decoder = pipeline.models["slat_decoder_mesh"]
    decoder.train()
    

    initial_mesh = output["mesh"][0]
    gt_mesh_path = Path(
        config.get("gt_mesh", transforms_path.parent / "model.glb")
    ).expanduser().resolve()
    if not gt_mesh_path.is_file():
        raise FileNotFoundError(f"Ground-truth mesh does not exist: {gt_mesh_path}")
    gt_mesh = trimesh.load_mesh(gt_mesh_path)
    gt_mesh.apply_transform(base_object_pose)
    align_meshes_fixed_scale_ransac = load_alignment_function(config)
    alignment = torch.as_tensor(
        align_meshes_fixed_scale_ransac(initial_mesh, gt_mesh),
        dtype=torch.float32,
        device=device,
    )
    print(f"Mesh alignment transform:\n{alignment}", flush=True)

    extrinsics = torch.tensor(
        np.asarray([frame["transform_matrix"] for frame in frames]),
        dtype=torch.float32,
        device=device,
    )
    intrinsics = scaled_intrinsics(
        transforms["intrinsic"], (prompt_image.shape[1], prompt_image.shape[0])
    ).to(device)
    reference_images = torch.stack(
        [resize_image(path, resolution, device) for path in image_paths]
    )

    renderer = NVDiffRasterMeshRenderer(
        resolution=resolution,
        near=float(config.get("near", 0.01)),
        far=float(config.get("far", 100.0)),
        ssaa=int(config.get("ssaa", 1)),
        device=device,
    )
    rng = np.random.default_rng(int(config.get("seed", 42)))
    num_iters = int(config.get("num_iters", 100))
    frames_per_iter = int(config.get("frames_per_iter", len(frames)))
    display_interval = int(config.get("display_interval", 10))
    criterion = nn.MSELoss()

    for iteration in range(num_iters + 1):
        count = min(frames_per_iter, len(frames))
        indices = rng.choice(len(frames), size=count, replace=False)
        optimizer.zero_grad()
        optimized_slat = slat.replace(feats=slat_opt)
        mesh = decoder(optimized_slat)[0]
        total_loss = torch.zeros((), device=device)
        last_preview = None
        for index in indices:
            buffers = renderer.render(
                mesh=mesh,
                extrinsics=extrinsics[index] @ alignment,
                intrinsics=intrinsics,
                return_types=["color"],
            )
            predicted = buffers["color"]
            if predicted.ndim == 4:
                predicted = predicted[0].permute(2, 0, 1)
            total_loss = total_loss + criterion(predicted, reference_images[index])
            last_preview = torch.cat([predicted, reference_images[index]], dim=2)

        total_loss = total_loss / len(indices)
        if iteration < num_iters:
            total_loss.backward()
            optimizer.step()
        print(f"[Iter {iteration:04d}] loss={float(total_loss.detach().cpu()):.6f}", flush=True)
        print(f" SLAT grad norm: {slat_opt.grad.detach().norm().item():.6f}", flush=True)

        if last_preview is not None and iteration % display_interval == 0:
            image = (last_preview.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            imageio.imwrite(output_dir / f"opt_{iteration:04d}.png", image)

    final_slat = slat.replace(feats=slat_opt.detach())
    final_mesh = decoder(final_slat)[0]
    mesh_to_trimesh(final_mesh).export(output_dir / "mesh.glb")
    torch.save(
        {"coords": final_slat.coords.detach().cpu(), "feats": final_slat.feats.detach().cpu()},
        output_dir / "slat.pt",
    )
    print(f"Saved optimized mesh and SLAT to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml-config", type=Path, required=True)
    optimize(parser.parse_args().yaml_config.expanduser().resolve())
