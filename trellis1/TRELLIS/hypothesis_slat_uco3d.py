"""SLAT refinement on a uCO3D sequence -- no GT mesh, no explicit alignment."""
import os
os.environ['ATTN_BACKEND'] = 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

import argparse
import json
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
import yaml
from PIL import Image
from scipy import ndimage
from torchvision import transforms

import utils3d.torch

from trellis.pipelines import TrellisImageTo3DPipeline
import trellis.renderers.mesh_renderer as mesh_renderer_module
from trellis.renderers.mesh_renderer import MeshRenderer
from trellis.representations.mesh.cube2mesh import MeshExtractResult
from trellis.models.structured_latent_vae.decoder_mesh import SLatMeshDecoder
from trellis.modules import sparse as sp
import trellis.models as models
from trellis.utils.loss_utils import l1_loss
from opt_utils.utility import save_image
from hypothesis_slat_downsample import downsample_sparse_slat, _perceptual_loss, ensure_nchw

RES = 512
DINO_RES = 518
HULL_RES = 64

CAMERA_CONVENTIONS = {
    "pytorch3d": np.diag([-1.0, -1.0, 1.0, 1.0]),
    "opencv": np.eye(4),
    "opengl": np.diag([1.0, -1.0, -1.0, 1.0]),
}


def _intrinsics_to_projection_fixed(
        intrinsics: torch.Tensor,
        near: float,
        far: float,
    ) -> torch.Tensor:
    """trellis.renderers.mesh_renderer.intrinsics_to_projection with the principal point's y term"""
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = 2 * cy - 1
    ret[2, 2] = far / (far - near)
    ret[2, 3] = near * far / (near - far)
    ret[3, 2] = 1.
    return ret


mesh_renderer_module.intrinsics_to_projection = _intrinsics_to_projection_fixed



def load_uco3d(data_dir: Path, convention: str, mask_threshold: float) -> Dict:
    meta = json.load(open(data_dir / "cameras.json", "r"))
    flip = CAMERA_CONVENTIONS[convention]

    rgbs, masks, c2ws = [], [], []
    for f in meta["frames"]:
        rgbs.append(Image.open(data_dir / f["path"]).convert("RGB"))
        masks.append(Image.open(data_dir / f["mask"]).convert("L"))
        c2ws.append(np.asarray(f["to_world"], dtype=np.float64) @ flip)

    W, H = rgbs[0].size
    K = np.array([[meta["fx"], 0.0, meta["cx"]],
                  [0.0, meta["fy"], meta["cy"]],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    split = meta.get("split", {"train": list(range(len(rgbs))), "test": []})
    masks_bin = np.stack([np.asarray(m, dtype=np.float32) / 255.0 > mask_threshold for m in masks])

    print(f"Loaded {len(rgbs)} frames ({W}x{H}), train={len(split['train'])} test={len(split['test'])}")
    return {
        "rgbs": rgbs,
        "masks": masks,
        "masks_bin": masks_bin,
        "c2w": np.stack(c2ws),
        "K": K,
        "size": (W, H),
        "split": split,
        "aabb": meta.get("aabb"),
    }



def dilate_masks(masks: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return masks
    return F.max_pool2d(masks[:, None].float(), 2 * radius + 1, stride=1, padding=radius)[:, 0] > 0


@torch.no_grad()
def carve(
    points: torch.Tensor,
    w2c: torch.Tensor,
    K: torch.Tensor,
    masks: torch.Tensor,
    view_frac: float,
    chunk: int = 2 ** 18,
) -> torch.Tensor:
    """A point survives if it lands inside the silhouette in >= view_frac of the views whose image"""
    V, H, W = masks.shape
    min_seen = max(2, V // 2)
    view_idx = torch.arange(V, device=points.device)[:, None]
    keep = torch.empty(points.shape[0], dtype=torch.bool, device=points.device)
    for s in range(0, points.shape[0], chunk):
        pc = torch.einsum('vij,pj->vpi', w2c[:, :3, :3], points[s:s + chunk]) + w2c[:, None, :3, 3]
        z = pc[..., 2].clamp_min(1e-8)
        u = K[0, 0] * pc[..., 0] / z + K[0, 2]
        v = K[1, 1] * pc[..., 1] / z + K[1, 2]
        in_img = (pc[..., 2] > 0) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
        hit = masks[view_idx, v.round().long().clamp(0, H - 1), u.round().long().clamp(0, W - 1)] & in_img
        seen, hits = in_img.sum(0), hit.sum(0)
        keep[s:s + chunk] = (seen >= min_seen) & (hits >= view_frac * seen)
    return keep


def _grid(lo: torch.Tensor, hi: torch.Tensor, n: int) -> Tuple[torch.Tensor, float]:
    axes = [torch.linspace(float(lo[k]), float(hi[k]), n, device=lo.device) for k in range(3)]
    pts = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1).reshape(-1, 3)
    return pts, float((hi - lo).max()) / (n - 1)


def _mask_radius(K: np.ndarray, c2w: np.ndarray, center: np.ndarray, voxel: float) -> int:
    """Pixel radius covering half a voxel diagonal at the closest camera, so carving at voxel"""
    min_depth = np.linalg.norm(c2w[:, :3, 3] - center, axis=1).min() * 0.75
    return int(math.ceil(K[0, 0] * voxel * math.sqrt(3) / 2 / min_depth))


def estimate_up(c2w: np.ndarray) -> np.ndarray:
    """An orbit around the object puts the camera centers on a (tilted) plane whose normal is the"""
    mean_up = -c2w[:, :3, 1].mean(0)
    centers = c2w[:, :3, 3]
    evals, evecs = np.linalg.eigh(np.cov((centers - centers.mean(0)).T))
    if evals[0] < 0.25 * evals[1]:
        up = evecs[:, 0] * np.sign(evecs[:, 0] @ mean_up)
    else:
        up = mean_up
    return up / np.linalg.norm(up)


@torch.no_grad()
def compute_normalization(data: Dict, cfg: Dict, prompt_view: int, device) -> Dict:
    """Similarity p_n = scale * R (p - center) that puts the visual hull, z-up, in the middle of"""
    train = data["split"]["train"]
    c2w = data["c2w"][train]
    K = data["K"]
    view_frac = float(cfg.get("hull_view_frac", 1.0))
    fill = float(cfg.get("hull_fill", 0.9))

    up = estimate_up(c2w)
    fwd = data["c2w"][prompt_view][:3, 2]
    fwd = fwd - (fwd @ up) * up
    fwd = fwd / np.linalg.norm(fwd)
    R = np.stack([np.cross(fwd, up), fwd, up])

    w2c_t = torch.from_numpy(np.linalg.inv(c2w)).float().to(device)
    K_t = torch.from_numpy(K).float().to(device)
    masks_t = torch.from_numpy(data["masks_bin"][train]).to(device)

    if data["aabb"] is not None:
        lo, hi = (torch.tensor(a, dtype=torch.float32, device=device) for a in data["aabb"])
    else:
        A, b = np.zeros((3, 3)), np.zeros(3)
        for m in c2w:
            P = np.eye(3) - np.outer(m[:3, 2], m[:3, 2])
            A, b = A + P, b + P @ m[:3, 3]
        c0 = np.linalg.solve(A, b)
        r0 = 0.5 * np.linalg.norm(c2w[:, :3, 3] - c0, axis=1).mean()
        lo = torch.tensor(c0 - r0, dtype=torch.float32, device=device)
        hi = torch.tensor(c0 + r0, dtype=torch.float32, device=device)

    center = extent = None
    for level in range(6):
        pts, voxel = _grid(lo, hi, 128)
        radius = _mask_radius(K, c2w, (lo + hi).cpu().numpy() / 2, voxel)
        inside = pts[carve(pts, w2c_t, K_t, dilate_masks(masks_t, radius), view_frac)]
        if inside.shape[0] == 0:
            raise RuntimeError(
                "Visual hull is empty -- check camera_convention / mask_threshold / hull_view_frac")
        aligned = inside.cpu().numpy().astype(np.float64) @ R.T
        a_lo, a_hi = aligned.min(0) - voxel, aligned.max(0) + voxel
        center = R.T @ ((a_lo + a_hi) / 2)
        extent = float((a_hi - a_lo).max())
        clipped = bool(((inside - lo).min() < 1.5 * voxel) or ((hi - inside).min() < 1.5 * voxel))
        print(f"[hull level {level}] {inside.shape[0]} pts, voxel {voxel:.4f}, "
              f"extent (z-up) {np.round(a_hi - a_lo, 3)}, center {np.round(center, 3)}"
              f"{' -- clipped by the box, growing it' if clipped else ''}")
        if level >= 1 and not clipped:
            break
        half = 0.75 * extent
        if clipped:
            half = max(half, 0.75 * float((hi - lo).max()))
        lo = torch.tensor(center - half, dtype=torch.float32, device=device)
        hi = torch.tensor(center + half, dtype=torch.float32, device=device)
    else:
        print("WARNING: visual hull still touches the carving box, the normalization may be off")

    scale = fill / extent
    world_to_norm = np.eye(4)
    world_to_norm[:3, :3] = scale * R
    world_to_norm[:3, 3] = -scale * R @ center
    return {"R": R, "center": center, "scale": scale, "up": up, "world_to_norm": world_to_norm}


def normalized_extrinsics(c2w: np.ndarray, norm: Dict) -> np.ndarray:
    """OpenCV w2c of the cameras in the normalized frame (rigid; depths in normalized units)."""
    R, c, s = norm["R"], norm["center"], norm["scale"]
    c2w_n = np.tile(np.eye(4), (c2w.shape[0], 1, 1))
    c2w_n[:, :3, :3] = R @ c2w[:, :3, :3]
    c2w_n[:, :3, 3] = s * (c2w[:, :3, 3] - c) @ R.T
    return np.linalg.inv(c2w_n)


@torch.no_grad()
def build_hull_structure(data: Dict, extr_n: np.ndarray, norm: Dict, cfg: Dict, device) -> torch.Tensor:
    """Sparse structure [N, 4] (batch, x, y, z) = shell of the HULL_RES^3 visual hull."""
    train = data["split"]["train"]
    view_frac = float(cfg.get("hull_view_frac", 1.0))
    shell = int(cfg.get("hull_shell", 2))
    K = data["K"]

    idx = torch.stack(torch.meshgrid(*[torch.arange(HULL_RES, device=device)] * 3, indexing='ij'), -1).reshape(-1, 3)
    pos = (idx.float() + 0.5) / HULL_RES - 0.5
    voxel_world = 1.0 / HULL_RES / norm["scale"]
    radius = _mask_radius(K, data["c2w"][train], norm["center"], voxel_world)
    masks_t = dilate_masks(torch.from_numpy(data["masks_bin"][train]).to(device), radius)
    occ = carve(
        pos,
        torch.from_numpy(extr_n[train]).float().to(device),
        torch.from_numpy(K).float().to(device),
        masks_t,
        view_frac,
    ).reshape(HULL_RES, HULL_RES, HULL_RES).cpu().numpy()

    labels, n = ndimage.label(occ, structure=np.ones((3, 3, 3)))
    if n > 1:
        sizes = ndimage.sum(occ, labels, index=range(1, n + 1))
        occ = labels == (1 + int(np.argmax(sizes)))
    solid = int(occ.sum())

    if shell > 0:
        interior = ndimage.binary_erosion(occ, structure=np.ones((3, 3, 3)), iterations=shell, border_value=0)
        occ = occ & ~interior
    print(f"Visual hull: {solid} solid voxels ({n} components, mask dilation {radius}px) "
          f"-> {int(occ.sum())} voxels with shell={shell}")

    coords = torch.from_numpy(np.argwhere(occ)).int()
    return torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1).to(device)



def crop_box(points_n: np.ndarray, w2c_n: np.ndarray, K: np.ndarray, margin: float) -> Tuple[int, int, int]:
    pc = points_n @ w2c_n[:3, :3].T + w2c_n[:3, 3]
    uv = pc[:, :2] / pc[:, 2:] * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
    (u0, v0), (u1, v1) = uv.min(0), uv.max(0)
    size = int(math.ceil(max(u1 - u0, v1 - v0) * margin))
    return int(round((u0 + u1) / 2 - size / 2)), int(round((v0 + v1) / 2 - size / 2)), size


def crop_intrinsics(K: np.ndarray, box: Tuple[int, int, int]) -> np.ndarray:
    """Normalized (TRELLIS / utils3d) intrinsics of the square crop `box` of the image."""
    x0, y0, size = box
    return np.array([[K[0, 0] / size, 0.0, (K[0, 2] - x0) / size],
                     [0.0, K[1, 1] / size, (K[1, 2] - y0) / size],
                     [0.0, 0.0, 1.0]], dtype=np.float32)


def crop_image(img: Image.Image, box: Tuple[int, int, int], res: int) -> torch.Tensor:
    """[C, res, res] in [0, 1]; PIL pads the parts of the box outside the image with 0."""
    x0, y0, size = box
    img = img.crop((x0, y0, x0 + size, y0 + size)).resize((res, res), Image.Resampling.BILINEAR)
    t = torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0)
    return t[None] if t.ndim == 2 else t.permute(2, 0, 1).contiguous()


def build_views(data: Dict, extr_n: np.ndarray, coords: torch.Tensor, margin: float, res: int, device) -> Dict:
    points_n = ((coords[:, 1:].float() + 0.5) / HULL_RES - 0.5).cpu().numpy()
    extr, intr, rgb, mask, valid, boxes = [], [], [], [], [], []
    full_frame = Image.new("L", data["size"], 255)
    for i in range(len(data["rgbs"])):
        box = crop_box(points_n, extr_n[i], data["K"], margin)
        boxes.append(box)
        extr.append(torch.from_numpy(extr_n[i]).float())
        intr.append(torch.from_numpy(crop_intrinsics(data["K"], box)))
        rgb.append(crop_image(data["rgbs"][i], box, res))
        mask.append(crop_image(data["masks"][i], box, res))
        valid.append(crop_image(full_frame, box, res))
    return {
        "extr": torch.stack(extr).to(device),
        "intr": torch.stack(intr).to(device),
        "rgb": torch.stack(rgb).to(device),
        "mask": torch.stack(mask).to(device),
        "valid": torch.stack(valid).to(device),
        "boxes": boxes,
    }



@torch.no_grad()
def encode_slat_from_views(
    coords: torch.Tensor,
    views: Dict,
    view_ids: List[int],
    dinov2_model,
    encoder,
    batch_size: int = 8,
) -> sp.SparseTensor:
    """TRELLIS' own encoding recipe (dataset_toolkits/extract_feature.py + encode_latent.py), with"""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    n_patch = DINO_RES // 14
    positions = (coords[:, 1:].float() + 0.5) / HULL_RES - 0.5
    feats = torch.zeros(coords.shape[0], 1024, device=coords.device)
    for s in range(0, len(view_ids), batch_size):
        ids = view_ids[s:s + batch_size]
        images = views["rgb"][ids] * views["mask"][ids]
        images = F.interpolate(images, size=(DINO_RES, DINO_RES), mode='bilinear', align_corners=False)
        images = normalize(images)
        tokens = dinov2_model(images, is_training=True)['x_prenorm'][:, dinov2_model.num_register_tokens + 1:]
        tokens = tokens.permute(0, 2, 1).reshape(len(ids), 1024, n_patch, n_patch)
        uv = utils3d.torch.project_cv(positions, views["extr"][ids], views["intr"][ids])[0] * 2 - 1
        feats += F.grid_sample(
            tokens, uv.unsqueeze(1), mode='bilinear', align_corners=False,
        ).squeeze(2).permute(0, 2, 1).sum(0)
    feats /= len(view_ids)
    return encoder(sp.SparseTensor(feats=feats.float(), coords=coords), sample_posterior=False)


@torch.no_grad()
def flow_slat_from_prompt(
    pipeline: TrellisImageTo3DPipeline,
    prompt: Image.Image,
    coords: torch.Tensor,
    seed: int,
    sampler_params: Dict,
) -> sp.SparseTensor:
    """TRELLIS stage 2 on the visual-hull structure (instead of the sampled one)."""
    cond = pipeline.get_cond([pipeline.preprocess_image(prompt)])
    torch.manual_seed(seed)
    return pipeline.sample_slat(cond, coords, sampler_params)



def render(renderer: MeshRenderer, mesh: MeshExtractResult, extr, intr, types,
           valid: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """All outputs [1, C, H, W]; with `valid` ([1, H', W'] view["valid"]) they are zeroed outside"""
    out = renderer.render(mesh=mesh, extrinsics=extr, intrinsics=intr, return_types=types)
    out = {k: ensure_nchw(out[k]) for k in types}
    if valid is not None:
        res = out[types[0]].shape[-1]
        if valid.shape[-1] != res:
            valid = F.interpolate(valid[None], size=(res, res), mode="area")[0]
        out = {k: v * valid[None] for k, v in out.items()}
    return out


def export_mesh(mesh: MeshExtractResult, path: Path, transform: Optional[np.ndarray] = None):
    v = mesh.vertices.detach().cpu().numpy()
    if transform is not None:
        v = v @ transform[:3, :3].T + transform[:3, 3]
    colors = (mesh.vertex_attrs[:, :3].detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    trimesh.Trimesh(v, mesh.faces.detach().cpu().numpy(), vertex_colors=colors, process=False).export(path)


def voxel_mesh(coords: torch.Tensor) -> MeshExtractResult:
    """Cubes of the sparse structure, in the normalized frame (for --check renders)."""
    occ = np.zeros((HULL_RES,) * 3, dtype=bool)
    c = coords[:, 1:].cpu().numpy()
    occ[c[:, 0], c[:, 1], c[:, 2]] = True
    boxes = trimesh.voxel.VoxelGrid(occ).as_boxes()
    device = coords.device
    return MeshExtractResult(
        vertices=torch.from_numpy((boxes.vertices + 0.5) / HULL_RES - 0.5).float().to(device),
        faces=torch.from_numpy(boxes.faces).long().to(device),
        vertex_attrs=torch.ones(len(boxes.vertices), 3, device=device),
    )


@torch.no_grad()
def evaluate(renderer, mesh, views, view_ids, out_path_fmt: Optional[str]) -> Dict[str, float]:
    psnrs, ious = [], []
    for vid in view_ids:
        pred = render(renderer, mesh, views["extr"][vid], views["intr"][vid], ["color", "mask", "normal"],
                      views["valid"][vid])
        gt_mask = views["mask"][vid][None]
        gt_rgb = views["rgb"][vid][None] * gt_mask
        pred_rgb = pred["color"].clamp(0, 1)
        psnrs.append(float(-10 * torch.log10(F.mse_loss(pred_rgb, gt_rgb).clamp_min(1e-10))))
        p, g = pred["mask"] > 0.5, gt_mask > 0.5
        ious.append(float((p & g).sum() / (p | g).sum().clamp_min(1)))
        if out_path_fmt is not None:
            vis = torch.cat([pred_rgb, gt_rgb, pred["normal"].clamp(0, 1)], dim=3)
            save_image(out_path_fmt.format(vid), vis[0].permute(1, 2, 0).cpu().numpy())
    return {"psnr": float(np.mean(psnrs)), "mask_iou": float(np.mean(ious))}



def transform_mesh(mesh: MeshExtractResult, A: torch.Tensor) -> MeshExtractResult:
    """Similarity A [4, 4] (uniform scale allowed) on the vertices, differentiable. Rebuilt rather"""
    out = MeshExtractResult(
        vertices=mesh.vertices @ A[:3, :3].T + A[:3, 3],
        faces=mesh.faces,
        vertex_attrs=mesh.vertex_attrs,
        res=mesh.res,
    )
    out.reg_loss = mesh.reg_loss
    return out


def _rot_z(theta: float) -> torch.Tensor:
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _axis_angle_to_matrix(w: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros((), device=w.device)
    skew = torch.stack([
        torch.stack([zero, -w[2], w[1]]),
        torch.stack([w[2], zero, -w[0]]),
        torch.stack([-w[1], w[0], zero]),
    ])
    return torch.linalg.matrix_exp(skew)


_UP_CANDIDATES = {
    "+z": torch.eye(3),
    "-z": torch.tensor([[1.0, 0, 0], [0, -1, 0], [0, 0, -1]]),
    "+y": torch.tensor([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]]),
    "-y": torch.tensor([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]]),
    "+x": torch.tensor([[0.0, 0, -1], [0, 1, 0], [1, 0, 0]]),
    "-x": torch.tensor([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
}


def _silhouette_loss(renderer, mesh, views, ids, gt_mask, gt_rgb, color_weight):
    """Mask L1 + color_weight * color L1 (object on black) over `ids`, and the mean mask IoU."""
    loss, iou = 0.0, 0.0
    for k, vid in enumerate(ids):
        pred = render(renderer, mesh, views["extr"][vid], views["intr"][vid], ["color", "mask"],
                      views["valid"][vid])
        loss = loss + l1_loss(pred["mask"], gt_mask[k:k + 1]) + color_weight * l1_loss(pred["color"], gt_rgb[k:k + 1])
        p, g = pred["mask"] > 0.5, gt_mask[k:k + 1] > 0.5
        iou += float((p & g).sum() / (p | g).sum().clamp_min(1))
    return loss / len(ids), iou / len(ids)


def estimate_alignment(
    mesh: MeshExtractResult,
    hull_coords: torch.Tensor,
    views: Dict,
    train: List[int],
    cfg: Dict,
    device,
) -> torch.Tensor:
    """Similarity A [4, 4]: canonical (pipeline) frame -> normalized frame, from the GT masks."""
    near, far = float(cfg.get("near", 0.1)), float(cfg.get("far", 10.0))
    color_weight = float(cfg.get("align_color_weight", 0.5))
    yaw_step = float(cfg.get("align_yaw_step", 10.0))
    ups = list(_UP_CANDIDATES) if cfg.get("align_search_up", False) else ["+z"]
    rng = np.random.default_rng(int(cfg.get("seed", 1)))

    def gt_at(res, ids):
        m = F.interpolate(views["mask"][ids], size=(res, res), mode="area")
        rgb = F.interpolate(views["rgb"][ids] * views["mask"][ids], size=(res, res), mode="area")
        return m, rgb

    verts = mesh.vertices.detach()
    hull = (hull_coords[:, 1:].float() + 0.5) / HULL_RES - 0.5
    h_lo, h_hi = hull.min(0).values, hull.max(0).values

    def bbox_similarity(R: torch.Tensor) -> torch.Tensor:
        v = verts @ R.T
        v_lo, v_hi = v.min(0).values, v.max(0).values
        s = (h_hi - h_lo).max() / (v_hi - v_lo).max()
        A = torch.eye(4, device=device)
        A[:3, :3] = s * R
        A[:3, 3] = (h_lo + h_hi) / 2 - s * (v_lo + v_hi) / 2
        return A

    search_res = int(cfg.get("align_search_res", 128))
    search_renderer = MeshRenderer({"resolution": search_res, "near": near, "far": far, "ssaa": 1}, device="cuda")
    search_ids = [train[k] for k in np.linspace(0, len(train) - 1, min(16, len(train))).round().astype(int)]
    gt_mask, gt_rgb = gt_at(search_res, search_ids)
    best = None
    with torch.no_grad():
        for up in ups:
            for yaw in np.arange(0.0, 360.0, yaw_step):
                R = (_rot_z(math.radians(yaw)) @ _UP_CANDIDATES[up]).to(device)
                A = bbox_similarity(R)
                loss, iou = _silhouette_loss(search_renderer, transform_mesh(mesh, A), views,
                                             search_ids, gt_mask, gt_rgb, color_weight)
                if best is None or float(loss) < best[0]:
                    best = (float(loss), iou, up, yaw, A)
    print(f"[align] grid search: up {best[2]}, yaw {best[3]:.0f} deg -> loss {best[0]:.4f}, mask IoU {best[1]:.3f}")
    A0 = best[4]

    refine_res = int(cfg.get("align_refine_res", 256))
    refine_renderer = MeshRenderer({"resolution": refine_res, "near": near, "far": far, "ssaa": 1}, device="cuda")
    c = verts.mean(0) @ A0[:3, :3].T + A0[:3, 3]
    w = torch.zeros(3, device=device, requires_grad=True)
    log_s = torch.zeros((), device=device, requires_grad=True)
    dt = torch.zeros(3, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([w, log_s, dt], lr=float(cfg.get("align_lr", 1e-2)))

    def compose() -> torch.Tensor:
        M = torch.exp(log_s) * _axis_angle_to_matrix(w)
        top = torch.cat([M @ A0[:3, :3], (c + M @ (A0[:3, 3] - c) + dt)[:, None]], dim=1)
        return torch.cat([top, torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)], dim=0)

    for it in range(int(cfg.get("align_iters", 300))):
        ids = list(rng.choice(train, size=min(8, len(train)), replace=False))
        gt_mask, gt_rgb = gt_at(refine_res, ids)
        optimizer.zero_grad()
        loss, iou = _silhouette_loss(refine_renderer, transform_mesh(mesh, compose()), views,
                                     ids, gt_mask, gt_rgb, color_weight)
        loss.backward()
        optimizer.step()
        if it % 50 == 0:
            print(f"[align] refine {it:03d}: loss {float(loss):.4f}, mask IoU {iou:.3f}, "
                  f"rot {math.degrees(float(w.norm())):.1f} deg, scale x{float(torch.exp(log_s)):.3f}")

    return compose().detach()



def prepare(cfg: Dict, device) -> Tuple[Dict, Dict, torch.Tensor, Dict]:
    data_dir = Path(cfg["data_dir"])
    assert (data_dir / "cameras.json").exists(), f"No cameras.json in {data_dir}"
    data = load_uco3d(data_dir, cfg.get("camera_convention", "pytorch3d"), float(cfg.get("mask_threshold", 0.5)))
    train = data["split"]["train"]
    prompt_view = train[int(cfg.get("prompt_idx", 0))]

    norm = compute_normalization(data, cfg, prompt_view, device)
    extr_n = normalized_extrinsics(data["c2w"], norm)
    coords = build_hull_structure(data, extr_n, norm, cfg, device)
    views = build_views(data, extr_n, coords, float(cfg.get("crop_margin", 1.1)), RES, device)
    return data, norm, coords, views


def check(cfg: Dict, out_dir: Path, device):
    """Cameras / hull / crops only -- no models. Quick enough for a login or dev node."""
    data, norm, coords, views = prepare(cfg, device)
    check_dir = out_dir / "check"
    check_dir.mkdir(parents=True, exist_ok=True)

    points_n = (coords[:, 1:].float() + 0.5) / HULL_RES - 0.5
    for vid in range(len(data["rgbs"])):
        uv = utils3d.torch.project_cv(points_n, views["extr"][vid], views["intr"][vid])[0]
        px = (uv * RES).long().clamp(0, RES - 1)
        hull = torch.zeros(RES, RES, device=device)
        hull[px[:, 1], px[:, 0]] = 1.0
        overlay = views["rgb"][vid].clone()
        overlay[0] = torch.maximum(overlay[0], hull * 0.8)
        gt = views["mask"][vid].expand(3, -1, -1)
        vis = torch.cat([overlay, gt], dim=2).permute(1, 2, 0).cpu().numpy()
        split = "train" if vid in data["split"]["train"] else "test"
        save_image(str(check_dir / f"hull_{split}_{vid:02d}.png"), vis)

    if device.type == "cuda":
        renderer = MeshRenderer({"resolution": RES, "near": 0.01, "far": 100.0, "ssaa": 1}, device="cuda")
        mesh = voxel_mesh(coords)
        metrics = evaluate(renderer, mesh, views, list(range(len(data["rgbs"]))),
                           str(check_dir / "render_hull_{:02d}.png"))
        print(f"Hull rendered through MeshRenderer vs GT masks: mask IoU {metrics['mask_iou']:.3f}")
    else:
        print("No GPU: skipped the MeshRenderer check (overlays only).")
    print(f"Check images written to {check_dir}")


def refine_latent(cfg: Dict, out_dir: Path, device):
    lr = float(cfg["lr"])
    num_iters = int(cfg["num_iters"])
    frames_per_iter = int(cfg.get("frames_per_iter", 8))
    seed = int(cfg.get("seed", 1))
    display_interval = int(cfg.get("display_interval", 25))
    eval_interval = int(cfg.get("eval_interval", 50))
    lambda_mask = float(cfg.get("lambda_mask", 1.0))
    lambda_color = float(cfg.get("lambda_color", 1.0))
    lambda_ssim = float(cfg.get("lambda_ssim", 0.2))
    lambda_lpips = float(cfg.get("lambda_lpips", 0.2))
    reg_weight = float(cfg.get("reg_weight", 0.5))
    latent_l2_weight = float(cfg.get("latent_l2_weight", 0.0))
    downsample_factor = int(cfg.get("downsample_factor", 1))
    slat_init = cfg.get("slat_init", "encode")

    torch.manual_seed(seed)
    data, norm, coords, views = prepare(cfg, device)
    train = data["split"]["train"]
    test = data["split"]["test"]
    prompt_view = train[int(cfg.get("prompt_idx", 0))]

    with open(out_dir / "normalization.json", "w") as f:
        json.dump({
            "camera_convention": cfg.get("camera_convention", "pytorch3d"),
            "world_to_norm": norm["world_to_norm"].tolist(),
            "norm_to_world": np.linalg.inv(norm["world_to_norm"]).tolist(),
            "scale": norm["scale"],
            "center": norm["center"].tolist(),
            "up": norm["up"].tolist(),
            "num_voxels": int(coords.shape[0]),
            "crop_boxes_xy_size": views["boxes"],
        }, f, indent=2)

    print("Loading TRELLIS ImageTo3DPipeline...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()


    prompt_img = data["rgbs"][prompt_view].copy()
    prompt_img.putalpha(data["masks"][prompt_view])
    outputs = pipeline.run(
        prompt_img,
        seed=seed,
        return_latent=True,
    )
    slat = outputs['latent']
    print(f"SLAT (pipeline.run): {slat.feats.shape[0]} voxels, feats mean {slat.feats.mean():.3f} std {slat.feats.std():.3f}")

    slat, slat_res = downsample_sparse_slat(slat, res=HULL_RES, factor=downsample_factor, device=device)
    torch.save({"feats": slat.feats.cpu(), "coords": slat.coords.cpu(), "res": slat_res}, out_dir / "slat_init.pt")

    decoder = SLatMeshDecoder(
        resolution=slat_res,
        model_channels=768,
        latent_channels=8,
        num_blocks=12,
        num_heads=12,
        mlp_ratio=4,
        attn_mode="swin",
        window_size=8,
        use_fp16=True,
        representation_config={"use_color": True},
    )
    decoder.load_state_dict(pipeline.models['slat_decoder_mesh'].state_dict())
    decoder.to(device=device).train()
    decoder.requires_grad_(False)

    renderer = MeshRenderer(
        rendering_options={
            "resolution": RES,
            "near": float(cfg.get("near", 0.1)),
            "far": float(cfg.get("far", 10.0)),
            "ssaa": int(cfg.get("ssaa", 1)),
        },
        device=device.type,
    )

    vis_dir = out_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    slat_init_feats = slat.feats.clone().detach()
    slat_opt = nn.Parameter(slat.feats.clone().detach())
    if cfg.get("optimizer", "adam") == "sgd":
        optimizer = torch.optim.SGD([slat_opt], lr=lr)
    else:
        optimizer = torch.optim.Adam([slat_opt], lr=lr)

    metrics_file = open(out_dir / "metrics.jsonl", "w")
    rng = np.random.default_rng(seed=seed)
    n_views = min(frames_per_iter, len(train))

    with torch.no_grad():
        mesh = decoder(sp.SparseTensor(feats=slat_opt, coords=slat.coords))[0]
        export_mesh(mesh, out_dir / "mesh_init_canonical.glb")

    if cfg.get("align_to_hull", True):
        alignment = estimate_alignment(mesh, coords, views, train, cfg, device)
    else:
        alignment = torch.eye(4, device=device)
    with open(out_dir / "alignment.json", "w") as f:
        json.dump({"canonical_to_norm": alignment.cpu().tolist()}, f, indent=2)
    with torch.no_grad():
        mesh = transform_mesh(mesh, alignment)
        export_mesh(mesh, out_dir / "mesh_init.glb")
        metrics = evaluate(renderer, mesh, views, test or train[:4], str(vis_dir / "aligned_init_{:02d}.png"))
    print(f"Aligned initial mesh: {metrics}")

    print("Starting latent optimization...")
    for i in range(num_iters + 1):
        optimizer.zero_grad()
        mesh = transform_mesh(decoder(sp.SparseTensor(feats=slat_opt, coords=slat.coords))[0], alignment)

        if i % eval_interval == 0 or i == num_iters:
            fmt = str(vis_dir / f"test_{i:04d}_{{:02d}}.png")
            metrics = {"iter": i, "train": evaluate(renderer, mesh, views, train, None)}
            if test:
                metrics["test"] = evaluate(renderer, mesh, views, test, fmt)
            print(f"[Iter {i:04d}] eval {metrics}")
            metrics_file.write(json.dumps(metrics) + "\n")
            metrics_file.flush()
        if i == num_iters:
            break

        img_loss = torch.zeros((), device=device)
        mask_loss = torch.zeros((), device=device)
        for vid in rng.choice(train, size=n_views, replace=False):
            pred = render(renderer, mesh, views["extr"][vid], views["intr"][vid], ["color", "mask", "normal"],
                          views["valid"][vid])
            gt_mask = views["mask"][vid][None]
            gt_rgb = views["rgb"][vid][None] * gt_mask

            mask_loss = mask_loss + l1_loss(pred["mask"], gt_mask)
            color_terms = _perceptual_loss(gt_rgb, pred["color"], "color", lambda_ssim, lambda_lpips)
            img_loss = img_loss + color_terms["color_loss_perceptual"]

            if i % display_interval == 0:
                vis = torch.cat([pred["color"].clamp(0, 1), gt_rgb, pred["normal"].clamp(0, 1)], dim=3)
                save_image(str(vis_dir / f"train_{i:04d}_{vid:02d}.png"),
                           vis[0].detach().permute(1, 2, 0).cpu().numpy())

        mask_loss = mask_loss / n_views
        img_loss = img_loss / n_views
        reg_loss = mesh.reg_loss if mesh.reg_loss is not None else torch.zeros((), device=device)
        latent_loss = ((slat_opt - slat_init_feats) ** 2).mean()
        total_loss = (lambda_mask * mask_loss + lambda_color * img_loss
                      + reg_weight * reg_loss + latent_l2_weight * latent_loss)
        total_loss.backward()
        optimizer.step()

        print(f"[Iter {i:04d}] loss {total_loss.item():.4f} | mask {mask_loss.item():.4f} "
              f"color {img_loss.item():.4f} reg {float(reg_loss):.4f} latent {latent_loss.item():.4f} "
              f"| grad {slat_opt.grad.norm().item():.4f} | verts {mesh.vertices.shape[0]}")

    metrics_file.close()
    with torch.no_grad():
        mesh = decoder(sp.SparseTensor(feats=slat_opt, coords=slat.coords))[0]
        export_mesh(mesh, out_dir / "mesh_final_canonical.glb")
        mesh = transform_mesh(mesh, alignment)
        export_mesh(mesh, out_dir / "mesh_final.glb")
        export_mesh(mesh, out_dir / "mesh_final_world.glb", np.linalg.inv(norm["world_to_norm"]))
    torch.save({"feats": slat_opt.detach().cpu(), "coords": slat.coords.cpu(), "res": slat_res},
               out_dir / "slat_final.pt")
    print(f"Latent optimization completed! Outputs in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refine a TRELLIS SLAT on a uCO3D sequence")
    parser.add_argument("--yaml-config", type=Path, required=True, help="Path to the YAML config file")
    parser.add_argument("--check", action="store_true",
                        help="Only build cameras / hull / crops and write debug images (no models, quick)")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.yaml_config, "r"))
    out_dir = Path(cfg.get("out_dir", "optimization_output/uco3d"))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.check:
        check(cfg, out_dir, device)
    else:
        assert device.type == "cuda", "Optimization needs a GPU -- submit it as a batch job"
        refine_latent(cfg, out_dir, device)
