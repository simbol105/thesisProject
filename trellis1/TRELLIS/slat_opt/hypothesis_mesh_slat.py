"""SLAT (or cubefeats) optimization starting from an existing reconstruction (recgen / SAM3D /
TRELLIS), against the DTC ground truth.
"""

import os

os.environ.setdefault('SPCONV_ALGO', 'native')

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
import yaml
from PIL import Image
from torchvision import transforms

import open3d as o3d
import utils3d.torch

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils
from trellis.renderers.mesh_renderer import MeshRenderer
from opt_utils.utility import save_image, align_meshes_similarity, align_meshes, align_meshes_fixed_scale_ransac
from trellis.representations.mesh.cube2mesh import MeshExtractResult
from trellis.modules import sparse as sp
import trellis.models as models
from trellis.utils.loss_utils import l1_loss, smooth_l1_loss, ssim, lpips
from dataset_toolkits.utils import sphere_hammersley_sequence
from slat_optimization import select_alignment, split_similarity, chamfer_to_gt, grad_wrt

RES = 512
DECODER_RES = 64
DINO_RES = 518
TRELLIS_CAM_RADIUS = 2
TRELLIS_CAM_FOV_DEG = 40

SLAT_ENCODER_CKPT = "microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16"
MESH_KEYS = {"recgen": "recgen_mesh_path", "sam3d": "sam3d_mesh_path", "trellis": "trellis_mesh_path",
             "trellis2": "trellis2_mesh_path"}



def _load_json(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def _try_image_path(base_path_no_ext: Path) -> Path:
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        cand = base_path_no_ext.with_suffix(ext)
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Could not find image for: {base_path_no_ext}")


def _scale_opencv_intrinsics(intr: Dict, orig_wh: Tuple[int, int]) -> torch.Tensor:
    """OpenCV pixel intrinsics -> the normalized (resolution independent) form the renderer wants."""
    fx, fy, cx, cy = float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"])
    orig_w, orig_h = orig_wh
    return torch.tensor(
        [[fx / orig_w, 0.0, cx / orig_w],
         [0.0, fy / orig_h, cy / orig_h],
         [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


def _resize_rgb_to_tensor(img_path: Path, render_resolution: int, device: torch.device) -> torch.Tensor:
    img = Image.open(img_path).convert("RGBA")
    if img.size != (render_resolution, render_resolution):
        img = img.resize((render_resolution, render_resolution), Image.Resampling.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).to(device=device, dtype=torch.float32).permute(2, 0, 1).contiguous()


def ensure_nchw(x):
    """Normalize a renderer buffer of any of the usual shapes to (B, C, H, W)."""
    if x is None:
        return None
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if x.dim() == 4:
        _, a, _, c = x.shape
        if c <= 4 and a > c:
            return x.permute(0, 3, 1, 2)
        return x
    if x.dim() == 3:
        a, b, c = x.shape
        if c <= 4 and a > c:
            return x.permute(2, 0, 1).unsqueeze(0)
        if a <= 4 and b > 16 and c > 16:
            return x.unsqueeze(0)
        return x.unsqueeze(1)
    if x.dim() == 2:
        return x.unsqueeze(0).unsqueeze(0)
    return x


def _to_nchw_3ch(x):
    if x is None:
        return None
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if x.dim() == 4:
        if x.shape[1] == 3:
            return x
        if x.shape[-1] == 3:
            return x.permute(0, 3, 1, 2)
    elif x.dim() == 3:
        if x.shape[0] == 3:
            return x.unsqueeze(0)
        if x.shape[-1] == 3:
            return x.permute(2, 0, 1).unsqueeze(0)
    raise ValueError(f"Expected a 3-channel image/normal tensor, got shape={tuple(x.shape)}")


def _perceptual_loss(gt, pred, name: str, lambda_ssim: float = 0.2, lambda_lpips: float = 0.2):
    gt, pred = _to_nchw_3ch(gt), _to_nchw_3ch(pred)
    terms = {
        f"{name}_loss": l1_loss(gt, pred),
        f"{name}_loss_ssim": 1.0 - ssim(gt, pred),
        f"{name}_loss_lpips": lpips(gt, pred),
    }
    terms[f"{name}_loss_perceptual"] = (
        terms[f"{name}_loss"]
        + terms[f"{name}_loss_ssim"] * lambda_ssim
        + terms[f"{name}_loss_lpips"] * lambda_lpips
    )
    return terms


def _tsdf_reg_loss_from_depth(rep, depth_map, extrinsics, intrinsics) -> torch.Tensor:
    with torch.no_grad():
        projected_pts, pts_depth = utils3d.torch.project_cv(
            extrinsics=extrinsics, intrinsics=intrinsics, points=rep.tsdf_v,
        )
        projected_pts = (projected_pts - 0.5) * 2.0
        depth_map_res = depth_map.shape[-1]
        gt_depth = F.grid_sample(
            depth_map.reshape(1, 1, depth_map_res, depth_map_res),
            projected_pts.reshape(1, 1, -1, 2),
            mode='bilinear', padding_mode='border', align_corners=True,
        )
        pseudo_sdf = gt_depth.flatten() - pts_depth.flatten()
        delta = 1.0 / rep.res * 3.0
        trunc_mask = pseudo_sdf > -delta

    gt_tsdf = torch.clamp(pseudo_sdf[trunc_mask], -delta, delta)
    tsdf = rep.tsdf_s.flatten()[trunc_mask]
    return torch.mean((tsdf - gt_tsdf) ** 2)


def tsdf_reg_loss(gt_mesh, rep, extrinsics, intrinsics, renderer: MeshRenderer) -> torch.Tensor:
    """Regress the current TSDF samples to the pseudo-TSDF read off the GT depth (trainer recipe)."""
    if getattr(rep, "tsdf_v", None) is None or getattr(rep, "tsdf_s", None) is None:
        raise ValueError("Current mesh must expose tsdf_v and tsdf_s to compute the TSDF loss.")
    depth_map = renderer.render(
        mesh=gt_mesh, extrinsics=extrinsics, intrinsics=intrinsics, return_types=["depth"],
    )["depth"]
    return _tsdf_reg_loss_from_depth(rep, depth_map, extrinsics, intrinsics)


def compute_slat_mesh_decoder_losses(
    gt_rgb, gt_mask, gt_normal, gt_depth,
    pred_rgb, pred_mask, pred_normal, pred_depth,
    *,
    depth_loss_type: str = 'l1',
    lambda_depth: float = 1.0,
    lambda_ssim: float = 0.2,
    lambda_lpips: float = 0.2,
    lambda_color: float = 0.1,
    use_color: bool = True,
):
    """mask + depth + normal (+ color) terms, exactly as in hypothesis_cubefeats.py."""
    device = gt_rgb.device
    terms = {"loss": torch.tensor(0.0, device=device, dtype=gt_rgb.dtype)}

    if gt_mask is not None and pred_mask is not None:
        terms["mask_loss"] = l1_loss(pred_mask, gt_mask)
        terms["loss"] = terms["loss"] + terms["mask_loss"]

    if gt_depth is not None and pred_depth is not None:
        if depth_loss_type == 'l1':
            terms["depth_loss"] = l1_loss(pred_depth, gt_depth)
        elif depth_loss_type == 'smooth_l1':
            terms["depth_loss"] = smooth_l1_loss(pred_depth, gt_depth, beta=1.0 / 128.0)
        else:
            raise ValueError(f"Unsupported depth loss type: {depth_loss_type}")
        terms["loss"] = terms["loss"] + terms["depth_loss"] * lambda_depth

    if gt_normal is not None and pred_normal is not None:
        normal_terms = _perceptual_loss(gt_normal, pred_normal, "normal", lambda_ssim, lambda_lpips)
        terms.update(normal_terms)
        terms["loss"] = terms["loss"] + normal_terms["normal_loss_perceptual"]

    if use_color and gt_rgb is not None and pred_rgb is not None:
        color_terms = _perceptual_loss(gt_rgb, pred_rgb, "color", lambda_ssim, lambda_lpips)
        terms.update(color_terms)
        terms["loss"] = terms["loss"] + color_terms["color_loss_perceptual"] * lambda_color

    return terms



def mesh_vertex_colors(mesh) -> np.ndarray:
    """Per-vertex RGB in [0, 1], whatever the mesh carries."""
    visual = getattr(mesh, "visual", None)
    colors = None
    if visual is not None:
        try:
            if hasattr(visual, "to_color"):
                visual = visual.to_color()
            colors = np.asarray(visual.vertex_colors)
        except Exception:
            colors = None
    if colors is None or len(colors) != len(mesh.vertices):
        return np.ones((len(mesh.vertices), 3), dtype=np.float64)
    return np.asarray(colors)[:, :3].astype(np.float64) / 255.0


def load_source_mesh(path: Path, normalize: bool = True) -> trimesh.Trimesh:
    """Load a recgen (.obj, vertex colors) or SAM3D (.glb, texture) mesh with per-vertex colors."""
    mesh = trimesh.load(str(path), force='mesh', process=False)
    colors = mesh_vertex_colors(mesh)

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if normalize:
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        scale = float((hi - lo).max())
        verts = (verts - (lo + hi) / 2.0) / max(scale, 1e-8) * (1.0 - 2.0 / DECODER_RES)

    out = trimesh.Trimesh(vertices=verts, faces=np.asarray(mesh.faces), process=False)
    out.visual = trimesh.visual.ColorVisuals(
        mesh=out, vertex_colors=(np.clip(colors, 0, 1) * 255).astype(np.uint8)
    )
    return out


def to_mesh_extract_result(tri_mesh: trimesh.Trimesh, device) -> MeshExtractResult:
    """trimesh -> the renderer's mesh container, with the vertex colors as vertex_attrs."""
    colors = mesh_vertex_colors(tri_mesh)
    return MeshExtractResult(
        vertices=torch.as_tensor(np.asarray(tri_mesh.vertices), dtype=torch.float32, device=device),
        faces=torch.as_tensor(np.asarray(tri_mesh.faces), dtype=torch.long, device=device),
        vertex_attrs=torch.as_tensor(colors, dtype=torch.float32, device=device),
    )


def voxelize_mesh(tri_mesh: trimesh.Trimesh, res: int, fill_interior: bool, device) -> torch.Tensor:
    """Surface voxelization on the res^3 grid -> SLAT coords [N, 4] (batch, x, y, z)."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        np.clip(np.asarray(tri_mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    )
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tri_mesh.faces))

    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh, voxel_size=1 / res, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5)
    )
    grid_idx = np.array([v.grid_index for v in voxel_grid.get_voxels()])
    if grid_idx.size == 0:
        raise RuntimeError("Voxelization produced no voxels -- is the mesh inside [-0.5, 0.5]^3?")

    if fill_interior:
        from scipy.ndimage import binary_fill_holes
        occ = np.zeros((res, res, res), dtype=bool)
        occ[grid_idx[:, 0], grid_idx[:, 1], grid_idx[:, 2]] = True
        grid_idx = np.argwhere(binary_fill_holes(occ))

    grid_idx = np.clip(grid_idx, 0, res - 1)
    order = np.lexsort((grid_idx[:, 2], grid_idx[:, 1], grid_idx[:, 0]))
    grid_idx = grid_idx[order]

    idx = torch.from_numpy(grid_idx.astype(np.int32)).to(device)
    return torch.cat([torch.zeros(idx.shape[0], 1, dtype=torch.int32, device=device), idx], dim=1)


@torch.no_grad()
def encode_slat(
    coords: torch.Tensor,
    encoder,
    dinov2_model,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    images: torch.Tensor,
    res: int = DECODER_RES,
    batch_size: int = 4,
) -> sp.SparseTensor:
    """DINOv2 patch tokens of `images`, bilinearly sampled at every voxel's projection and"""
    device = coords.device
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    n_patch = DINO_RES // 14
    positions = (coords[:, 1:].float() + 0.5) / res - 0.5

    feats = torch.zeros(coords.shape[0], 1024, device=device)
    n_views = extrinsics.shape[0]
    for s in range(0, n_views, batch_size):
        sl = slice(s, min(s + batch_size, n_views))
        imgs = F.interpolate(images[sl], size=(DINO_RES, DINO_RES), mode="bilinear", align_corners=False)
        imgs = normalize(imgs)
        tokens = dinov2_model(imgs, is_training=True)['x_prenorm'][:, dinov2_model.num_register_tokens + 1:]
        tokens = tokens.permute(0, 2, 1).reshape(imgs.shape[0], 1024, n_patch, n_patch)
        intr = intrinsics[None].expand(imgs.shape[0], -1, -1)
        uv = utils3d.torch.project_cv(positions, extrinsics[sl], intr)[0] * 2 - 1
        feats += F.grid_sample(
            tokens.float(), uv.unsqueeze(1), mode="bilinear", align_corners=False,
        ).squeeze(2).permute(0, 2, 1).sum(0)
    feats /= n_views

    return encoder(sp.SparseTensor(feats=feats.float(), coords=coords), sample_posterior=False)


def trellis_feature_cameras(num_views: int, seed: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """TRELLIS' own render rig, expressed in the mesh's normalized [-0.5, 0.5]^3 frame."""
    rng = np.random.default_rng(seed)
    offset = (float(rng.random()), float(rng.random()))
    yaws, pitchs = [], []
    for i in range(num_views):
        yaw, pitch = sphere_hammersley_sequence(i, num_views, offset)
        yaws.append(yaw)
        pitchs.append(pitch)
    extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaws, pitchs, TRELLIS_CAM_RADIUS, TRELLIS_CAM_FOV_DEG
    )
    return torch.stack(extrinsics).to(device=device, dtype=torch.float32), \
        intrinsics[0].to(device=device, dtype=torch.float32)


@torch.no_grad()
def render_feature_views(
    renderer: MeshRenderer,
    mesh: MeshExtractResult,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    """Render the source mesh (object on black) from every camera -> [V, 3, RES, RES]."""
    out = []
    for extr in extrinsics:
        buf = renderer.render(mesh=mesh, extrinsics=extr, intrinsics=intrinsics,
                              return_types=["color", "mask"])
        color = ensure_nchw(buf["color"])
        mask = ensure_nchw(buf["mask"])
        out.append((color * mask)[0].clamp(0, 1))
    return torch.stack(out, dim=0)





def compute_alignment(cfg: Dict, src_mesh, src_tm, gt_world, gt_tm, renderer, extrinsics_world, intr,
                      device) -> np.ndarray:
    """Similarity mesh frame -> world frame, as a float64 4x4."""
    path = cfg.get("alignment_path", "")
    if path and Path(path).exists():
        print(f"[align] loaded {path}")
        return np.array(np.load(path), dtype=np.float64)

    method = str(cfg.get("alignment", "search"))
    if method == "search":
        A = select_alignment(src_mesh, gt_world, gt_tm, renderer, extrinsics_world, intr,
                             int(cfg.get("align_iou_views", 16)), int(cfg.get("align_top_k", 4)), device)
    elif method == "identity":
        A = np.eye(4)
    elif method == "ransac":
        A = align_meshes_fixed_scale_ransac(src_tm, gt_tm)
    elif method == "similarity":
        A = align_meshes_similarity(src_tm, gt_tm)
        A = A[0] if isinstance(A, tuple) else A
    elif method == "bbox":
        A = align_meshes(dtc_mesh=gt_tm, pred_mesh=src_tm)
        A = A[0] if isinstance(A, tuple) else A
    else:
        raise ValueError(f"Unknown alignment method: {method}")
    return np.array(np.asarray(A), dtype=np.float64)


def to_mesh_frame(A: np.ndarray, gt_world: MeshExtractResult, extrinsics_world: torch.Tensor):
    """GT mesh and cameras in the mesh frame."""
    s, R, t = split_similarity(A)
    ortho_err = np.abs(R @ R.T - np.eye(3)).max()
    if ortho_err > 1e-3:
        print(f"[align] WARNING: the alignment is not a similarity (|RR^T - I| = {ortho_err:.2e}); "
              f"the cameras below will not be rigid")
    device = extrinsics_world.device
    R_t = torch.tensor(R, dtype=torch.float32, device=device)
    t_t = torch.tensor(t, dtype=torch.float32, device=device)
    A_t = torch.tensor(A, dtype=torch.float32, device=device)
    gt_mesh = MeshExtractResult(
        vertices=((gt_world.vertices - t_t) @ R_t) / s,
        faces=gt_world.faces,
        vertex_attrs=gt_world.vertex_attrs,
    )
    extrinsics = extrinsics_world @ A_t
    extrinsics[:, :3, :] = extrinsics[:, :3, :] / s
    return gt_mesh, extrinsics



@torch.no_grad()
def visual_panel(renderer, mesh, gt_mesh, extr, intr) -> np.ndarray:
    """pred color | gt color | pred normal | gt normal | overlay (red = pred only, green = gt only)."""
    pred = renderer.render(mesh=mesh, extrinsics=extr, intrinsics=intr, return_types=["color", "mask", "normal"])
    gt = renderer.render(mesh=gt_mesh, extrinsics=extr, intrinsics=intr, return_types=["color", "mask", "normal"])
    overlay = torch.stack([pred["mask"], gt["mask"], torch.zeros_like(gt["mask"])], dim=0)
    vis = torch.cat([
        ensure_nchw(pred["color"] * pred["mask"]), ensure_nchw(gt["color"] * gt["mask"]),
        ensure_nchw(pred["normal"]), ensure_nchw(gt["normal"]), ensure_nchw(overlay),
    ], dim=3)
    return vis[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()


@torch.no_grad()
def eval_views(renderer, mesh, gt_mesh, extrinsics, intr, view_ids, out_fmt: Optional[str]):
    """PSNR (masked color) and silhouette IoU; the extrinsics are mesh-frame cameras."""
    psnrs, ious = [], []
    for vid in view_ids:
        gt = renderer.render(mesh=gt_mesh, extrinsics=extrinsics[vid], intrinsics=intr, return_types=["color", "mask"])
        pred = renderer.render(mesh=mesh, extrinsics=extrinsics[vid], intrinsics=intr, return_types=["color", "mask"])
        gt_mask, pred_mask = ensure_nchw(gt["mask"]), ensure_nchw(pred["mask"])
        gt_rgb = ensure_nchw(gt["color"]) * gt_mask
        pred_rgb = (ensure_nchw(pred["color"]) * pred_mask).clamp(0, 1)
        psnrs.append(float(-10 * torch.log10(F.mse_loss(pred_rgb, gt_rgb).clamp_min(1e-10))))
        p, g = pred_mask > 0.5, gt_mask > 0.5
        ious.append(float((p & g).sum() / (p | g).sum().clamp_min(1)))
        if out_fmt is not None:
            save_image(out_fmt.format(vid), visual_panel(renderer, mesh, gt_mesh, extrinsics[vid], intr))
    return {"psnr": float(np.mean(psnrs)), "mask_iou": float(np.mean(ious))}


def export_mesh(mesh: MeshExtractResult, path: Path, A: Optional[np.ndarray] = None):
    v = mesh.vertices.detach().cpu().numpy()
    if A is not None:
        v = v @ A[:3, :3].T + A[:3, 3]
    kwargs = {}
    if getattr(mesh, "vertex_attrs", None) is not None:
        kwargs["vertex_colors"] = (mesh.vertex_attrs[:, :3].detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(v, mesh.faces.detach().cpu().numpy(), process=False, **kwargs).export(str(path))
    print(f"[io] wrote {path}")



def build_setup(cfg: Dict, mesh_source: str, device):
    seed = int(cfg.get("seed", 1))
    data_dir = Path(cfg["data_dir"])
    assert data_dir.exists(), f"Data directory does not exist: {data_dir}"

    train_transforms = _load_json(data_dir / "transforms_train.json")
    test_transforms = _load_json(data_dir / "transforms_test.json")
    frames = train_transforms["frames"]
    object_pose = np.array(train_transforms["object_pose"], dtype=np.float64)
    image_paths = [_try_image_path(data_dir / f["file_path"]) for f in frames]
    intr = _scale_opencv_intrinsics(train_transforms["intrinsic"], Image.open(image_paths[0]).size).to(device)
    extrinsics_world = torch.stack(
        [torch.tensor(f["transform_matrix"], dtype=torch.float32) for f in frames]).to(device)
    test_extrinsics_world = torch.stack(
        [torch.tensor(f["transform_matrix"], dtype=torch.float32) for f in test_transforms["frames"]]).to(device)

    renderer = MeshRenderer(
        rendering_options={"resolution": RES, "near": float(cfg.get("near", 0.01)),
                           "far": float(cfg.get("far", 1000.0)), "ssaa": int(cfg.get("ssaa", 1))},
        device=device.type,
    )

    print("Loading TRELLIS pipeline...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained(str(cfg.get("model_name", "microsoft/TRELLIS-image-large")))
    pipeline.cuda()

    src_path = Path(cfg[MESH_KEYS[mesh_source]])
    assert src_path.exists(), (f"{mesh_source} mesh does not exist: {src_path} -- generate it with "
                               f"benchmark/run_all.py --methods {mesh_source}")
    print(f"[mesh] {mesh_source}: {src_path}")
    src_tm = load_source_mesh(src_path, normalize=bool(cfg.get("normalize_mesh", True)))
    src_mesh = to_mesh_extract_result(src_tm, device)
    print(f"[mesh] {len(src_tm.vertices)} verts, bounds {np.round(src_tm.bounds, 4).tolist()}")

    gt_tm = trimesh.load_mesh(str(cfg.get("gt_mesh", data_dir / "model.glb")), force='mesh')
    gt_tm.apply_transform(object_pose)
    gt_world = to_mesh_extract_result(gt_tm, device)
    gt_plain = trimesh.Trimesh(vertices=gt_tm.vertices, faces=gt_tm.faces, process=False)
    print(f"[gt] extent (world units) {np.round(gt_plain.extents, 4).tolist()}")

    A = compute_alignment(cfg, src_mesh, src_tm, gt_world, gt_plain, renderer, extrinsics_world, intr, device)
    print(f"[align] scale {split_similarity(A)[0]:.4f}\n{np.round(A, 4)}")
    gt_mesh, extrinsics = to_mesh_frame(A, gt_world, extrinsics_world)
    _, test_extrinsics = to_mesh_frame(A, gt_world, test_extrinsics_world)

    coords = voxelize_mesh(src_tm, DECODER_RES, bool(cfg.get("fill_interior", False)), device)
    print(f"[voxel] {coords.shape[0]} voxels on the {DECODER_RES}^3 grid")

    n_feat_views = int(cfg.get("feature_views", 150))
    feature_cameras = str(cfg.get("feature_cameras", "trellis"))
    feature_source = str(cfg.get("feature_source", "render"))
    if feature_cameras == "trellis":
        if feature_source != "render":
            raise ValueError("feature_cameras: trellis renders the source mesh itself; the GT photos "
                             "belong to the DTC cameras, so use feature_cameras: dataset.")
        extr_feat, intr_feat = trellis_feature_cameras(n_feat_views, seed, device)
        feat_renderer = MeshRenderer(
            rendering_options={"resolution": RES, "near": 0.8, "far": 4.0, "ssaa": 1}, device=device.type)
        feat_images = render_feature_views(feat_renderer, src_mesh, extr_feat, intr_feat)
        del feat_renderer
    elif feature_cameras == "dataset":
        view_ids = np.unique(np.linspace(0, len(frames) - 1, min(n_feat_views, len(frames))).round().astype(int))
        extr_feat, intr_feat = extrinsics[view_ids], intr
        if feature_source == "render":
            feat_images = render_feature_views(renderer, src_mesh, extr_feat, intr_feat)
        elif feature_source == "dataset":
            rgba = torch.stack([_resize_rgb_to_tensor(image_paths[i], RES, device) for i in view_ids])
            feat_images = rgba[:, :3] * (rgba[:, 3:] if rgba.shape[1] == 4 else 1.0)
        else:
            raise ValueError(f"Unknown feature_source: {feature_source}")
    else:
        raise ValueError(f"Unknown feature_cameras: {feature_cameras}")
    print(f"[feat] {feat_images.shape[0]} views (cameras={feature_cameras}, source={feature_source})")

    encoder = models.from_pretrained(str(cfg.get("encoder_ckpt", SLAT_ENCODER_CKPT))).eval().cuda()
    slat = encode_slat(coords, encoder, pipeline.models['image_cond_model'],
                       extrinsics=extr_feat, intrinsics=intr_feat, images=feat_images,
                       res=DECODER_RES, batch_size=int(cfg.get("dino_batch", 8)))
    print(f"[slat] feats {tuple(slat.feats.shape)} (mean {slat.feats.mean():.3f}, std {slat.feats.std():.3f})")
    feat_sample = feat_images[:4].detach().cpu()
    del encoder, feat_images
    torch.cuda.empty_cache()

    return {
        "pipeline": pipeline, "renderer": renderer, "intr": intr,
        "extrinsics": extrinsics, "test_extrinsics": test_extrinsics,
        "alignment": A, "gt_mesh": gt_mesh, "gt_plain": gt_plain, "src_mesh": src_mesh,
        "slat": slat, "feat_sample": feat_sample,
    }



def run(config_path: Path, mesh_source: Optional[str], out_dir_override: Optional[str], check: bool):
    cfg = yaml.safe_load(open(config_path, "r"))
    mesh_source = mesh_source or str(cfg.get("mesh_source", "recgen"))
    target = str(cfg.get("optimize", "slat"))
    if target not in ("slat", "cubefeats"):
        raise ValueError(f"optimize must be slat or cubefeats, got {target}")

    seed = int(cfg.get("seed", 1))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda")

    out_dir = Path(out_dir_override or cfg["out_dir"].format(mesh_source=mesh_source, optimize=target))
    if check:
        out_dir = out_dir / "check"
    (out_dir / "test").mkdir(parents=True, exist_ok=True)
    print(f"[out] {out_dir}  (optimize={target}, mesh_source={mesh_source})")

    setup = build_setup(cfg, mesh_source, device)
    renderer, intr = setup["renderer"], setup["intr"]
    extrinsics, test_extrinsics = setup["extrinsics"], setup["test_extrinsics"]
    A, gt_mesh, slat = setup["alignment"], setup["gt_mesh"], setup["slat"]
    np.save(out_dir / "alignment.npy", A)

    decoder = setup["pipeline"].models['slat_decoder_mesh']
    decoder.requires_grad_(False)
    decoder.train()

    if target == "slat":
        normalization = setup["pipeline"].slat_normalization
        slat_mean = torch.tensor(normalization['mean'], device=device)[None]
        slat_std = torch.tensor(normalization['std'], device=device)[None]
        param_init = ((slat.feats - slat_mean) / slat_std).detach()

        def decode(p):
            return decoder(sp.SparseTensor(feats=p * slat_std + slat_mean, coords=slat.coords))[0]

        def to_save(p):
            return {"coords": slat.coords.cpu(), "feats": (p * slat_std + slat_mean).detach().cpu(), "alignment": A}
    else:
        with torch.no_grad():
            cube_feats = decoder(slat, return_hidden=True)
        chan_scale = torch.ones(cube_feats.feats.shape[1], device=device, dtype=cube_feats.feats.dtype)
        if bool(cfg.get("cubefeats_sdf_voxel_units", True)):
            lo, hi = decoder.mesh_extractor.layouts['sdf']['range']
            chan_scale[lo:hi] = 1.0 / decoder.mesh_extractor.res
        param_init = (cube_feats.feats / chan_scale).detach().clone()
        print(f"[cubefeats] {tuple(cube_feats.feats.shape)} @ {decoder.mesh_extractor.res}^3, "
              f"sdf std {cube_feats.feats[:, :8].std():.3e} (world units)")

        def decode(p):
            return decoder.mesh_extractor(cube_feats.replace(feats=p * chan_scale), training=True)

        def to_save(p):
            return {"coords": cube_feats.coords.cpu(), "feats": (p * chan_scale).detach().cpu(), "alignment": A}
    param = nn.Parameter(param_init.clone())
    print(f"[opt] {param.numel()} values ({tuple(param.shape)})")

    eval_ids = np.unique(np.linspace(0, test_extrinsics.shape[0] - 1,
                                     min(int(cfg.get("eval_views", 8)), test_extrinsics.shape[0])).round().astype(int)).tolist()

    with torch.no_grad():
        init_mesh = decode(param)
    export_mesh(init_mesh, out_dir / "init_mesh" / f"{target}_init_world.glb", A)
    export_mesh(setup["src_mesh"], out_dir / "init_mesh" / "source_mesh_world.glb", A)
    cd_init = chamfer_to_gt(init_mesh, A, setup["gt_plain"])
    cd_src = chamfer_to_gt(setup["src_mesh"], A, setup["gt_plain"])
    init_metrics = eval_views(renderer, init_mesh, gt_mesh, test_extrinsics, intr, eval_ids,
                              str(out_dir / "init_mesh" / "init_{:03d}.png"))
    print(f"[init] {init_metrics} | chamfer: source mesh {cd_src * 1000:.2f}mm, "
          f"after encode/decode {cd_init * 1000:.2f}mm")
    with open(out_dir / "metrics.jsonl", "w") as f:
        f.write(json.dumps({"iter": -1, **init_metrics, "chamfer_mm": cd_init * 1000,
                            "source_chamfer_mm": cd_src * 1000}) + "\n")

    if check:
        strip = torch.cat(list(setup["feat_sample"]), dim=2).permute(1, 2, 0).clamp(0, 1)
        save_image(str(out_dir / "feature_views.png"), strip.numpy())
        train_ids = np.unique(np.linspace(0, extrinsics.shape[0] - 1, 6).round().astype(int)).tolist()
        print(f"[check] train views {eval_views(renderer, init_mesh, gt_mesh, extrinsics, intr, train_ids, str(out_dir / 'train_{:03d}.png'))}")
        print("[check] done -- inspect the renders before launching the run.")
        return

    lr = float(cfg["lr"])
    num_iters = int(cfg.get("num_iters", 500))
    frames_per_iter = int(cfg.get("frames_per_iter", 10))
    display_interval = int(cfg.get("display_interval", 25))
    eval_interval = int(cfg.get("eval_interval", 50))
    weight = float(cfg.get("weight", 1.5))
    sdf_weight = float(cfg.get("sdf_weight", 0.5))
    tsdf_weight = float(cfg.get("tsdf_weight", 0.5))
    latent_l2_weight = float(cfg.get("latent_l2_weight", 0.0))
    lambda_mask = float(cfg.get("lambda_mask", 1.0))
    lambda_depth = float(cfg.get("lambda_depth", 1.2))
    lambda_normal = float(cfg.get("lambda_normal", 1.0))
    lambda_color = float(cfg.get("lambda_color", 0.9))
    lambda_ssim = float(cfg.get("lambda_ssim", 0.2))
    lambda_lpips = float(cfg.get("lambda_lpips", 0.2))
    grad_clip = float(cfg.get("grad_clip", 0.0))

    optimizer = (torch.optim.SGD([param], lr=lr) if str(cfg.get("optimizer", "adam")).lower() == "sgd"
                 else torch.optim.Adam([param], lr=lr))
    scaler = torch.amp.GradScaler('cuda', enabled=bool(cfg.get("loss_scaling", True)))

    N = extrinsics.shape[0]
    rng = np.random.default_rng(seed=seed)
    gt_cache: Dict[int, dict] = {}

    @torch.no_grad()
    def gt_buffers(idx: int) -> dict:
        if idx not in gt_cache:
            gt_cache[idx] = renderer.render(mesh=gt_mesh, extrinsics=extrinsics[idx], intrinsics=intr,
                                            return_types=["color", "mask", "normal", "depth"])
        return gt_cache[idx]

    for i in range(num_iters + 1):
        indices = np.arange(N) if frames_per_iter >= N else rng.choice(N, size=frames_per_iter, replace=False)
        mesh = decode(param)
        if not mesh.success:
            print(f"[Iter {i:04d}] empty mesh, skipping", flush=True)
            continue

        comps = {k: torch.zeros((), device=device) for k in ["mask", "depth", "normal", "color", "tsdf"]}
        ious = []
        for idx in indices:
            idx = int(idx)
            gt = gt_buffers(idx)
            m = gt["mask"]
            pred = renderer.render(mesh=mesh, extrinsics=extrinsics[idx], intrinsics=intr,
                                   return_types=["color", "mask", "normal", "depth"])
            terms = compute_slat_mesh_decoder_losses(
                gt_rgb=gt["color"] * m, gt_mask=m, gt_normal=gt["normal"] * m, gt_depth=gt["depth"] * m,
                pred_rgb=pred["color"] * m, pred_mask=pred["mask"], pred_normal=pred["normal"] * m,
                pred_depth=pred["depth"] * m,
                lambda_depth=lambda_depth, lambda_ssim=lambda_ssim, lambda_lpips=lambda_lpips,
                lambda_color=lambda_color, use_color=True,
            )
            comps["mask"] = comps["mask"] + weight * lambda_mask * terms["mask_loss"]
            comps["depth"] = comps["depth"] + weight * lambda_depth * terms["depth_loss"]
            comps["normal"] = comps["normal"] + weight * lambda_normal * terms["normal_loss_perceptual"]
            comps["color"] = comps["color"] + weight * lambda_color * terms["color_loss_perceptual"]
            comps["tsdf"] = comps["tsdf"] + tsdf_weight * _tsdf_reg_loss_from_depth(mesh, gt["depth"], extrinsics[idx], intr)
            with torch.no_grad():
                pm, gm = pred["mask"] > 0.5, m > 0.5
                ious.append(float((pm & gm).sum() / (pm | gm).sum().clamp_min(1)))
        comps = {k: v / len(indices) for k, v in comps.items()}
        comps["reg"] = sdf_weight * mesh.reg_loss
        comps["latent_l2"] = latent_l2_weight * ((param - param_init) ** 2).mean()
        total = sum(comps.values())

        if i == 0:
            print("[diag] grad of each term w.r.t. the optimized variable (loss scale 1024):")
            for k, v in comps.items():
                gn, nz = grad_wrt(v, param, 1024.0)
                print(f"[diag]   {k:>9s}: value {v.item():.4e}, grad norm {gn:.3e}, non-zero {nz * 100:.1f}%")

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        grad_norm = param.grad.norm().item()
        nonzero = (param.grad != 0).float().mean().item()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([param], max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            drift = (param - param_init).abs().mean().item()
        print(f"[Iter {i:04d}] total {total.item():.4f} | "
              + " ".join(f"{k} {v.item():.4f}" for k, v in comps.items())
              + f" | IoU {np.mean(ious):.4f} | grad {grad_norm:.3e} ({nonzero * 100:.0f}% nz)"
              + f" | scale {scaler.get_scale():.0f} | drift {drift:.4f}", flush=True)

        if display_interval and i % display_interval == 0:
            vid = int(indices[0])
            save_image(str(out_dir / f"opt_{i:04d}_{vid:03d}.png"),
                       visual_panel(renderer, mesh, gt_mesh, extrinsics[vid], intr))

        if eval_interval and (i % eval_interval == 0 or i == num_iters):
            with torch.no_grad():
                eval_mesh = decode(param)
            metrics = eval_views(renderer, eval_mesh, gt_mesh, test_extrinsics, intr, eval_ids,
                                 str(out_dir / "test" / f"test_{i:04d}_{{:03d}}.png"))
            print(f"[eval {i:04d}] {metrics}", flush=True)
            with open(out_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps({"iter": i, "loss": total.item(), **metrics}) + "\n")

    with torch.no_grad():
        final_mesh = decode(param)
    export_mesh(final_mesh, out_dir / "final_mesh" / f"{target}_opt.glb")
    export_mesh(final_mesh, out_dir / "final_mesh" / f"{target}_opt_world.glb", A)
    torch.save(to_save(param), out_dir / "final_mesh" / f"{target}.pt")
    cd_final = chamfer_to_gt(final_mesh, A, setup["gt_plain"])
    final_metrics = eval_views(renderer, final_mesh, gt_mesh, test_extrinsics, intr, eval_ids, None)
    with open(out_dir / "metrics.jsonl", "a") as f:
        f.write(json.dumps({"iter": "final", **final_metrics, "chamfer_mm": cd_final * 1000}) + "\n")
    print(f"[final] {final_metrics} | chamfer {cd_init * 1000:.2f}mm -> {cd_final * 1000:.2f}mm "
          f"(source mesh {cd_src * 1000:.2f}mm)")
    print("Optimization completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize the TRELLIS SLAT encoded from a recgen / SAM3D / TRELLIS mesh")
    parser.add_argument("--yaml-config", type=Path, required=True)
    parser.add_argument("--mesh-source", choices=list(MESH_KEYS), default=None,
                        help="overrides mesh_source in the config")
    parser.add_argument("--out-dir", default=None, help="overrides out_dir in the config")
    parser.add_argument("--check", action="store_true",
                        help="stop after the initial mesh (alignment / encoding sanity check)")
    args = parser.parse_args()
    run(args.yaml_config, args.mesh_source, args.out_dir, args.check)
