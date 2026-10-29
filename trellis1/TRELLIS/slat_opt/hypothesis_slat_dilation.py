"""Can SLAT refinement grow a structure that the initial latent does not contain?"""

import os

os.environ.setdefault('SPCONV_ALGO', 'native')

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import shutil
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DECODER_RES = 64
# Bumped when the method changes, so arms finished by older code are redone instead of skipped.
VERSION = 2
DEFAULT_CONFIG = PROJECT_ROOT / "slat_opt_configs" / "dilation" / "Birdhouse_chimney.yaml"


# Dilation of the active voxel set (pure torch, no TRELLIS import: unit-testable on a CPU)

def occupancy_grid(coords: torch.Tensor, res: int) -> torch.Tensor:
    """coords [N, 3 or 4] (the last three columns are x, y, z) -> dense bool [res, res, res]."""
    xyz = coords[:, -3:].long()
    if xyz.numel() and (xyz.min() < 0 or xyz.max() > res - 1):
        raise ValueError(f"coords outside the {res}^3 grid: min {int(xyz.min())}, max {int(xyz.max())}")
    occ = torch.zeros((res, res, res), dtype=torch.bool, device=coords.device)
    occ[xyz[:, 0], xyz[:, 1], xyz[:, 2]] = True
    return occ


def dilate_grid(occ: torch.Tensor, radius: int, connectivity: int = 26) -> torch.Tensor:
    """Morphological dilation of a dense bool grid; the grid border is the only limit."""
    if radius <= 0:
        return occ.clone()
    x = occ[None, None].float()
    if connectivity == 26:
        x = F.max_pool3d(x, kernel_size=2 * radius + 1, stride=1, padding=radius)
    elif connectivity == 6:
        for _ in range(radius):
            p = F.pad(x, (1, 1, 1, 1, 1, 1))
            x = torch.stack([x, p[..., 2:, 1:-1, 1:-1], p[..., :-2, 1:-1, 1:-1], p[..., 1:-1, 2:, 1:-1],
                             p[..., 1:-1, :-2, 1:-1], p[..., 1:-1, 1:-1, 2:], p[..., 1:-1, 1:-1, :-2]]).amax(0)
    else:
        raise ValueError(f"connectivity must be 6 or 26, got {connectivity}")
    return x[0, 0] > 0.5


def dilation_candidates(
    active: torch.Tensor,
    res: int,
    radius: int = 1,
    connectivity: int = 26,
    seed: Optional[torch.Tensor] = None,
    restrict: Optional[torch.Tensor] = None,
    keep_prob: float = 1.0,
    max_new: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Voxels to activate: within `radius` of `seed` (default: the active set) and not active."""
    occ = occupancy_grid(active, res)
    cand = dilate_grid(occ if seed is None else seed, radius, connectivity) & ~occ
    if restrict is not None:
        cand &= restrict
    new = cand.nonzero().to(torch.int32)
    if keep_prob < 1.0 and new.shape[0]:
        keep = torch.rand(new.shape[0], generator=generator, device=new.device) < keep_prob
        new = new[keep]
    if max_new is not None and new.shape[0] > max_new:
        new = new[torch.randperm(new.shape[0], generator=generator, device=new.device)[:max_new].sort().values]
    return new


def init_new_feats(
    new_xyz: torch.Tensor,
    old_xyz: torch.Tensor,
    old_feats: torch.Tensor,
    mode: str = "nearest",
    noise_std: float = 0.0,
    generator: Optional[torch.Generator] = None,
    chunk: int = 4096,
) -> torch.Tensor:
    """Latent values for newly activated voxels, in the normalized units the SLAT is optimized in."""
    n, c = new_xyz.shape[0], old_feats.shape[1]
    if mode == "zero" or old_xyz.shape[0] == 0:
        feats = torch.zeros((n, c), dtype=old_feats.dtype, device=old_feats.device)
    elif mode == "nearest":
        idx = torch.empty(n, dtype=torch.long, device=old_feats.device)
        old_f = old_xyz[:, -3:].float()
        for s in range(0, n, chunk):
            idx[s:s + chunk] = torch.cdist(new_xyz[s:s + chunk, -3:].float(), old_f).argmin(dim=1)
        feats = old_feats[idx].clone()
    else:
        raise ValueError(f"init must be zero or nearest, got {mode}")
    if noise_std > 0:
        feats = feats + noise_std * torch.randn(feats.shape, generator=generator, device=feats.device, dtype=feats.dtype)
    return feats


def dilate_sparse_tensor(x, res: int, radius: int = 1, connectivity: int = 26, keep_prob: float = 1.0,
                         restrict: Optional[torch.Tensor] = None, max_new: Optional[int] = None,
                         init: str = "nearest", noise_std: float = 0.0,
                         generator: Optional[torch.Generator] = None):
    """Dilate the active set of a SparseTensor; returns (dilated tensor, is_new mask), always both."""
    from trellis.modules import sparse as sp

    new_xyz = dilation_candidates(x.coords, res, radius, connectivity, restrict=restrict, keep_prob=keep_prob,
                                  max_new=max_new, generator=generator)
    is_new = torch.zeros(x.coords.shape[0] + new_xyz.shape[0], dtype=torch.bool, device=x.coords.device)
    if new_xyz.shape[0] == 0:
        return x, is_new
    is_new[x.coords.shape[0]:] = True
    new_coords = torch.cat([torch.zeros_like(new_xyz[:, :1]), new_xyz], dim=1).to(x.coords.dtype)
    new_feats = init_new_feats(new_xyz, x.coords, x.feats, init, noise_std, generator)
    return sp.SparseTensor(feats=torch.cat([x.feats, new_feats]), coords=torch.cat([x.coords, new_coords])), is_new


# Geometry helpers

def voxel_centers(xyz: torch.Tensor, res: int = DECODER_RES) -> torch.Tensor:
    return (xyz[:, -3:].float() + 0.5) / res - 0.5


def surface_grid(vertices: torch.Tensor, res: int = DECODER_RES) -> torch.Tensor:
    """Dense bool grid of the latent voxels that hold a vertex of the decoded mesh."""
    grid = torch.zeros((res, res, res), dtype=torch.bool, device=vertices.device)
    if vertices.shape[0]:
        idx = ((vertices.detach() + 0.5) * res).floor().long().clamp(0, res - 1)
        grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return grid


class Region:
    """The structure to grow: a box (and optional half-space) in the object frame of the GT mesh."""

    def __init__(self, spec: Dict, object_pose: np.ndarray, A: np.ndarray, device):
        self.lo = torch.tensor(spec["box_lo"], dtype=torch.float32, device=device)
        self.hi = torch.tensor(spec["box_hi"], dtype=torch.float32, device=device)
        plane = spec.get("plane")
        self.plane = None if plane is None else torch.tensor(plane, dtype=torch.float32, device=device)
        # mesh frame -> object frame: object_pose^-1 @ A
        self.M = torch.tensor(np.linalg.inv(object_pose) @ A, dtype=torch.float32, device=device)

    def contains_object(self, p_obj: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
        inside = ((p_obj > self.lo - margin) & (p_obj < self.hi + margin)).all(dim=1)
        if self.plane is not None:
            inside &= (p_obj @ self.plane[:3]) > self.plane[3] - margin
        return inside

    def contains_mesh_frame(self, p: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
        return self.contains_object(p @ self.M[:3, :3].T + self.M[:3, 3], margin)


@torch.no_grad()
def pseudo_sdf(points: torch.Tensor, buffers: Dict, extr: torch.Tensor, intr: torch.Tensor):
    """Signed distance along the ray between `points` and a rendered depth map"""
    import utils3d.torch

    uv, z = utils3d.torch.project_cv(extrinsics=extr, intrinsics=intr, points=points)
    uv, z = uv.reshape(-1, 2), z.flatten()
    depth, mask = buffers["depth"], buffers["mask"]
    res = depth.shape[-1]
    grid = ((uv - 0.5) * 2).reshape(1, 1, -1, 2)
    sample = lambda img: F.grid_sample(img.reshape(1, 1, res, res), grid, mode='nearest', padding_mode='zeros',
                                       align_corners=False).flatten()
    d, m = sample(depth), sample(mask)
    out = torch.where(m > 0.99, d - z, torch.full_like(z, float('inf')))
    unknown = ((uv < 0) | (uv > 1)).any(dim=1) | (z <= 0) | ((m >= 0.01) & (m <= 0.99))
    return torch.where(unknown, torch.full_like(z, float('nan')), out)


@torch.no_grad()
def observed_surface_grid(views: List[Dict], extrinsics: torch.Tensor, intr: torch.Tensor,
                          res: int = DECODER_RES, band_voxels: float = 1.0, free_votes: int = 2) -> torch.Tensor:
    """Voxels the rendered views (dicts with "depth" and "mask") say hold a surface"""
    device = extrinsics.device
    ax = torch.arange(res, device=device)
    xyz = torch.stack(torch.meshgrid(ax, ax, ax, indexing="ij"), dim=-1).reshape(-1, 3)
    centers = voxel_centers(xyz, res)
    band = band_voxels / res
    on = torch.zeros(centers.shape[0], dtype=torch.bool, device=device)
    free = torch.zeros(centers.shape[0], dtype=torch.int32, device=device)
    for buffers, extr in zip(views, extrinsics):
        s = pseudo_sdf(centers, buffers, extr, intr)
        valid = ~torch.isnan(s)
        on |= valid & (s.abs() <= band)
        free += (valid & (s > band)).int()
    return (on & (free < free_votes)).reshape(res, res, res)


# The experiment

def run(config_path: Path, arms: Optional[List[str]], check: bool, force: bool, out_dir_override: Optional[str]):
    import trimesh
    import yaml
    from PIL import Image
    from scipy.spatial import cKDTree

    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.renderers.mesh_renderer import MeshRenderer
    from trellis.representations.mesh.cube2mesh import MeshExtractResult
    from trellis.modules import sparse as sp
    from opt_utils.utility import save_image
    from slat_optimization import select_alignment, split_similarity, chamfer_to_gt
    from slat_opt.hypothesis_mesh_slat import (
        RES, _load_json, _try_image_path, _scale_opencv_intrinsics, compute_slat_mesh_decoder_losses,
        _tsdf_reg_loss_from_depth, to_mesh_extract_result, to_mesh_frame, export_mesh, eval_views, visual_panel,
    )

    cfg = yaml.safe_load(open(config_path, "r"))
    seed = int(cfg.get("seed", 1))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda")

    out_root = Path(out_dir_override or cfg["out_dir"])
    if check:
        out_root = out_root / "check"
    prompt_dir = out_root / "prompt"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, out_root / "config.yaml")
    print(f"[out] {out_root}")

    # ---- data ------------------------------------------------------------------------------
    data_dir = Path(cfg["data_dir"])
    train_tf, test_tf = _load_json(data_dir / "transforms_train.json"), _load_json(data_dir / "transforms_test.json")
    frames = train_tf["frames"]
    object_pose = np.array(train_tf["object_pose"], dtype=np.float64)
    image_paths = [_try_image_path(data_dir / f["file_path"]) for f in frames]
    intr = _scale_opencv_intrinsics(train_tf["intrinsic"], Image.open(image_paths[0]).size).to(device)
    extrinsics_world = torch.stack([torch.tensor(f["transform_matrix"], dtype=torch.float32) for f in frames]).to(device)
    test_extrinsics_world = torch.stack(
        [torch.tensor(f["transform_matrix"], dtype=torch.float32) for f in test_tf["frames"]]).to(device)

    renderer = MeshRenderer(
        rendering_options={"resolution": RES, "near": float(cfg.get("near", 0.01)),
                           "far": float(cfg.get("far", 1000.0)), "ssaa": int(cfg.get("ssaa", 1))},
        device=device.type)

    gt_tm = trimesh.load_mesh(str(cfg.get("gt_mesh", data_dir / "model.glb")), force='mesh')
    gt_object_tm = gt_tm.copy()
    gt_tm.apply_transform(object_pose)
    gt_world = to_mesh_extract_result(gt_tm, device)
    gt_plain = trimesh.Trimesh(vertices=gt_tm.vertices, faces=gt_tm.faces, process=False)

    # ---- the prompt sample: a SLAT without the structure --------------------------------------
    print("Loading TRELLIS pipeline...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained(str(cfg.get("model_name", "microsoft/TRELLIS-image-large")))
    pipeline.cuda()
    decoder = pipeline.models['slat_decoder_mesh']
    decoder.requires_grad_(False)
    decoder.train()                      # training=True in the extractor: reg loss and TSDF samples
    slat_mean = torch.tensor(pipeline.slat_normalization['mean'], device=device)[None]
    slat_std = torch.tensor(pipeline.slat_normalization['std'], device=device)[None]

    def decode(param: torch.Tensor, coords: torch.Tensor):
        return decoder(sp.SparseTensor(feats=param * slat_std + slat_mean, coords=coords))[0]

    def decode_cubes(param: torch.Tensor, coords: torch.Tensor):
        """The decoder output before mesh extraction: 101 channels per cell of the 256^3 grid."""
        return decoder(sp.SparseTensor(feats=param * slat_std + slat_mean, coords=coords), return_hidden=True)

    extractor = decoder.mesh_extractor
    CUBE_RES, SDF_BIAS = extractor.res, extractor.sdf_bias
    SDF_LO, SDF_HI = extractor.layouts['sdf']['range']
    SUB = CUBE_RES // DECODER_RES                                    # cells per latent voxel, per axis

    prompt_name = str(cfg["prompt_image"])
    prompt_path = _try_image_path(data_dir / prompt_name)
    slat_cache = prompt_dir / "slat_prompt.pt"
    if slat_cache.exists() and not force:
        cached = torch.load(slat_cache, map_location=device)
        coords0, feats0 = cached["coords"].to(device), cached["feats"].to(device)
        print(f"[prompt] reusing {slat_cache} ({coords0.shape[0]} voxels)")
    else:
        sampler = cfg.get("sampler", {})
        print(f"[prompt] {prompt_path} (the structure is hidden in this view)")
        outputs = pipeline.run(
            Image.open(prompt_path), seed=seed, formats=["mesh"], return_latent=True,
            sparse_structure_sampler_params={"steps": int(sampler.get("structure_steps", 12)),
                                             "cfg_strength": float(sampler.get("structure_cfg", 7.5))},
            slat_sampler_params={"steps": int(sampler.get("slat_steps", 12)),
                                 "cfg_strength": float(sampler.get("slat_cfg", 5.0))},
        )
        slat = outputs["latent"]
        coords0, feats0 = slat.coords.detach(), slat.feats.detach().float()
        torch.save({"coords": coords0.cpu(), "feats": feats0.cpu()}, slat_cache)
        shutil.copy2(prompt_path, prompt_dir / f"prompt{prompt_path.suffix}")
        del outputs, slat
    param0 = ((feats0 - slat_mean) / slat_std).detach()
    print(f"[prompt] {coords0.shape[0]} active voxels, normalized feats mean {param0.mean():.3f} std {param0.std():.3f}")

    with torch.no_grad():
        prompt_mesh = decode(param0, coords0)
    assert prompt_mesh.success, "the prompt sample decodes to an empty mesh"

    # ---- mesh frame -> world -----------------------------------------------------------------
    align_path = prompt_dir / "alignment.npy"
    if align_path.exists() and not force:
        A = np.array(np.load(align_path), dtype=np.float64)
        print(f"[align] reusing {align_path}")
    else:
        A = np.array(np.asarray(select_alignment(
            prompt_mesh, gt_world, gt_plain, renderer, extrinsics_world, intr,
            int(cfg.get("align_iou_views", 16)), int(cfg.get("align_top_k", 4)), device)), dtype=np.float64)
        np.save(align_path, A)
    scale = split_similarity(A)[0]
    print(f"[align] scale {scale:.4f} (one latent voxel = {scale / DECODER_RES * 1000:.2f} mm)")
    gt_mesh, extrinsics = to_mesh_frame(A, gt_world, extrinsics_world)
    _, test_extrinsics = to_mesh_frame(A, gt_world, test_extrinsics_world)
    export_mesh(prompt_mesh, prompt_dir / "prompt_mesh_world.glb", A)

    # ---- the region to grow ------------------------------------------------------------------
    region = Region(cfg["region"], object_pose, A, device)
    samples = torch.tensor(np.asarray(gt_object_tm.sample(int(cfg["region"].get("gt_samples", 400000)))),
                           dtype=torch.float32, device=device)
    region_obj = samples[region.contains_object(samples)]
    assert region_obj.shape[0] > 100, "the region box holds almost no GT surface: check region in the config"
    Minv = torch.linalg.inv(region.M)
    region_pts = region_obj @ Minv[:3, :3].T + Minv[:3, 3]           # GT surface of the structure, mesh frame
    region_np = region_pts.cpu().numpy()
    lo, hi = region_pts.min(0).values, region_pts.max(0).values
    print(f"[region] {cfg['region'].get('name', 'region')}: {region_pts.shape[0]} GT samples, "
          f"{((hi - lo) * DECODER_RES).round().int().tolist()} latent voxels in size")

    N = extrinsics.shape[0]
    gt_cache: Dict[int, dict] = {}

    @torch.no_grad()
    def gt_buffers(idx: int) -> dict:
        if idx not in gt_cache:
            gt_cache[idx] = renderer.render(mesh=gt_mesh, extrinsics=extrinsics[idx], intrinsics=intr,
                                            return_types=["color", "mask", "normal", "depth"])
        return gt_cache[idx]

    # training views that see the structure best, for the growth panels
    vis = []
    for idx in range(N):
        s = pseudo_sdf(region_pts, gt_buffers(idx), extrinsics[idx], intr)
        vis.append(float(((s.abs() < 1.5 / DECODER_RES) & ~torch.isnan(s)).float().mean()))
    order = np.argsort(vis)[::-1]
    region_views: List[int] = []
    for idx in order:                                                # spread over the turntable
        if all(min(abs(int(idx) - v), N - abs(int(idx) - v)) >= N // 8 for v in region_views):
            region_views.append(int(idx))
        if len(region_views) == int(cfg.get("region_views", 3)):
            break
    hidden = [frames[i]["file_path"] for i in range(N) if vis[i] < 0.02]
    print(f"[region] panel views {region_views} (visible fraction {[round(vis[i], 2) for i in region_views]}); "
          f"hidden in {len(hidden)} of {N} training views"
          + ("" if prompt_name in hidden else f" -- WARNING: not hidden in the prompt view {prompt_name}"))

    eval_ids = np.unique(np.linspace(0, test_extrinsics.shape[0] - 1,
                                     min(int(cfg.get("eval_views", 8)), test_extrinsics.shape[0])).round().astype(int)).tolist()
    voxel_mm = scale / DECODER_RES * 1000

    def region_metrics(mesh) -> Dict[str, float]:
        """How much of the GT structure the mesh covers, and how good what it built there is."""
        v, f = mesh.vertices.detach().cpu().numpy(), mesh.faces.detach().cpu().numpy()
        if len(f) == 0:
            return {"region_recall_1vox": 0.0, "region_recall_2vox": 0.0, "region_dist_mm": float("nan"),
                    "region_pred_points": 0, "region_pred_dist_mm": float("nan")}
        pred = np.asarray(trimesh.Trimesh(v, f, process=False).sample(200000))
        d = cKDTree(pred).query(region_np)[0]
        inside = region.contains_mesh_frame(torch.tensor(pred, dtype=torch.float32, device=device),
                                            margin=float(cfg["region"].get("margin", 0.002))).cpu().numpy()
        back = cKDTree(region_np).query(pred[inside])[0] if inside.any() else np.array([np.nan])
        return {"region_recall_1vox": float((d < 1.0 / DECODER_RES).mean()),
                "region_recall_2vox": float((d < 2.0 / DECODER_RES).mean()),
                "region_dist_mm": float(d.mean() * scale * 1000),
                "region_pred_points": int(inside.sum()),
                "region_pred_dist_mm": float(np.nanmean(back) * scale * 1000)}

    def voxel_stats(coords, is_new, mesh) -> Dict[str, int]:
        surf = surface_grid(mesh.vertices)
        xyz = coords[:, 1:].long()
        holds = surf[xyz[:, 0], xyz[:, 1], xyz[:, 2]]
        in_region = region.contains_mesh_frame(voxel_centers(coords), margin=0.5 * scale / DECODER_RES)
        return {"voxels": int(coords.shape[0]), "new_voxels": int(is_new.sum()),
                "new_with_surface": int((is_new & holds).sum()),
                "region_voxels": int(in_region.sum()), "region_voxels_with_surface": int((in_region & holds).sum())}

    def evaluate(coords, param, is_new, panel_fmt: Optional[str], region_fmt: Optional[str]) -> Tuple[Dict, object]:
        with torch.no_grad():
            mesh = decode(param, coords)
        if not mesh.success:
            return {"empty_mesh": True}, mesh
        out = eval_views(renderer, mesh, gt_mesh, test_extrinsics, intr, eval_ids, panel_fmt)
        out["chamfer_mm"] = chamfer_to_gt(mesh, A, gt_plain) * 1000
        out.update(region_metrics(mesh))
        out.update(voxel_stats(coords, is_new, mesh))
        if region_fmt is not None:
            for vid in region_views:
                save_image(region_fmt.format(vid), visual_panel(renderer, mesh, gt_mesh, extrinsics[vid], intr))
        return out, mesh

    base_metrics, _ = evaluate(coords0, param0, torch.zeros(coords0.shape[0], dtype=torch.bool, device=device),
                               None, str(prompt_dir / "region_view_{:03d}.png"))
    print(f"[prompt] {json.dumps(base_metrics)}")
    (prompt_dir / "metrics.json").write_text(json.dumps(base_metrics, indent=2))
    if base_metrics.get("region_recall_2vox", 0) > float(cfg.get("max_initial_recall", 0.3)):
        print(f"[prompt] WARNING: {base_metrics['region_recall_2vox'] * 100:.0f}% of the structure is already "
              f"there; TRELLIS hallucinated it. Set excise_region: true to remove it, or try another seed.")
    if bool(cfg.get("excise_region", False)):
        keep = ~region.contains_mesh_frame(voxel_centers(coords0), margin=1.5 * scale / DECODER_RES)
        print(f"[prompt] excise_region: removing {int((~keep).sum())} active voxels in the region")
        coords0, param0 = coords0[keep], param0[keep]

    # ---- optimization settings ---------------------------------------------------------------
    lr = float(cfg["lr"])
    num_iters = 2 if check else int(cfg.get("num_iters", 500))
    frames_per_iter = int(cfg.get("frames_per_iter", 10))
    eval_interval = int(cfg.get("eval_interval", 50))
    grad_interval = int(cfg.get("grad_diag_interval", 100))
    weight, sdf_weight, tsdf_weight = float(cfg.get("weight", 1.5)), float(cfg.get("sdf_weight", 0.5)), float(cfg.get("tsdf_weight", 0.5))
    lam = {k: float(cfg.get(f"lambda_{k}", d)) for k, d in
           [("mask", 1.0), ("depth", 1.2), ("normal", 1.0), ("color", 0.9), ("ssim", 0.2), ("lpips", 0.2)]}
    max_active = int(cfg.get("max_active_voxels", 60000))
    # The TSDF term is a squared SDF error in world units
    tsdf_unit = (CUBE_RES / 3.0) ** 2 if bool(cfg.get("tsdf_normalize", True)) else 1.0
    silence_lr = float(cfg.get("silence_lr", lr))
    silence_margin = float(cfg.get("silence_margin_cells", 1.5)) / CUBE_RES

    def loss_terms(mesh, indices, buffers_of) -> Tuple[Dict[str, torch.Tensor], float]:
        comps = {k: torch.zeros((), device=device) for k in ["mask", "depth", "normal", "color", "tsdf"]}
        ious = []
        for idx in indices:
            idx = int(idx)
            gt = buffers_of(idx)
            m = gt["mask"]
            pred = renderer.render(mesh=mesh, extrinsics=extrinsics[idx], intrinsics=intr,
                                   return_types=["color", "mask", "normal", "depth"])
            terms = compute_slat_mesh_decoder_losses(
                gt_rgb=gt["color"] * m, gt_mask=m, gt_normal=gt["normal"] * m, gt_depth=gt["depth"] * m,
                pred_rgb=pred["color"] * m, pred_mask=pred["mask"], pred_normal=pred["normal"] * m,
                pred_depth=pred["depth"] * m, lambda_depth=lam["depth"], lambda_ssim=lam["ssim"],
                lambda_lpips=lam["lpips"], lambda_color=lam["color"], use_color=True)
            comps["mask"] = comps["mask"] + weight * lam["mask"] * terms["mask_loss"]
            comps["depth"] = comps["depth"] + weight * lam["depth"] * terms["depth_loss"]
            comps["normal"] = comps["normal"] + weight * lam["normal"] * terms["normal_loss_perceptual"]
            comps["color"] = comps["color"] + weight * lam["color"] * terms["color_loss_perceptual"]
            comps["tsdf"] = comps["tsdf"] + tsdf_weight * tsdf_unit * _tsdf_reg_loss_from_depth(mesh, gt["depth"], extrinsics[idx], intr)
            with torch.no_grad():
                pm, gm = pred["mask"] > 0.5, m > 0.5
                ious.append(float((pm & gm).sum() / (pm | gm).sum().clamp_min(1)))
        comps = {k: v / len(indices) for k, v in comps.items()}
        comps["reg"] = sdf_weight * mesh.reg_loss
        return comps, float(np.mean(ious))

    def grad_split(comps, param, coords, is_new) -> Dict[str, Dict[str, float]]:
        """Mean |grad| per voxel of every loss term, on old voxels, new voxels and new voxels in the region."""
        in_region = region.contains_mesh_frame(voxel_centers(coords), margin=0.5 * scale / DECODER_RES)
        groups = {"old": ~is_new, "new": is_new, "new_in_region": is_new & in_region}
        out = {}
        for name, value in comps.items():
            if not value.requires_grad:
                continue
            g, = torch.autograd.grad(value * 1024.0, param, retain_graph=True, allow_unused=True)
            g = torch.zeros_like(param) if g is None else torch.nan_to_num(g / 1024.0)
            per_voxel = g.abs().mean(dim=1)
            out[name] = {k: (float(per_voxel[m].mean()) if m.any() else 0.0) for k, m in groups.items()}
        return out

    def add_voxels(coords, param, is_new, new_xyz, init, noise_std, gen):
        new_coords = torch.cat([torch.zeros_like(new_xyz[:, :1]), new_xyz], dim=1).to(coords.dtype)
        new_feats = init_new_feats(new_xyz, coords, param.detach(), init, noise_std, gen)
        return (torch.cat([coords, new_coords]), torch.cat([param.detach(), new_feats]),
                torch.cat([is_new, torch.ones(new_xyz.shape[0], dtype=torch.bool, device=device)]))

    def silence(coords, param_t, fresh, iters: int, tag: str) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Make freshly activated voxels invisible"""
        stats = {"fresh_cells_inside": float("nan"), "other_cells_sdf_shift": float("nan"), "silence_iters": 0}
        if not fresh.any():
            return param_t, stats
        with torch.no_grad():
            ref = decode_cubes(param_t[~fresh], coords[~fresh])
        index = torch.full((CUBE_RES,) * 3, -1, dtype=torch.long, device=device)
        rc = ref.coords[:, 1:].long()
        index[rc[:, 0], rc[:, 1], rc[:, 2]] = torch.arange(rc.shape[0], device=device)
        ref_feats = ref.feats.float()
        fresh_grid = occupancy_grid(coords[fresh], DECODER_RES)

        p = nn.Parameter(param_t.clone())
        opt = torch.optim.Adam([p], lr=silence_lr)
        scaler = torch.amp.GradScaler('cuda', enabled=bool(cfg.get("loss_scaling", True)))
        for i in range(iters + 1):
            h = decode_cubes(p, coords)
            hc, feats = h.coords[:, 1:].long(), h.feats.float()
            is_fresh = fresh_grid[hc[:, 0] // SUB, hc[:, 1] // SUB, hc[:, 2] // SUB]
            sdf = feats[:, SDF_LO:SDF_HI] + SDF_BIAS
            j = index[hc[~is_fresh, 0], hc[~is_fresh, 1], hc[~is_fresh, 2]]
            assert (j >= 0).all(), "a cell of an existing voxel has no counterpart before the activation"
            diff = feats[~is_fresh] - ref_feats[j]
            sdf_shift = (diff[:, SDF_LO:SDF_HI] * CUBE_RES).pow(2).mean()                  # in cells^2
            rest = torch.cat([diff[:, :SDF_LO], diff[:, SDF_HI:]], dim=1).pow(2).mean()
            leak = (F.relu(silence_margin - sdf[is_fresh]) * CUBE_RES).mean()               # in cells
            loss = leak + sdf_shift + rest
            with torch.no_grad():
                stats = {"fresh_cells_inside": float((sdf[is_fresh] < 0).any(dim=1).float().mean()),
                         "other_cells_sdf_shift": float(sdf_shift.sqrt()), "silence_iters": i}
            if i % 10 == 0 or i == iters:
                print(f"[{tag}] silence {i:03d}: {stats['fresh_cells_inside'] * 100:.2f}% of the new cells inside, "
                      f"other cells' SDF moved {stats['other_cells_sdf_shift']:.3f} cells (rms)", flush=True)
            if i == iters or (stats["fresh_cells_inside"] < 1e-3 and stats["other_cells_sdf_shift"] < 0.05):
                break
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            p.grad = torch.nan_to_num(p.grad)
            p.grad[~fresh] = 0                                         # existing latents stay as they are
            scaler.step(opt)
            scaler.update()
        del ref, ref_feats, index
        return p.detach(), stats

    def carry_adam(old_opt, old_param, new_param, keep_idx: torch.Tensor):
        """A new Adam for the resized parameter that keeps the moments of the rows that stay."""
        opt = torch.optim.Adam([new_param], lr=lr)
        state = old_opt.state.get(old_param)
        if state:
            pad = new_param.shape[0] - keep_idx.numel()
            opt.state[new_param] = {
                "step": state["step"],
                "exp_avg": torch.cat([state["exp_avg"][keep_idx], state["exp_avg"].new_zeros((pad, new_param.shape[1]))]),
                "exp_avg_sq": torch.cat([state["exp_avg_sq"][keep_idx], state["exp_avg_sq"].new_zeros((pad, new_param.shape[1]))]),
            }
        return opt

    def mesh_distance_mm(a, b, n: int = 100000) -> float:
        if not (a.success and b.success):
            return float("nan")
        pa = np.asarray(trimesh.Trimesh(a.vertices.detach().cpu().numpy(), a.faces.cpu().numpy(), process=False).sample(n))
        pb = np.asarray(trimesh.Trimesh(b.vertices.detach().cpu().numpy(), b.faces.cpu().numpy(), process=False).sample(n))
        return float(0.5 * (cKDTree(pb).query(pa)[0].mean() + cKDTree(pa).query(pb)[0].mean()) * scale * 1000)

    sampling_floor = mesh_distance_mm(prompt_mesh, prompt_mesh)

    observed = None

    def observed_grid():
        nonlocal observed
        if observed is None:
            ids = np.unique(np.linspace(0, N - 1, min(int(cfg.get("guided_views", 48)), N)).round().astype(int))
            observed = observed_surface_grid([gt_buffers(int(i)) for i in ids], extrinsics[ids], intr,
                                             DECODER_RES, float(cfg.get("guided_band_voxels", 1.0)),
                                             int(cfg.get("guided_free_votes", 2)))
            print(f"[guided] the GT depth maps mark {int(observed.sum())} voxels as holding a surface")
        return observed

    # ---- one arm --------------------------------------------------------------------------------
    def run_arm(name: str, arm: Dict, arm_dir: Path) -> Dict:
        t0 = time.time()
        gen = torch.Generator(device=device).manual_seed(seed)
        rng = np.random.default_rng(seed)
        strategy = str(arm.get("strategy", "none"))
        init_mode, noise_std = str(arm.get("init", "nearest")), float(arm.get("noise_std", 0.0))

        coords, param_t = coords0.clone(), param0.clone()
        is_new = torch.zeros(coords.shape[0], dtype=torch.bool, device=device)

        # one-shot dilation before the optimization
        if strategy in ("shell", "random", "guided"):
            budget = max(max_active - coords.shape[0], 0)
            new_xyz = dilation_candidates(
                coords, DECODER_RES, radius=int(arm.get("radius", 1)), connectivity=int(arm.get("connectivity", 26)),
                restrict=observed_grid() if strategy == "guided" else None,
                keep_prob=float(arm.get("keep_prob", 0.5 if strategy == "random" else 1.0)), max_new=budget, generator=gen)
            if new_xyz.shape[0] == budget:
                print(f"[{name}] WARNING: capped at max_active_voxels={max_active}; the shell is subsampled")
            coords, param_t, is_new = add_voxels(coords, param_t, is_new, new_xyz, init_mode, noise_std, gen)
        elif strategy not in ("none", "grow"):
            raise SystemExit(f"unknown strategy {strategy}")
        in_region0 = region.contains_mesh_frame(voxel_centers(coords), margin=0.5 * scale / DECODER_RES)
        print(f"[{name}] {coords.shape[0]} voxels, {int(is_new.sum())} new, {int((is_new & in_region0).sum())} new in the region")

        param = nn.Parameter(param_t.clone())
        log = open(arm_dir / "metrics.jsonl", "w")

        def record(entry: Dict):
            log.write(json.dumps(entry) + "\n")
            log.flush()

        # how much did activating the voxels change the decoded mesh, before any optimization
        with torch.no_grad():
            dilated_mesh = decode(param, coords)
        disturbance = mesh_distance_mm(prompt_mesh, dilated_mesh)
        print(f"[{name}] decoding with the new voxels moves the mesh by {disturbance:.2f} mm (chamfer to the prompt mesh; "
              f"two samplings of the same mesh give {sampling_floor:.2f} mm)")

        # silent activation: the new voxels must not change the mesh before the refinement starts
        silence_iters = min(int(arm.get("silence_iters", 0)), 40 if check else 10 ** 9)
        silence_stats = {}
        if silence_iters > 0 and is_new.any():
            silenced, silence_stats = silence(coords, param.detach(), is_new, silence_iters, name)
            param = nn.Parameter(silenced.clone())
            with torch.no_grad():
                silent_mesh = decode(param, coords)
            silence_stats["disturbance_after_silence_mm"] = mesh_distance_mm(prompt_mesh, silent_mesh)
            print(f"[{name}] after silencing the mesh is {silence_stats['disturbance_after_silence_mm']:.2f} mm from the "
                  f"prompt mesh (was {disturbance:.2f} mm; two samplings of the same mesh give {sampling_floor:.2f} mm)")

        init_metrics, init_mesh = evaluate(coords, param.detach(), is_new, None, str(arm_dir / "region" / "iter_init_view_{:03d}.png"))
        init_metrics["dilation_disturbance_mm"] = disturbance
        init_metrics["sampling_floor_mm"] = sampling_floor
        init_metrics.update(silence_stats)
        record({"iter": -1, **init_metrics})
        print(f"[{name}] init {json.dumps(init_metrics)}", flush=True)
        if init_mesh.success:
            export_mesh(init_mesh, arm_dir / "init_mesh_world.glb", A)
        torch.save({"coords": coords.cpu(), "is_new": is_new.cpu(), "feats": (param.detach() * slat_std + slat_mean).cpu()},
                   arm_dir / "voxels_init.pt")

        optimizer = torch.optim.Adam([param], lr=lr)
        scaler = torch.amp.GradScaler('cuda', enabled=bool(cfg.get("loss_scaling", True)))
        grow_every, grow_until = int(arm.get("grow_every", 25)), int(arm.get("grow_until", num_iters))
        peak_mem, last = 0.0, {}

        for i in range(num_iters + 1):
            # progressive growth: activate the voxels around the current surface, drop the new ones it left
            if strategy == "grow" and i % grow_every == 0 and i <= grow_until:
                with torch.no_grad():
                    mesh_now = decode(param, coords)
                if mesh_now.success:
                    surf = surface_grid(mesh_now.vertices)
                    keep = torch.ones(coords.shape[0], dtype=torch.bool, device=device)
                    if bool(arm.get("prune", True)):
                        near = dilate_grid(surf, int(arm.get("prune_radius", 2)))
                        xyz = coords[:, 1:].long()
                        keep = ~is_new | near[xyz[:, 0], xyz[:, 1], xyz[:, 2]]
                    keep_idx = keep.nonzero().flatten()
                    budget = max(max_active - int(keep.sum()), 0)
                    new_xyz = dilation_candidates(
                        coords[keep], DECODER_RES, radius=int(arm.get("radius", 1)),
                        connectivity=int(arm.get("connectivity", 26)), seed=surf,
                        keep_prob=float(arm.get("keep_prob", 1.0)),
                        max_new=min(budget, int(arm.get("max_new_per_round", budget))), generator=gen)
                    coords_n, param_n, is_new_n = add_voxels(coords[keep], param.detach()[keep], is_new[keep],
                                                             new_xyz, init_mode, noise_std, gen)
                    round_iters = min(int(arm.get("silence_iters", 0)), 40 if check else 10 ** 9)
                    if round_iters > 0 and new_xyz.shape[0]:
                        fresh = torch.zeros(coords_n.shape[0], dtype=torch.bool, device=device)
                        fresh[-new_xyz.shape[0]:] = True
                        param_n, _ = silence(coords_n, param_n, fresh, round_iters, name)
                    new_param = nn.Parameter(param_n.clone())
                    optimizer = carry_adam(optimizer, param, new_param, keep_idx)
                    pruned = int((~keep).sum())
                    coords, param, is_new = coords_n, new_param, is_new_n
                    in_reg = region.contains_mesh_frame(voxel_centers(coords), margin=0.5 * scale / DECODER_RES)
                    record({"iter": i, "event": "grow", "added": int(new_xyz.shape[0]), "pruned": pruned,
                            "voxels": int(coords.shape[0]), "new_voxels": int(is_new.sum()),
                            "new_in_region": int((is_new & in_reg).sum())})
                    print(f"[{name}] grow @ {i:04d}: +{new_xyz.shape[0]} -{pruned} -> {coords.shape[0]} voxels "
                          f"({int(is_new.sum())} new, {int((is_new & in_reg).sum())} in the region)", flush=True)

            indices = np.arange(N) if frames_per_iter >= N else rng.choice(N, size=frames_per_iter, replace=False)
            mesh = decode(param, coords)
            if not mesh.success:
                print(f"[{name}] iter {i:04d}: empty mesh, skipped", flush=True)
                continue
            comps, iou = loss_terms(mesh, indices, gt_buffers)
            total = sum(comps.values())

            if grad_interval and (i % grad_interval == 0 or i == num_iters):
                split = grad_split(comps, param, coords, is_new)
                record({"iter": i, "event": "grad", "mean_abs_grad_per_voxel": split})
                if i == 0:
                    print(f"[{name}] mean |grad| per voxel at iteration 0 (old / new / new in the region):")
                    for k, v in split.items():
                        print(f"[{name}]   {k:>7s}: {v['old']:.3e} / {v['new']:.3e} / {v['new_in_region']:.3e}")

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nan_to_num(param.grad).norm())
            scaler.step(optimizer)
            scaler.update()
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / 2 ** 30)

            if i % 10 == 0 or i == num_iters:
                print(f"[{name}] iter {i:04d} total {total.item():.4f} | "
                      + " ".join(f"{k} {v.item():.4f}" for k, v in comps.items())
                      + f" | IoU {iou:.4f} | grad {grad_norm:.2e} | {coords.shape[0]} vox | {peak_mem:.1f} GB", flush=True)

            if eval_interval and (i % eval_interval == 0 or i == num_iters):
                last, _ = evaluate(coords, param.detach(), is_new, str(arm_dir / "test" / f"test_{i:04d}_{{:03d}}.png"),
                                   str(arm_dir / "region" / f"iter_{i:04d}_view_{{:03d}}.png"))
                record({"iter": i, "loss": total.item(), **last})
                print(f"[{name}] eval {i:04d} region recall@2vox {last.get('region_recall_2vox', 0) * 100:.1f}% "
                      f"dist {last.get('region_dist_mm', float('nan')):.2f} mm | chamfer {last.get('chamfer_mm', float('nan')):.2f} mm "
                      f"| IoU {last.get('mask_iou', 0):.4f} | new voxels with surface {last.get('new_with_surface', 0)}", flush=True)

        final, final_mesh = evaluate(coords, param.detach(), is_new, None, str(arm_dir / "final" / "region_view_{:03d}.png"))
        if final_mesh.success:
            export_mesh(final_mesh, arm_dir / "final" / "mesh_world.glb", A)
        torch.save({"coords": coords.cpu(), "feats": (param.detach() * slat_std + slat_mean).cpu(),
                    "is_new": is_new.cpu(), "alignment": A}, arm_dir / "final" / "slat.pt")
        result = {"arm": name, "strategy": strategy, "settings": arm, "prompt": base_metrics, "init": init_metrics,
                  "final": final, "minutes": round((time.time() - t0) / 60, 2), "peak_memory_gb": round(peak_mem, 1),
                  "iterations": num_iters, "version": VERSION}
        record({"iter": "final", **final})
        log.close()
        print(f"[{name}] done in {result['minutes']} min: region recall@2vox "
              f"{init_metrics.get('region_recall_2vox', 0) * 100:.1f}% -> {final.get('region_recall_2vox', 0) * 100:.1f}%", flush=True)
        return result

    # ---- arms --------------------------------------------------------------------------------
    all_arms = cfg["arms"]
    names = arms or list(all_arms)
    summary = {}
    for name in names:
        if name not in all_arms:
            raise SystemExit(f"unknown arm {name}; the config has {list(all_arms)}")
        arm = dict(all_arms[name])
        arm_dir = out_root / name
        done = arm_dir / "done.json"
        if done.exists() and not force:
            previous = json.loads(done.read_text())
            if previous.get("version") == VERSION:
                summary[name] = previous
                print(f"\n=== {name}: finished already, skipped ({done})")
                continue
            print(f"\n=== {name}: finished by an older version of this script, redone")
            done.unlink()
        for sub in ("test", "region", "final"):
            (arm_dir / sub).mkdir(parents=True, exist_ok=True)
        print(f"\n=== {name}: {arm}", flush=True)
        try:
            result = run_arm(name, arm, arm_dir)
        except torch.cuda.OutOfMemoryError as e:
            print(f"[{name}] OUT OF MEMORY, arm skipped: lower max_active_voxels or the radius ({str(e)[:120]})", flush=True)
            (arm_dir / "failed.txt").write_text(str(e))
            continue
        finally:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        if not check:
            done.write_text(json.dumps(result, indent=2))
        summary[name] = result

    # ---- comparison over the arms --------------------------------------------------------------
    for other in sorted(out_root.glob("*/done.json")):                 # arms finished in an earlier session
        previous = json.loads(other.read_text())
        if previous.get("version") == VERSION:
            summary.setdefault(other.parent.name, previous)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    pct = lambda r, k: f"{r.get(k, 0) * 100:.1f}"
    num = lambda r, k: f"{r.get(k, float('nan')):.2f}"
    lines = [f"Structure: {cfg['region'].get('name', 'region')}; prompt {prompt_name}; one latent voxel = {voxel_mm:.2f} mm.", "",
             "| arm | voxels (new) | new voxels with surface | region recall @2 voxels, init -> final (%) "
             "| region distance, init -> final (mm) | chamfer, init -> final (mm) | test IoU | test PSNR | min | peak GB |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in summary.items():
        a, b = r["init"], r["final"]
        lines.append(f"| {name} | {b.get('voxels', 0)} ({b.get('new_voxels', 0)}) | {b.get('new_with_surface', 0)} "
                     f"| {pct(a, 'region_recall_2vox')} -> {pct(b, 'region_recall_2vox')} "
                     f"| {num(a, 'region_dist_mm')} -> {num(b, 'region_dist_mm')} "
                     f"| {num(a, 'chamfer_mm')} -> {num(b, 'chamfer_mm')} | {b.get('mask_iou', 0):.4f} "
                     f"| {num(b, 'psnr')} | {r['minutes']} | {r['peak_memory_gb']} |")
    (out_root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nsummary: {out_root / 'summary.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grow a missing structure by dilating the SLAT's active voxels")
    parser.add_argument("--yaml-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--arms", nargs="*", default=None, help="default: every arm in the config")
    parser.add_argument("--check", action="store_true", help="setup, dilation and 2 iterations per arm, into out_dir/check")
    parser.add_argument("--force", action="store_true", help="resample the prompt and redo finished arms")
    parser.add_argument("--out-dir", default=None, help="overrides out_dir in the config")
    args = parser.parse_args()
    run(args.yaml_config, args.arms, args.check, args.force, args.out_dir)
