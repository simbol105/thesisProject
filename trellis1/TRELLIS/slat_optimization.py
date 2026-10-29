"""Optimize the full-resolution TRELLIS SLAT against a GT mesh with a rendering loss."""
import os
os.environ['SPCONV_ALGO'] = 'native'
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

import argparse
import itertools
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import trimesh
import yaml
from PIL import Image
from scipy.spatial import cKDTree

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.renderers.mesh_renderer import MeshRenderer
from trellis.representations.mesh.cube2mesh import MeshExtractResult
from trellis.modules import sparse as sp
from opt_utils.utility import save_image, unwrap_mesh, align_meshes_fixed_scale_ransac
from hypothesis_slat_downsample import (
    RES,
    compute_slat_mesh_decoder_losses,
    _tsdf_reg_loss_from_depth,
    _scale_opencv_intrinsics,
    _load_json,
    _try_image_path,
    ensure_nchw,
)


def _proper_rotations_24() -> List[np.ndarray]:
    """The 24 rotations of the cube (signed permutation matrices with det +1)."""
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1.0, -1.0], repeat=3):
            R = np.zeros((3, 3))
            R[range(3), perm] = signs
            if np.linalg.det(R) > 0:
                mats.append(R)
    return mats


def _apply(A: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ A[:3, :3].T + A[:3, 3]


def _chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """Mean symmetric nearest-neighbour distance."""
    return 0.5 * (cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean())


def split_similarity(A: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    s = float(np.cbrt(np.linalg.det(A[:3, :3])))
    return s, A[:3, :3] / s, A[:3, 3]


def alignment_candidates(pred_mesh: trimesh.Trimesh, gt_mesh: trimesh.Trimesh, n_points: int = 8000):
    """ICP (with scale) started from each of the 24 axis-aligned rotations."""
    src = np.asarray(pred_mesh.sample(n_points))
    tgt = np.asarray(gt_mesh.sample(n_points))
    c_s, c_t = src.mean(0), tgt.mean(0)
    scale0 = np.sqrt(((tgt - c_t) ** 2).sum(1).mean() / ((src - c_s) ** 2).sum(1).mean())
    diag = np.linalg.norm(tgt.max(0) - tgt.min(0))

    src_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src))
    tgt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt))
    est = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
    crit = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)

    candidates = []
    for k, R in enumerate(_proper_rotations_24()):
        A = np.eye(4)
        A[:3, :3] = scale0 * R
        A[:3, 3] = c_t - scale0 * R @ c_s
        for thr in (0.1 * diag, 0.03 * diag):
            A = o3d.pipelines.registration.registration_icp(src_pcd, tgt_pcd, thr, A, est, crit).transformation
        s = split_similarity(A)[0]
        if not (0.5 * scale0 < s < 2.0 * scale0):
            continue
        candidates.append({"name": f"rot{k:02d}", "A": A, "chamfer": _chamfer(_apply(A, src), tgt)})
    return candidates


@torch.no_grad()
def mean_silhouette_iou(renderer, mesh_a, mesh_b, extrinsics, intrinsics) -> float:
    ious = []
    for E in extrinsics:
        ma = renderer.render(mesh_a, E, intrinsics, ["mask"])["mask"] > 0.5
        mb = renderer.render(mesh_b, E, intrinsics, ["mask"])["mask"] > 0.5
        ious.append((ma & mb).sum() / (ma | mb).sum().clamp_min(1))
    return torch.stack(ious).float().mean().item()


def transformed_mesh(mesh: MeshExtractResult, A: torch.Tensor) -> MeshExtractResult:
    return MeshExtractResult(
        vertices=mesh.vertices @ A[:3, :3].T + A[:3, 3],
        faces=mesh.faces,
        vertex_attrs=mesh.vertex_attrs,
    )


def select_alignment(pred_mesh, gt_world, gt_world_trimesh, renderer, extrinsics, intr, n_views, top_k, device):
    pred_trimesh = unwrap_mesh(pred_mesh)
    candidates = sorted(alignment_candidates(pred_trimesh, gt_world_trimesh), key=lambda c: c["chamfer"])[:top_k]
    try:
        A = align_meshes_fixed_scale_ransac(pred_mesh, gt_world_trimesh)
        src = np.asarray(pred_trimesh.sample(20000))
        candidates.append({"name": "fpfh_ransac", "A": A,
                           "chamfer": _chamfer(_apply(A, src), np.asarray(gt_world_trimesh.sample(20000)))})
    except Exception as e:
        print(f"FPFH-RANSAC alignment failed: {e}")

    view_ids = np.linspace(0, len(extrinsics) - 1, n_views).round().astype(int)
    views = [extrinsics[i] for i in view_ids]
    print("Alignment candidates (TRELLIS -> world):")
    for c in candidates:
        A_t = torch.tensor(np.array(c["A"]), dtype=torch.float32, device=device)
        c["iou"] = mean_silhouette_iou(renderer, transformed_mesh(pred_mesh, A_t), gt_world, views, intr)
        print(f"  {c['name']:>12s}  scale={split_similarity(c['A'])[0]:.4f}  "
              f"chamfer={c['chamfer'] * 1000:.2f}mm  silhouette IoU={c['iou']:.4f}")
    best = max(candidates, key=lambda c: c["iou"])
    print(f"Selected alignment: {best['name']} (IoU={best['iou']:.4f})")
    return best["A"]


def export_mesh(mesh: MeshExtractResult, path: Path, A: np.ndarray = None):
    v = mesh.vertices.detach().cpu().numpy()
    if A is not None:
        v = _apply(A, v)
    colors = None
    if mesh.vertex_attrs is not None:
        colors = (mesh.vertex_attrs[:, :3].detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    trimesh.Trimesh(vertices=v, faces=mesh.faces.cpu().numpy(), vertex_colors=colors, process=False).export(path)


def chamfer_to_gt(mesh: MeshExtractResult, A: np.ndarray, gt_world_trimesh: trimesh.Trimesh, n: int = 50000) -> float:
    pred = trimesh.Trimesh(vertices=_apply(A, mesh.vertices.detach().cpu().numpy()),
                           faces=mesh.faces.cpu().numpy(), process=False)
    return _chamfer(np.asarray(pred.sample(n)), np.asarray(gt_world_trimesh.sample(n)))


def grad_wrt(loss: torch.Tensor, z: torch.Tensor, scale: float) -> Tuple[float, float]:
    """(grad norm, fraction of non-zero grad entries) of `loss` w.r.t. `z` with loss scaling."""
    if not loss.requires_grad:
        return 0.0, 0.0
    g, = torch.autograd.grad(loss * scale, z, retain_graph=True, allow_unused=True)
    if g is None:
        return 0.0, 0.0
    g = g / scale
    return g.norm().item(), (g != 0).float().mean().item()


def refine_latent(config_path: Path, out_dir_override: Path = None):
    cfg = yaml.safe_load(open(config_path, 'r'))

    lr = float(cfg['lr'])
    num_iters = int(cfg['num_iters'])
    frames_per_iter = int(cfg['frames_per_iter'])
    near = float(cfg.get('near', 0.01))
    far = float(cfg.get('far', 100.0))
    weight = float(cfg.get('weight', 5.0))
    sdf_weight = float(cfg.get('sdf_weight', 1.0))
    tsdf_weight = float(cfg.get('tsdf_weight', 1.0))
    latent_l2_weight = float(cfg.get('latent_l2_weight', 0.0))
    lambda_mask = float(cfg.get('lambda_mask', 1.0))
    lambda_depth = float(cfg.get('lambda_depth', 1.2))
    lambda_normal = float(cfg.get('lambda_normal', 1.0))
    lambda_color = float(cfg.get('lambda_color', 0.9))
    lambda_ssim = float(cfg.get('lambda_ssim', 0.2))
    lambda_lpips = float(cfg.get('lambda_lpips', 0.2))
    loss_scaling = bool(cfg.get('loss_scaling', True))
    grad_clip = float(cfg.get('grad_clip', 0.0))
    display_interval = int(cfg.get('display_interval', 10))
    seed = int(cfg.get('seed', 1))
    align_views = int(cfg.get('align_iou_views', 16))
    align_top_k = int(cfg.get('align_top_k', 4))
    alignment_path = cfg.get('alignment_path', '')
    data_dir = Path(cfg['data_dir'])
    out_dir = Path(out_dir_override or cfg.get('out_dir', 'optimization_output'))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda')
    torch.manual_seed(seed)

    transforms = _load_json(data_dir / 'transforms_train.json')
    frames = transforms['frames']
    object_pose = np.array(transforms['object_pose'], dtype=np.float64)
    image_paths = [_try_image_path(data_dir / f['file_path']) for f in frames]
    N = len(image_paths)
    prompt_idx = int(cfg.get('prompt_idx', 0))
    vis_views = [int(v) for v in cfg.get('vis_views', [prompt_idx, N // 4, N // 2, 3 * N // 4])]

    prompt_img = Image.open(image_paths[prompt_idx])
    intr = _scale_opencv_intrinsics(transforms['intrinsic'], prompt_img.size, RES).to(device)
    extrinsics_world = [torch.tensor(f['transform_matrix'], dtype=torch.float32, device=device) for f in frames]

    renderer = MeshRenderer(
        rendering_options={"resolution": RES, "near": near, "far": far, "ssaa": 1},
        device=device.type,
    )

    print("Loading TRELLIS pipeline...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()
    outputs = pipeline.run(prompt_img, seed=seed, formats=['mesh'], return_latent=True)
    slat = outputs['latent']
    init_mesh = outputs['mesh'][0]
    print(f"SLAT: {slat.feats.shape[0]} voxels x {slat.feats.shape[1]} channels "
          f"(feat mean {slat.feats.mean():.3f}, std {slat.feats.std():.3f})")

    gt_trimesh = trimesh.load_mesh(data_dir / 'model.glb', force='mesh')
    gt_trimesh.apply_transform(object_pose)
    gt_trimesh.visual = gt_trimesh.visual.to_color()
    gt_colors = torch.from_numpy(np.asarray(gt_trimesh.visual.vertex_colors)[:, :3] / 255.0).float().to(device)
    gt_world_trimesh = trimesh.Trimesh(vertices=gt_trimesh.vertices, faces=gt_trimesh.faces, process=False)
    gt_world = MeshExtractResult(
        vertices=torch.from_numpy(gt_trimesh.vertices).float().to(device),
        faces=torch.from_numpy(gt_trimesh.faces).long().to(device),
        vertex_attrs=gt_colors,
    )
    print(f"GT extent (world units): {gt_world_trimesh.extents}")

    if alignment_path and Path(alignment_path).exists():
        A = np.array(np.load(alignment_path))
        print(f"Loaded alignment from {alignment_path}")
    else:
        A = select_alignment(init_mesh, gt_world, gt_world_trimesh, renderer, extrinsics_world, intr,
                             align_views, align_top_k, device)
    A = np.array(A)
    np.save(out_dir / 'alignment.npy', A)
    s, R, t = split_similarity(A)
    print(f"Alignment scale {s:.4f} (1 TRELLIS unit = {s:.4f} world units)")

    R_t = torch.as_tensor(R, dtype=torch.float32, device=device)
    t_t = torch.as_tensor(t, dtype=torch.float32, device=device)
    A_t = torch.as_tensor(A, dtype=torch.float32, device=device)
    gt_mesh = MeshExtractResult(
        vertices=((gt_world.vertices - t_t) @ R_t) / s,
        faces=gt_world.faces,
        vertex_attrs=gt_colors,
    )
    extrinsics = []
    for E in extrinsics_world:
        E_t = E @ A_t
        E_t[:3, :] = E_t[:3, :] / s
        extrinsics.append(E_t)

    gt_cache: Dict[int, dict] = {}

    @torch.no_grad()
    def gt_buffers(idx: int) -> dict:
        if idx not in gt_cache:
            gt_cache[idx] = renderer.render(gt_mesh, extrinsics[idx], intr, ["color", "mask", "normal", "depth"])
        return gt_cache[idx]

    decoder = pipeline.models['slat_decoder_mesh']
    decoder.requires_grad_(False)
    decoder.train()
    print(f"Decoder: resolution {decoder.resolution}, torso dtype {decoder.dtype}")

    slat_mean = torch.tensor(pipeline.slat_normalization['mean'], device=device)[None]
    slat_std = torch.tensor(pipeline.slat_normalization['std'], device=device)[None]
    z0 = ((slat.feats - slat_mean) / slat_std).detach()
    z = nn.Parameter(z0.clone())

    def decode(z_: torch.Tensor) -> MeshExtractResult:
        return decoder(sp.SparseTensor(feats=z_ * slat_std + slat_mean, coords=slat.coords))[0]

    if cfg.get('optimizer', '') == 'sgd':
        optimizer = torch.optim.SGD([z], lr=lr)
    else:
        optimizer = torch.optim.Adam([z], lr=lr)
    scaler = torch.amp.GradScaler('cuda', enabled=loss_scaling)

    with torch.no_grad():
        mesh = decode(z)
        export_mesh(mesh, out_dir / 'mesh_init_world.ply', A)
        cd_init = chamfer_to_gt(mesh, A, gt_world_trimesh)
        iou_init = mean_silhouette_iou(renderer, mesh, gt_mesh, [extrinsics[i] for i in vis_views], intr)
    print(f"Init: chamfer {cd_init * 1000:.2f}mm, silhouette IoU {iou_init:.4f}, {mesh.vertices.shape[0]} verts")

    rng = np.random.default_rng(seed=seed)
    for it in range(num_iters + 1):
        indices = np.arange(N) if frames_per_iter >= N else rng.choice(N, size=frames_per_iter, replace=False)

        mesh = decode(z)
        comps = {k: torch.zeros((), device=device) for k in ["mask", "depth", "normal", "color", "tsdf"]}
        ious = []
        for idx in indices:
            gt = gt_buffers(int(idx))
            m = gt["mask"]
            pred = renderer.render(mesh, extrinsics[idx], intr, ["color", "mask", "normal", "depth"])
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
                ious.append(((pm & gm).sum() / (pm | gm).sum().clamp_min(1)).item())
        comps = {k: v / len(indices) for k, v in comps.items()}
        comps["reg"] = sdf_weight * mesh.reg_loss
        comps["latent_l2"] = latent_l2_weight * ((z - z0) ** 2).mean()
        total = sum(comps.values())

        if it == 0:
            print("[diag] grad of total loss w.r.t. latent:")
            for sc in (1.0, 1024.0):
                gn, nz = grad_wrt(total, z, sc)
                print(f"[diag]   loss scale {sc:>6.0f}: norm {gn:.3e}, non-zero entries {nz * 100:.1f}%")
            print("[diag] per-term grad (loss scale 1024):")
            for k, v in comps.items():
                gn, nz = grad_wrt(v, z, 1024.0)
                print(f"[diag]   {k:>9s}: value {v.item():.4e}, grad norm {gn:.3e}, non-zero {nz * 100:.1f}%")

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        grad_norm = z.grad.norm().item()
        nonzero = (z.grad != 0).float().mean().item()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([z], max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            drift = (z - z0).abs().mean().item()
        print(f"[Iter {it:04d}] total {total.item():.4f} | "
              + " ".join(f"{k} {v.item():.4f}" for k, v in comps.items())
              + f" | IoU {np.mean(ious):.4f} | grad {grad_norm:.3e} ({nonzero * 100:.0f}% nz)"
              + f" | scale {scaler.get_scale():.0f} | |z-z0| {drift:.4f} | verts {mesh.vertices.shape[0]}")

        if it % display_interval == 0 or it == num_iters:
            with torch.no_grad():
                for v in vis_views:
                    gt = gt_buffers(v)
                    pred = renderer.render(mesh, extrinsics[v], intr, ["color", "mask", "normal"])
                    overlay = torch.stack([pred["mask"], gt["mask"], torch.zeros_like(gt["mask"])], dim=0)
                    vis = torch.cat([ensure_nchw(pred["color"]), ensure_nchw(gt["color"]),
                                     ensure_nchw(pred["normal"]), ensure_nchw(gt["normal"]),
                                     ensure_nchw(overlay)], dim=3)
                    save_image(str(out_dir / f"opt_{it:04d}_view{v:03d}.png"),
                               vis[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy())

    with torch.no_grad():
        mesh = decode(z)
        export_mesh(mesh, out_dir / 'mesh_final_world.ply', A)
        cd_final = chamfer_to_gt(mesh, A, gt_world_trimesh)
        iou_final = mean_silhouette_iou(renderer, mesh, gt_mesh, [extrinsics[i] for i in vis_views], intr)
    torch.save({"coords": slat.coords.cpu(), "feats": (z * slat_std + slat_mean).detach().cpu(),
                "alignment": A}, out_dir / 'slat_final.pt')
    print(f"Chamfer {cd_init * 1000:.2f}mm -> {cd_final * 1000:.2f}mm, "
          f"silhouette IoU {iou_init:.4f} -> {iou_final:.4f}")
    print("Latent optimization completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize the full-resolution TRELLIS SLAT")
    parser.add_argument("--yaml-config", type=Path, required=True, help="Path to the YAML config file")
    parser.add_argument("--out-dir", type=Path, default=None, help="Overrides out_dir from the config")
    args = parser.parse_args()
    refine_latent(args.yaml_config, args.out_dir)
