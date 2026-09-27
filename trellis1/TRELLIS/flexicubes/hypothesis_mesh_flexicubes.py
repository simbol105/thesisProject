"""
Direct FlexiCubes optimization of a reconstructed mesh -- no TRELLIS decoder in the loop.

This is the trellis1 / FlexiCubes counterpart of
trellis2/TRELLIS.2/hypothesis_trellis_init.py, which takes the initial mesh, converts it to an
SDF on a DMTet grid and optimizes (sdf, deform) directly. Here:

    recgen / SAM3D mesh
      -> SDF on the (res+1)^3 cube grid            (Sparc3D-style UDF + exterior flood fill)
      -> nn.Parameter sdf / deform / weight / colour
      -> FlexiCubes                                -> mesh
      -> MeshRenderer vs. the DTC GT mesh          -> backprop into those parameters

It is the ablation partner of cubefeats/hypothesis_mesh_cubefeats.py: same objects, same
alignment, same cameras, same losses, same metrics and the same FlexiCubes extractor on the
same grid -- the only difference is that here sdf/deform/weights are free parameters seeded
from the mesh, whereas there they are produced by the SLAT mesh decoder from an encoded SLAT.
Nothing in this script loads a TRELLIS model.

Usage
-----
    python flexicubes/hypothesis_mesh_flexicubes.py --yaml-config flexicubes_configs/<obj>.yaml --check
    sbatch flexicubes/run_flexicubes.sh flexicubes_configs/<obj>.yaml [--mesh-source sam3d]
"""

import os

os.environ.setdefault('ATTN_BACKEND', 'flash-attn')
os.environ.setdefault('SPCONV_ALGO', 'native')

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]      # .../trellis1/TRELLIS
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import yaml
from PIL import Image

from trellis.renderers.mesh_renderer import MeshRenderer
from opt_utils.utility import save_image

# Everything shared with the cubefeats run is imported, not copied, so the two runs cannot
# drift apart in their losses, alignment, metrics or output layout.
from cubefeats.hypothesis_mesh_cubefeats import (
    RES,
    _load_json,
    _resize_rgb_to_tensor,
    _scale_opencv_intrinsics,
    _try_image_path,
    compute_alignment,
    compute_slat_mesh_decoder_losses,
    ensure_nchw,
    eval_views,
    export_mesh,
    load_source_mesh,
    mesh_vertex_colors,
    to_mesh_extract_result,
    tsdf_reg_loss,
)
from flexicubes.flexicubes_geometry import FlexiCubesGeometry

MESH_KEY = {"recgen": "recgen_mesh_path", "sam3d": "sam3d_mesh_path"}


def build_setup(cfg: Dict, mesh_source: str, device):
    """Cameras, GT mesh, alignment and the SDF-initialized FlexiCubes grid."""
    seed = int(cfg.get("seed", 1))
    data_dir = Path(cfg["data_dir"])
    assert data_dir.exists(), f"Data directory does not exist: {data_dir}"

    transforms_json = data_dir / "transforms_train.json"
    test_transforms_json = data_dir / "transforms_test.json"
    gt_mesh_path = Path(cfg.get("gt_mesh", data_dir / "model.glb"))

    train_transforms = _load_json(transforms_json)
    test_transforms = _load_json(test_transforms_json)
    frames = train_transforms["frames"]
    base_object_pose = np.array(train_transforms["object_pose"], dtype=np.float32)

    image_paths = [_try_image_path(transforms_json.parent / f["file_path"]) for f in frames]
    intr_scaled = _scale_opencv_intrinsics(
        train_transforms["intrinsic"], Image.open(image_paths[0]).size
    ).to(device)

    extrinsics_t = torch.stack(
        [torch.tensor(f["transform_matrix"], dtype=torch.float32) for f in frames], dim=0
    ).to(device)
    test_extrinsics_t = torch.stack(
        [torch.tensor(f["transform_matrix"], dtype=torch.float32) for f in test_transforms["frames"]],
        dim=0,
    ).to(device)

    renderer = MeshRenderer(
        rendering_options={
            "resolution": RES,
            "near": float(cfg.get("near", 0.01)),
            "far": float(cfg.get("far", 1000.0)),
            "ssaa": int(cfg.get("ssaa", 1)),
        },
        device=device.type,
    )

    # ---- 1. the reconstruction we start from ----------------------------------------------
    src_path = Path(cfg[MESH_KEY[mesh_source]])
    assert src_path.exists(), f"{mesh_source} mesh does not exist: {src_path}"
    print(f"[mesh] {mesh_source}: {src_path}")
    src_tm = load_source_mesh(src_path, normalize=bool(cfg.get("normalize_mesh", True)))
    src_colors = mesh_vertex_colors(src_tm)
    print(f"[mesh] {len(src_tm.vertices)} verts, {len(src_tm.faces)} faces, "
          f"watertight={src_tm.is_watertight}, bounds {np.round(src_tm.bounds, 4).tolist()}")

    # ---- 2. GT mesh in the dataset world frame --------------------------------------------
    gt_tm = trimesh.load_mesh(str(gt_mesh_path))
    gt_tm.apply_transform(base_object_pose)

    # ---- 3. mesh frame -> world similarity (folded into the cameras, as in the other run) --
    align_method = str(cfg.get("alignment", "ransac"))
    alignment = compute_alignment(src_tm, gt_tm, align_method, device)
    print(f"[align] method={align_method}\n{alignment.detach().cpu().numpy()}")
    gt_mesh = to_mesh_extract_result(gt_tm, device)

    # ---- 4. mesh -> SDF -> FlexiCubes parameters -------------------------------------------
    geometry = FlexiCubesGeometry(
        res=int(cfg.get("grid_res", 128)),
        init_mesh=src_tm,
        init_colors=src_colors,
        device=device.type,
        sdf_init=str(cfg.get("sdf_init", "floodfill")),
        tau_scale=float(cfg.get("sdf_tau_scale", 1.0)),
        sparc3d_path=cfg.get("sparc3d_path"),
        optimize_weights=bool(cfg.get("optimize_weights", True)),
        optimize_color=bool(cfg.get("optimize_color", True)),
    )

    return {
        "renderer": renderer,
        "intr": intr_scaled,
        "extrinsics_t": extrinsics_t,
        "test_extrinsics_t": test_extrinsics_t,
        "image_paths": image_paths,
        "alignment": alignment,
        "gt_mesh": gt_mesh,
        "src_mesh": to_mesh_extract_result(src_tm, device),
        "geometry": geometry,
        "seed": seed,
    }


def run(config_path: Path, mesh_source: Optional[str], out_dir_override: Optional[str], check: bool):
    cfg = yaml.safe_load(open(config_path, "r"))
    mesh_source = mesh_source or str(cfg.get("mesh_source", "recgen"))

    seed = int(cfg.get("seed", 1))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(out_dir_override or cfg["out_dir"].format(mesh_source=mesh_source))
    if check:
        out_dir = out_dir / "check"
    out_dir.mkdir(parents=True, exist_ok=True)
    test_out_dir = out_dir / "test"
    test_out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[out] {out_dir}")

    setup = build_setup(cfg, mesh_source, device)
    renderer = setup["renderer"]
    intr = setup["intr"]
    extrinsics_t = setup["extrinsics_t"]
    test_extrinsics_t = setup["test_extrinsics_t"]
    alignment = setup["alignment"]
    gt_mesh = setup["gt_mesh"]
    geometry: FlexiCubesGeometry = setup["geometry"]

    np.save(out_dir / "alignment.npy", alignment.detach().cpu().numpy())

    N = extrinsics_t.shape[0]
    frames_per_iter = int(cfg.get("frames_per_iter", 10))
    display_interval = int(cfg.get("display_interval", 25))
    eval_interval = int(cfg.get("eval_interval", 50))
    num_iters = int(cfg.get("num_iters", 500))
    weight = float(cfg.get("weight", 1.5))
    sdf_weight = float(cfg.get("sdf_weight", 0.5))
    tsdf_weight = float(cfg.get("tsdf_weight", 0.5))
    sdf_l2_weight = float(cfg.get("sdf_l2_weight", 0.0))
    tsdf_band = float(cfg.get("tsdf_band", 3.0))
    rng = np.random.default_rng(seed=seed)

    eval_ids = np.unique(np.linspace(
        0, test_extrinsics_t.shape[0] - 1,
        min(int(cfg.get("eval_views", 8)), test_extrinsics_t.shape[0]),
    ).round().astype(int)).tolist()

    # ---- the mesh the SDF initialization alone gives (the baseline to beat) ----------------
    with torch.no_grad():
        init_mesh = geometry.get_mesh(training=False)
    print(f"[init] FlexiCubes mesh: {init_mesh.vertices.shape[0]} verts, {init_mesh.faces.shape[0]} faces")
    export_mesh(init_mesh, out_dir / "init_mesh" / "flexicubes_init.glb")
    export_mesh(init_mesh, out_dir / "init_mesh" / "flexicubes_init_world.glb", alignment)
    export_mesh(setup["src_mesh"], out_dir / "init_mesh" / "source_mesh_world.glb", alignment)
    init_metrics = eval_views(renderer, init_mesh, gt_mesh, test_extrinsics_t, alignment, intr,
                              eval_ids, str(out_dir / "init_mesh" / "init_{:03d}.png"))
    print(f"[init] {init_metrics}")
    with open(out_dir / "metrics.jsonl", "w") as f:
        f.write(json.dumps({"iter": -1, **init_metrics}) + "\n")

    if check:
        train_ids = np.unique(np.linspace(0, N - 1, 6).round().astype(int)).tolist()
        train_metrics = eval_views(renderer, init_mesh, gt_mesh, extrinsics_t, alignment, intr,
                                   train_ids, str(out_dir / "train_{:03d}.png"))
        print(f"[check] train views {train_metrics}")
        print("[check] done -- check that flexicubes_init.glb resembles the source mesh before "
              "launching the run; if it is hollow or empty, revisit sdf_init.")
        return

    optimizer = geometry.getOptimizer(
        float(cfg["lr"]), str(cfg.get("optimizer", "adam")).lower()
    )
    sdf_init_value = geometry.sdf.detach().clone()
    n_params = sum(p.numel() for p in geometry.parameters())
    print(f"[opt] optimizing {n_params} values on a {geometry.res}^3 grid for {num_iters} iters")

    for i in range(num_iters + 1):
        indices = (
            np.arange(N) if frames_per_iter >= N
            else np.atleast_1d(rng.choice(N, size=frames_per_iter, replace=False))
        )
        optimizer.zero_grad()

        mesh = geometry.get_mesh(training=True, tsdf_band=tsdf_band)
        if not mesh.success:
            print(f"[Iter {i:04d}] empty mesh, skipping", flush=True)
            continue
        reg_loss = mesh.reg_loss

        img_losses = torch.zeros((), device=device)
        tsdf_loss = torch.zeros((), device=device)

        for idx in indices:
            extr_gt = extrinsics_t[idx]
            extr = extr_gt @ alignment                       # mesh frame -> camera

            gt_buffers = renderer.render(
                mesh=gt_mesh, extrinsics=extr_gt, intrinsics=intr,
                return_types=["color", "mask", "normal", "depth"],
            )
            gt_mask = gt_buffers["mask"]
            gt_rgb = gt_buffers["color"] * gt_mask
            gt_normal = gt_buffers["normal"] * gt_mask
            gt_depth = gt_buffers["depth"] * gt_mask

            buffers = renderer.render(
                mesh=mesh, extrinsics=extr, intrinsics=intr,
                return_types=["color", "mask", "normal", "depth"],
            )

            if tsdf_weight > 0 and mesh.tsdf_v is not None and mesh.tsdf_v.shape[0] > 0:
                tsdf_loss = tsdf_loss + tsdf_reg_loss(
                    gt_mesh=gt_mesh, rep=mesh, extrinsics=extr, intrinsics=intr, renderer=renderer,
                )

            loss_terms = compute_slat_mesh_decoder_losses(
                gt_rgb=gt_rgb, gt_mask=gt_mask, gt_normal=gt_normal, gt_depth=gt_depth,
                pred_rgb=buffers["color"] * gt_mask,
                pred_mask=buffers["mask"],
                pred_normal=buffers["normal"] * gt_mask,
                pred_depth=buffers["depth"] * gt_mask,
                lambda_depth=float(cfg.get("lambda_depth", 1.2)),
                lambda_ssim=float(cfg.get("lambda_ssim", 0.2)),
                lambda_lpips=float(cfg.get("lambda_lpips", 0.2)),
                lambda_color=float(cfg.get("lambda_color", 0.9)),
                use_color=True,
            )
            img_losses = img_losses + loss_terms["loss"]

        img_losses = img_losses / len(indices)
        tsdf_loss = tsdf_loss / len(indices)
        total_loss = (sdf_weight * reg_loss) + (tsdf_weight * tsdf_loss) + (weight * img_losses)
        if sdf_l2_weight > 0:
            total_loss = total_loss + sdf_l2_weight * F.mse_loss(geometry.sdf, sdf_init_value)

        total_loss.backward()
        grads = {n: (getattr(geometry, n).grad.norm().item()
                     if getattr(geometry, n).grad is not None else float("nan"))
                 for n in ("sdf", "deform", "weight", "color")}
        if float(cfg.get("grad_clip", 0.0)) > 0:
            torch.nn.utils.clip_grad_norm_(geometry.parameters(), max_norm=float(cfg["grad_clip"]))
        optimizer.step()

        print(f"[Iter {i:04d}] total {float(total_loss):.4f} | img {float(img_losses):.4f} "
              f"| tsdf {float(tsdf_loss):.4f} | reg {float(reg_loss):.4f} | "
              f"grad sdf {grads['sdf']:.3e} deform {grads['deform']:.3e} "
              f"weight {grads['weight']:.3e} color {grads['color']:.3e}", flush=True)

        if display_interval and i % display_interval == 0:
            with torch.no_grad():
                vid = int(indices[0])
                pred = renderer.render(mesh=mesh, extrinsics=extrinsics_t[vid] @ alignment,
                                       intrinsics=intr, return_types=["color", "mask", "normal"])
                gt = renderer.render(mesh=gt_mesh, extrinsics=extrinsics_t[vid], intrinsics=intr,
                                     return_types=["color", "mask"])
                normal = ensure_nchw(pred["normal"])
                normal = (normal - normal.min()) / (normal.max() - normal.min() + 1e-6)
                vis = torch.cat([
                    (ensure_nchw(pred["color"]) * ensure_nchw(pred["mask"])).clamp(0, 1),
                    (ensure_nchw(gt["color"]) * ensure_nchw(gt["mask"])).clamp(0, 1),
                    normal,
                ], dim=3)[0].permute(1, 2, 0).clamp(0, 1)
                save_image(f"{out_dir}/opt_{i:04d}_{vid:03d}.png", vis.detach().cpu().numpy())

        if eval_interval and (i % eval_interval == 0 or i == num_iters):
            with torch.no_grad():
                eval_mesh = geometry.get_mesh(training=False)
            metrics = eval_views(renderer, eval_mesh, gt_mesh, test_extrinsics_t, alignment, intr,
                                 eval_ids, str(test_out_dir / f"test_{i:04d}_{{:03d}}.png"))
            print(f"[eval {i:04d}] {metrics}", flush=True)
            with open(out_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps({"iter": i, "loss": float(total_loss), **metrics}) + "\n")

    with torch.no_grad():
        final_mesh = geometry.get_mesh(training=False)
    export_mesh(final_mesh, out_dir / "final_mesh" / "flexicubes_opt.glb")
    export_mesh(final_mesh, out_dir / "final_mesh" / "flexicubes_opt_world.glb", alignment)
    torch.save(geometry.state(), out_dir / "final_mesh" / "flexicubes_grid.pt")
    print("FlexiCubes optimization completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimize a FlexiCubes SDF grid seeded from a recgen / SAM3D mesh"
    )
    parser.add_argument("--yaml-config", type=Path, required=True)
    parser.add_argument("--mesh-source", choices=["recgen", "sam3d"], default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--check", action="store_true",
                        help="stop after the SDF initialization (mesh + alignment sanity check)")
    args = parser.parse_args()
    run(args.yaml_config, args.mesh_source, args.out_dir, args.check)
