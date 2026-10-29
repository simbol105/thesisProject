"""TRELLIS.2: optimize the o-voxel directly"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import yaml

import common as C

CHANNEL_GROUPS = {"dual_vertices": slice(0, 3), "intersected": slice(3, 6), "split_weight": slice(6, 7)}


def main(config_path: Path, regenerate: bool):
    cfg = yaml.safe_load(open(config_path))
    device = torch.device("cuda")
    torch.manual_seed(int(cfg.get("seed", 42)))
    out_root = Path(cfg["out_root"])
    out_dir = out_root / "ovoxel"
    (out_dir / "test").mkdir(parents=True, exist_ok=True)
    if regenerate and (out_root / "init_state.pt").exists():
        (out_root / "init_state.pt").unlink()

    data_dir = Path(cfg["data_dir"])
    data = C.load_dtc(data_dir, device)
    gt_world, gt_tm = C.load_gt_mesh(Path(cfg.get("gt_mesh", data_dir / "model.glb")), data["object_pose"], device)
    renderer = C.make_renderer(cfg, device)
    pipeline = C.load_pipeline(cfg)

    init = C.get_init_state(cfg, out_root, pipeline, data, gt_world, gt_tm, renderer, device)
    res, A = init["res"], init["alignment"]
    gt_mesh, extrinsics = C.to_model_frame(A, gt_world, data["extrinsics"])
    _, test_extrinsics = C.to_model_frame(A, gt_world, data["test_extrinsics"])

    with torch.no_grad():
        h0, subs0 = C.decode_shape_raw(pipeline, init["shape_slat"], res)
        tex0 = pipeline.decode_tex_slat(init["tex_slat"], subs0)[0]
    coords = h0.coords[:, 1:].contiguous()
    tex_coords, tex_shape = tex0.coords[:, 1:].contiguous(), torch.Size([*tex0.shape, *tex0.spatial_shape])
    h_init, tex_init = h0.feats.float().detach(), tex0.feats.float().detach()
    del pipeline
    torch.cuda.empty_cache()

    groups = [g for g in cfg.get("ovoxel_optimize", ["dual_vertices", "split_weight", "texture"])]
    h_param = nn.Parameter(h_init.clone())
    tex_param = nn.Parameter(tex_init.clone(), requires_grad="texture" in groups)
    h_mask = torch.zeros(1, 7, device=device)
    for g in groups:
        if g in CHANNEL_GROUPS:
            h_mask[:, CHANNEL_GROUPS[g]] = 1.0
    s = C.split_similarity(A)[0]
    print(f"[ovoxel] {coords.shape[0]} voxels @ {res}, texture voxels {tex_coords.shape[0]}, optimizing {groups}")
    print(f"[ovoxel] a dual vertex can move at most 1.5 voxels = {1.5 / res:.5f} model units "
          f"= {1.5 / res * s * 1000:.2f} mm in the world")

    def decode(train: bool):
        v, f = C.ovoxel_to_mesh(coords, h_param, res, train=train)
        return C.with_texture(v, f, tex_coords, tex_param, tex_shape, res, {
            'base_color': slice(0, 3), 'metallic': slice(3, 4), 'roughness': slice(4, 5), 'alpha': slice(5, 6)})

    eval_ids = np.unique(np.linspace(0, len(test_extrinsics) - 1, int(cfg.get("eval_views", 8))).round().astype(int)).tolist()
    with torch.no_grad():
        mesh0 = decode(train=False)
    inter0 = h_init[:, 3:6] > 0
    topo0 = C.topology_stats(mesh0.vertices, mesh0.faces)
    cd0 = C.chamfer_to_gt(mesh0, A, gt_tm)
    m0 = C.evaluate(renderer, mesh0, gt_mesh, test_extrinsics, data["intrinsics"], eval_ids,
                    str(out_dir / "test" / "init_{:03d}.png"))
    C.export_mesh(mesh0, out_dir / "mesh_init_world.ply", A)
    print(f"[init] {m0} | chamfer {cd0 * 1000:.2f}mm | topology {topo0}")
    metrics_path = out_dir / "metrics.jsonl"
    metrics_path.write_text("")
    C.append_jsonl(metrics_path, {"iter": -1, **m0, "chamfer_mm": cd0 * 1000, **topo0,
                                  "intersected_margin": C.logit_margin(h_init[:, 3:6])})

    params = [p for p in (h_param, tex_param) if p.requires_grad]
    optimizer = torch.optim.Adam([
        {"params": [h_param], "lr": float(cfg.get("lr_ovoxel", 1e-2))},
        *([{"params": [tex_param], "lr": float(cfg.get("lr_texture", 1e-2))}] if tex_param.requires_grad else []),
    ])
    weights = C.loss_weights(cfg)
    weight = float(cfg.get("weight", 1.0))
    l2_weight = float(cfg.get("ovoxel_l2_weight", 0.0))
    num_iters = int(cfg.get("num_iters", 300))
    frames_per_iter = int(cfg.get("frames_per_iter", 8))
    display_interval = int(cfg.get("display_interval", 25))
    eval_interval = int(cfg.get("eval_interval", 50))
    N = len(extrinsics)
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    gt_cache = {}

    for it in range(num_iters + 1):
        indices = rng.choice(N, size=min(frames_per_iter, N), replace=False)
        mesh = decode(train=True)
        comps = {k: torch.zeros((), device=device) for k in weights}
        ious = []
        for idx in indices:
            idx = int(idx)
            if idx not in gt_cache:
                gt_cache[idx] = C.render_gt(renderer, gt_mesh, extrinsics[idx], data["intrinsics"])
            gt = gt_cache[idx]
            pred = C.render_pred(renderer, mesh, extrinsics[idx], data["intrinsics"])
            for k, v in C.view_losses(pred, gt, cfg).items():
                comps[k] = comps[k] + weight * weights[k] * v
            with torch.no_grad():
                pm, gm = pred["mask"] > 0.5, gt["mask"] > 0.5
                ious.append(float((pm & gm).sum() / (pm | gm).sum().clamp_min(1)))
        comps = {k: v / len(indices) for k, v in comps.items()}
        comps["ovoxel_l2"] = l2_weight * (((h_param - h_init) ** 2).mean() + ((tex_param - tex_init) ** 2).mean())
        total = sum(comps.values())

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        if it == 0:
            print("[diag] gradient of the total loss per o-voxel channel group:")
            g = h_param.grad
            for name, sl in CHANNEL_GROUPS.items():
                print(f"[diag]   {name:>14s}: norm {g[:, sl].norm().item():.3e}, non-zero {(g[:, sl] != 0).float().mean().item() * 100:.1f}%"
                      + ("" if name in groups else " (not optimized)"))
            if tex_param.grad is not None:
                print(f"[diag]   {'texture':>14s}: norm {tex_param.grad.norm().item():.3e}")
        h_param.grad.mul_(h_mask)
        if float(cfg.get("grad_clip", 0.0)) > 0:
            torch.nn.utils.clip_grad_norm_(params, float(cfg["grad_clip"]))
        optimizer.step()
        with torch.no_grad():
            tex_param.clamp_(0.0, 1.0)
            drift = (h_param[:, 0:3] - h_init[:, 0:3]).abs().mean().item()

        print(f"[iter {it:04d}] total {total.item():.4f} | " + " ".join(f"{k} {v.item():.4f}" for k, v in comps.items())
              + f" | IoU {np.mean(ious):.4f} | vertex-logit drift {drift:.4f}", flush=True)

        if display_interval and it % display_interval == 0:
            v = int(indices[0])
            C.save_image(str(out_dir / f"opt_{it:04d}_{v:03d}.png"), C.panel(renderer, mesh, gt_mesh, extrinsics[v], data["intrinsics"]))

        if eval_interval and (it % eval_interval == 0 or it == num_iters):
            with torch.no_grad():
                mesh_e = decode(train=False)
                gates = C.gate_changes(coords, inter0, coords, h_param[:, 3:6] > 0, res)
                topo = C.topology_stats(mesh_e.vertices, mesh_e.faces)
                m = C.evaluate(renderer, mesh_e, gt_mesh, test_extrinsics, data["intrinsics"], eval_ids,
                               str(out_dir / "test" / f"test_{it:04d}_{{:03d}}.png"))
            print(f"[eval {it:04d}] {m} | gates {gates} | topology {topo}", flush=True)
            C.append_jsonl(metrics_path, {"iter": it, "loss": total.item(), **m, **gates, **topo})

    with torch.no_grad():
        mesh_f = decode(train=False)
        gates = C.gate_changes(coords, inter0, coords, h_param[:, 3:6] > 0, res)
        topo_f = C.topology_stats(mesh_f.vertices, mesh_f.faces)
    cd_f = C.chamfer_to_gt(mesh_f, A, gt_tm)
    C.export_mesh(mesh_f, out_dir / "mesh_final_world.ply", A)
    torch.save({"coords": coords.cpu(), "h": h_param.detach().cpu(), "tex_coords": tex_coords.cpu(),
                "tex": tex_param.detach().cpu(), "res": res, "alignment": A}, out_dir / "ovoxel_final.pt")
    summary = {"chamfer_mm": [cd0 * 1000, cd_f * 1000], "topology_init": topo0, "topology_final": topo_f,
               "gate_changes": gates}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[final] chamfer {cd0 * 1000:.2f}mm -> {cd_f * 1000:.2f}mm")
    print(f"[final] topology init  {topo0}\n[final] topology final {topo_f}\n[final] gate changes {gates}")
    print("O-voxel refinement completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRELLIS.2 o-voxel refinement with a topology check")
    parser.add_argument("--yaml-config", type=Path, required=True)
    parser.add_argument("--regenerate", action="store_true", help="resample the initial latents and alignment")
    args = parser.parse_args()
    main(args.yaml_config, args.regenerate)
