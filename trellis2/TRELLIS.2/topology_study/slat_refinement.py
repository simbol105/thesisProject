"""TRELLIS.2: optimize the shape SLAT and the texture SLAT jointly with a rendering loss"""
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


def main(config_path: Path, regenerate: bool):
    cfg = yaml.safe_load(open(config_path))
    device = torch.device("cuda")
    torch.manual_seed(int(cfg.get("seed", 42)))
    out_root = Path(cfg["out_root"])
    out_dir = out_root / "slat"
    (out_dir / "test").mkdir(parents=True, exist_ok=True)
    if regenerate and (out_root / "init_state.pt").exists():
        (out_root / "init_state.pt").unlink()

    data_dir = Path(cfg["data_dir"])
    data = C.load_dtc(data_dir, device)
    gt_world, gt_tm = C.load_gt_mesh(Path(cfg.get("gt_mesh", data_dir / "model.glb")), data["object_pose"], device)
    renderer = C.make_renderer(cfg, device)
    pipeline = C.load_pipeline(cfg)

    init = C.get_init_state(cfg, out_root, pipeline, data, gt_world, gt_tm, renderer, device)
    shape_slat, tex_slat, res, A = init["shape_slat"], init["tex_slat"], init["res"], init["alignment"]
    gt_mesh, extrinsics = C.to_model_frame(A, gt_world, data["extrinsics"])
    _, test_extrinsics = C.to_model_frame(A, gt_world, data["test_extrinsics"])

    def stats(norm):
        return (torch.tensor(norm['mean'], device=device)[None], torch.tensor(norm['std'], device=device)[None])
    s_mean, s_std = stats(pipeline.shape_slat_normalization)
    t_mean, t_std = stats(pipeline.tex_slat_normalization)
    zs0 = ((shape_slat.feats - s_mean) / s_std).detach()
    zt0 = ((tex_slat.feats - t_mean) / t_std).detach()
    zs, zt = nn.Parameter(zs0.clone()), nn.Parameter(zt0.clone())
    optimize_shape = bool(cfg.get("optimize_shape", True))
    optimize_texture = bool(cfg.get("optimize_texture", True))
    zs.requires_grad_(optimize_shape)
    zt.requires_grad_(optimize_texture)

    def decode(train: bool):
        h, subs = C.decode_shape_raw(pipeline, shape_slat.replace(feats=zs * s_std + s_mean), res)
        v, f = C.ovoxel_to_mesh(h.coords[:, 1:], h.feats.float(), res, train=train)
        tex = pipeline.decode_tex_slat(tex_slat.replace(feats=zt * t_std + t_mean), subs)[0]
        mesh = C.with_texture(v, f, tex.coords[:, 1:], tex.feats, torch.Size([*tex.shape, *tex.spatial_shape]),
                              res, pipeline.pbr_attr_layout)
        return mesh, h, subs

    eval_ids = np.unique(np.linspace(0, len(test_extrinsics) - 1, int(cfg.get("eval_views", 8))).round().astype(int)).tolist()
    with torch.no_grad():
        mesh0, h0, subs0 = decode(train=False)
    coords0, inter0 = h0.coords.clone(), h0.feats[:, 3:6] > 0
    with torch.no_grad():
        mesh0b, h0b, _ = decode(train=False)
    noise = C.gate_changes(coords0, inter0, h0b.coords, h0b.feats[:, 3:6] > 0, res)
    noise_topo = C.topology_stats(mesh0b.vertices, mesh0b.faces)
    del mesh0b, h0b
    print(f"[noise floor] same latent decoded twice: gates {noise} | euler {noise_topo['euler']}, "
          f"components {noise_topo['components']}")
    print(f"[noise floor] intersected logits within 0.5 of the threshold: {C.logit_margin(h0.feats[:, 3:6]) * 100:.1f}%, "
          f"subdivision logits: {C.logit_margin(torch.cat([s.feats.flatten() for s in subs0])) * 100:.1f}%")
    topo0 = C.topology_stats(mesh0.vertices, mesh0.faces)
    cd0 = C.chamfer_to_gt(mesh0, A, gt_tm)
    m0 = C.evaluate(renderer, mesh0, gt_mesh, test_extrinsics, data["intrinsics"], eval_ids,
                    str(out_dir / "test" / "init_{:03d}.png"))
    C.export_mesh(mesh0, out_dir / "mesh_init_world.ply", A)
    print(f"[init] {m0} | chamfer {cd0 * 1000:.2f}mm | {h0.feats.shape[0]} voxels @ {res} | topology {topo0}")
    metrics_path = out_dir / "metrics.jsonl"
    metrics_path.write_text("")
    C.append_jsonl(metrics_path, {"iter": -1, **m0, "chamfer_mm": cd0 * 1000, **topo0,
                                  **{f"noise_{k}": v for k, v in noise.items()},
                                  "intersected_margin": C.logit_margin(h0.feats[:, 3:6]),
                                  "subdiv_margin": C.logit_margin(torch.cat([s.feats.flatten() for s in subs0]))})

    params = [p for p in (zs, zt) if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=float(cfg["lr"]))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.get("loss_scaling", True)))
    weights = C.loss_weights(cfg)
    weight = float(cfg.get("weight", 1.0))
    latent_l2_weight = float(cfg.get("latent_l2_weight", 0.0))
    num_iters = int(cfg.get("num_iters", 300))
    frames_per_iter = int(cfg.get("frames_per_iter", 8))
    display_interval = int(cfg.get("display_interval", 25))
    eval_interval = int(cfg.get("eval_interval", 50))
    N = len(extrinsics)
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    gt_cache = {}

    for it in range(num_iters + 1):
        indices = rng.choice(N, size=min(frames_per_iter, N), replace=False)
        mesh, h, subs = decode(train=True)
        if mesh.faces.shape[0] == 0:
            print(f"[iter {it:04d}] empty mesh, skipping")
            continue
        if it == 0:
            h.feats.retain_grad()
            for s in subs:
                if s.feats.requires_grad:
                    s.feats.retain_grad()

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
        comps["latent_l2"] = latent_l2_weight * (((zs - zs0) ** 2).mean() + ((zt - zt0) ** 2).mean())
        total = sum(comps.values())

        if it == 0:
            print("[diag] per-term gradient on the latents (loss scale 1024):")
            for k, v in comps.items():
                g = C.grad_wrt(v, {"shape": zs, "tex": zt})
                print(f"[diag]   {k:>9s} value {v.item():.4e} | shape {g['shape'][0]:.3e} ({g['shape'][1] * 100:.0f}% nz)"
                      f" | tex {g['tex'][0]:.3e} ({g['tex'][1] * 100:.0f}% nz)")
            g1, g1024 = C.grad_wrt(total, {"shape": zs}, 1.0), C.grad_wrt(total, {"shape": zs}, 1024.0)
            print(f"[diag] fp16 check, shape latent non-zero grad entries: loss scale 1 -> {g1['shape'][1] * 100:.1f}%, "
                  f"1024 -> {g1024['shape'][1] * 100:.1f}%")

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)

        if it == 0 and h.feats.grad is not None:
            gh = h.feats.grad.float()
            print("[diag] gradient reaching the o-voxel channels of h (these decide the mesh):")
            for name, sl in (("dual vertices", slice(0, 3)), ("intersected (gate)", slice(3, 6)), ("split weight", slice(6, 7))):
                print(f"[diag]   {name:>18s}: norm {gh[:, sl].norm().item():.3e}, non-zero {(gh[:, sl] != 0).float().mean().item() * 100:.1f}%")
            sub_grads = [s.feats.grad for s in subs if s.feats.requires_grad]
            nz = [0.0 if g is None else (g != 0).float().mean().item() for g in sub_grads]
            print(f"[diag]   subdivision logits per level: non-zero grad "
                  + ", ".join(f"{x * 100:.1f}%" for x in nz) + " (None/0% = hard gate, not in the graph)")

        grad_norms = {n: (p.grad.norm().item() if p.grad is not None else 0.0) for n, p in (("shape", zs), ("tex", zt))}
        if float(cfg.get("grad_clip", 0.0)) > 0:
            torch.nn.utils.clip_grad_norm_(params, float(cfg["grad_clip"]))
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            drift_s, drift_t = (zs - zs0).abs().mean().item(), (zt - zt0).abs().mean().item()
        print(f"[iter {it:04d}] total {total.item():.4f} | " + " ".join(f"{k} {v.item():.4f}" for k, v in comps.items())
              + f" | IoU {np.mean(ious):.4f} | grad shape {grad_norms['shape']:.2e} tex {grad_norms['tex']:.2e}"
              + f" | drift shape {drift_s:.4f} tex {drift_t:.4f} | scale {scaler.get_scale():.0f}", flush=True)

        if display_interval and it % display_interval == 0:
            v = int(indices[0])
            C.save_image(str(out_dir / f"opt_{it:04d}_{v:03d}.png"), C.panel(renderer, mesh, gt_mesh, extrinsics[v], data["intrinsics"]))

        if eval_interval and (it % eval_interval == 0 or it == num_iters):
            with torch.no_grad():
                mesh_e, h_e, subs_e = decode(train=False)
                gates = C.gate_changes(coords0, inter0, h_e.coords, h_e.feats[:, 3:6] > 0, res)
                topo = C.topology_stats(mesh_e.vertices, mesh_e.faces)
                m = C.evaluate(renderer, mesh_e, gt_mesh, test_extrinsics, data["intrinsics"], eval_ids,
                               str(out_dir / "test" / f"test_{it:04d}_{{:03d}}.png"))
            print(f"[eval {it:04d}] {m} | gates {gates} | topology {topo}", flush=True)
            C.append_jsonl(metrics_path, {"iter": it, "loss": total.item(), **m, **gates, **topo,
                                          "intersected_margin": C.logit_margin(h_e.feats[:, 3:6]),
                                          "subdiv_margin": C.logit_margin(torch.cat([s.feats.flatten() for s in subs_e]))})

    with torch.no_grad():
        mesh_f, h_f, _ = decode(train=False)
        gates = C.gate_changes(coords0, inter0, h_f.coords, h_f.feats[:, 3:6] > 0, res)
        topo_f = C.topology_stats(mesh_f.vertices, mesh_f.faces)
    cd_f = C.chamfer_to_gt(mesh_f, A, gt_tm)
    C.export_mesh(mesh_f, out_dir / "mesh_final_world.ply", A)
    torch.save({"shape_feats": (zs * s_std + s_mean).detach().cpu(), "tex_feats": (zt * t_std + t_mean).detach().cpu(),
                "shape_coords": shape_slat.coords.cpu(), "tex_coords": tex_slat.coords.cpu(), "res": res,
                "alignment": A}, out_dir / "latents_final.pt")
    summary = {"chamfer_mm": [cd0 * 1000, cd_f * 1000], "topology_init": topo0, "topology_final": topo_f,
               "gate_changes": gates, "gate_noise_floor": noise}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[final] chamfer {cd0 * 1000:.2f}mm -> {cd_f * 1000:.2f}mm")
    print(f"[final] topology init  {topo0}\n[final] topology final {topo_f}\n[final] gate changes {gates}")
    print("SLAT refinement completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRELLIS.2 shape + texture SLAT refinement with a topology check")
    parser.add_argument("--yaml-config", type=Path, required=True)
    parser.add_argument("--regenerate", action="store_true", help="resample the initial latents and alignment")
    args = parser.parse_args()
    main(args.yaml_config, args.regenerate)
