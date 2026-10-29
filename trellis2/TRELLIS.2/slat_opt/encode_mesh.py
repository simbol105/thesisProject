"""Encode any mesh (RecGen / SAM3D / TRELLIS / TRELLIS.2 output) into TRELLIS.2 shape + texture SLATs"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "topology_study"))

import argparse
import json
import time

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

import common as C
import o_voxel
import trellis2.models as models
from trellis2.modules.sparse import SparseTensor

SHAPE_ENC = "microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16"
TEX_ENC = "microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16"
AABB = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]


def load_normalized(path: Path):
    """trimesh (single geometry, visuals kept) refit to [-0.5, 0.5]^3 like data_toolkit/dual_grid.py."""
    tm = trimesh.load(str(path), force="mesh", process=False)
    v = np.asarray(tm.vertices, dtype=np.float64)
    lo, hi = v.min(0), v.max(0)
    tm.vertices = (v - (lo + hi) / 2) * (0.99999 / (hi - lo).max())
    return tm


def sample_colors(tm: trimesh.Trimesh, n: int, seed: int = 0):
    """n surface points with their base color in [0, 1]: texture (uv) or vertex colors."""
    rng = np.random.default_rng(seed)
    points, faces = trimesh.sample.sample_surface(tm, n, seed=int(rng.integers(1 << 31)))
    tri = tm.triangles[faces]
    bary = trimesh.triangles.points_to_barycentric(tri, points)
    visual = tm.visual
    colors = None
    if isinstance(visual, trimesh.visual.TextureVisuals) and visual.uv is not None:
        mat = visual.material
        image = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
        if image is not None:
            uv = (visual.uv[tm.faces[faces]] * bary[..., None]).sum(1)
            colors = trimesh.visual.color.uv_to_color(uv, image)[:, :3] / 255.0
            factor = getattr(mat, "baseColorFactor", None)
            if factor is not None:
                colors = colors * (np.asarray(factor[:3], dtype=np.float64) / 255.0)
    if colors is None:
        try:
            vc = np.asarray(visual.to_color().vertex_colors if hasattr(visual, "to_color") else visual.vertex_colors)
            colors = (vc[tm.faces[faces]][..., :3] / 255.0 * bary[..., None]).sum(1)
        except Exception:
            colors = np.full((n, 3), 0.8)
    return points, np.clip(colors, 0, 1)


def mesh_to_ovoxel(tm: trimesh.Trimesh, res: int):
    """Flexible dual grid, packed exactly like data_toolkit/dual_grid.py (uint8 dual vertices)."""
    vertices = torch.from_numpy(np.asarray(tm.vertices)).float()
    faces = torch.from_numpy(np.asarray(tm.faces)).long()
    idx, dual, inter = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices=vertices, faces=faces, grid_size=res, aabb=AABB,
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2, timing=False)
    dual = torch.clamp(dual * res - idx, 0, 1)
    dual = (dual * 255).type(torch.uint8).float() / 255.0
    key = (idx[:, 0].long() * res + idx[:, 1].long()) * res + idx[:, 2].long()
    order = torch.argsort(key)
    return idx[order].int(), dual[order], inter[order].bool()


def pbr_voxels(tm: trimesh.Trimesh, coords: torch.Tensor, res: int, samples_per_voxel: int = 8):
    """[N, 6] PBR attributes (base color, metallic, roughness, alpha) in [0, 1] on the given voxels."""
    n = int(min(max(samples_per_voxel * len(coords), 1_000_000), 20_000_000))
    points, colors = sample_colors(tm, n)
    vidx = np.clip(np.floor((points + 0.5) * res).astype(np.int64), 0, res - 1)
    skeys = (vidx[:, 0] * res + vidx[:, 1]) * res + vidx[:, 2]
    c = coords.long().numpy()
    vkeys = (c[:, 0] * res + c[:, 1]) * res + c[:, 2]
    order = np.argsort(vkeys)
    pos = np.clip(np.searchsorted(vkeys[order], skeys), 0, len(vkeys) - 1)
    hit = vkeys[order][pos] == skeys
    target = order[pos[hit]]
    base = np.zeros((len(c), 3))
    count = np.zeros(len(c))
    np.add.at(base, target, colors[hit])
    np.add.at(count, target, 1)
    empty = count == 0
    base[~empty] /= count[~empty, None]
    if empty.any():
        centers = (c[empty] + 0.5) / res - 0.5
        _, nn = cKDTree(points).query(centers)
        base[empty] = colors[nn]
    attrs = np.concatenate([base, np.zeros((len(c), 1)), np.ones((len(c), 1)), np.ones((len(c), 1))], 1)
    return torch.from_numpy(attrs).float(), float(empty.mean())


@torch.no_grad()
def encode_mesh(mesh_path: Path, res: int, device="cuda"):
    t0 = time.time()
    tm = load_normalized(mesh_path)
    idx, dual, inter = mesh_to_ovoxel(tm, res)
    attrs, empty_frac = pbr_voxels(tm, idx, res)
    coords = torch.cat([torch.zeros_like(idx[:, :1]), idx], 1).int()
    print(f"[encode] {mesh_path}: {len(tm.faces)} faces -> {len(idx)} o-voxels @ {res} "
          f"({empty_frac * 100:.1f}% of voxels colored by nearest sample)")

    shape_enc = models.from_pretrained(SHAPE_ENC).eval().to(device)
    vertices = SparseTensor(feats=dual.to(device), coords=coords.to(device))
    shape = shape_enc(vertices, vertices.replace(inter.to(device)))
    del shape_enc
    tex_enc = models.from_pretrained(TEX_ENC).eval().to(device)
    tex = tex_enc(SparseTensor(feats=(attrs * 2 - 1).to(device), coords=coords.to(device)))
    del tex_enc
    torch.cuda.empty_cache()

    if not torch.equal(shape.coords, tex.coords):
        ks = lambda c: ((c[:, 1].long() * 4096 + c[:, 2].long()) * 4096 + c[:, 3].long())
        s_order, t_order = torch.argsort(ks(shape.coords)), torch.argsort(ks(tex.coords))
        assert torch.equal(ks(shape.coords)[s_order], ks(tex.coords)[t_order]), "shape / texture SLAT coords differ"
        feats = torch.empty_like(tex.feats)
        feats[s_order] = tex.feats[t_order]
        tex = shape.replace(feats)
    for name, z in (("shape", shape), ("texture", tex)):
        assert torch.isfinite(z.feats).all(), f"non-finite {name} latent"
    print(f"[encode] shape SLAT {tuple(shape.feats.shape)}, texture SLAT {tuple(tex.feats.shape)} "
          f"@ {res // 16}^3 in {time.time() - t0:.1f}s")
    state = {"shape_coords": shape.coords.cpu(), "shape_feats": shape.feats.float().cpu(),
             "tex_coords": tex.coords.cpu(), "tex_feats": tex.feats.float().cpu(), "res": int(res),
             "source_mesh": str(mesh_path), "num_ovoxels": int(len(idx))}
    return state, tm


@torch.no_grad()
def roundtrip_check(state: dict, tm: trimesh.Trimesh, out_dir: Path, cfg: dict):
    """Decode the encoded SLATs and compare with the (normalized) source mesh: the encoding cost."""
    pipeline = C.load_pipeline(cfg)
    shape = SparseTensor(feats=state["shape_feats"].cuda(), coords=state["shape_coords"].cuda())
    tex = SparseTensor(feats=state["tex_feats"].cuda(), coords=state["tex_coords"].cuda())
    res = state["res"]
    h, subs = C.decode_shape_raw(pipeline, shape, res)
    v, f = C.ovoxel_to_mesh(h.coords[:, 1:], h.feats.float(), res, train=False)
    t = pipeline.decode_tex_slat(tex, subs)[0]
    mesh = C.with_texture(v, f, t.coords[:, 1:], t.feats, torch.Size([*t.shape, *t.spatial_shape]), res,
                          pipeline.pbr_attr_layout)
    out_dir.mkdir(parents=True, exist_ok=True)
    C.export_mesh(mesh, out_dir / "decoded_normalized.ply")
    tm.export(str(out_dir / "source_normalized.ply"))
    cd = C.chamfer(np.asarray(C.to_trimesh(mesh).sample(50000)), np.asarray(tm.sample(50000)))
    stats = {"roundtrip_chamfer_normalized": cd, "ovoxels": state["num_ovoxels"],
             "slat_tokens": int(shape.feats.shape[0]), "decoded_faces": int(f.shape[0])}
    (out_dir / "roundtrip.json").write_text(json.dumps(stats, indent=2))
    print(f"[check] round trip source -> SLAT -> mesh: chamfer {cd:.5f} (normalized units, object size 1) | {stats}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Encode a mesh into TRELLIS.2 shape + texture SLATs")
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True, help="init_state.pt is written here")
    parser.add_argument("--res", type=int, default=512)
    parser.add_argument("--check", action="store_true", help="also decode and report the round-trip error")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    path = args.out_root / "init_state.pt"
    if path.exists() and not args.overwrite:
        print(f"[encode] {path} exists, skipped")
        return
    state, tm = encode_mesh(args.mesh, args.res)
    args.out_root.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"[encode] wrote {path}")
    if args.check:
        roundtrip_check(state, tm, args.out_root / "encode_check", {})


if __name__ == "__main__":
    main()
