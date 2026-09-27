"""Stanford-ORB benchmark on the DTC thesis objects, for RecGen, SAM3D, nvdiffrec (+FlexiCubes) and
any other method that produces a mesh.
"""
import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import open3d as o3d
import torch
import trimesh
from scipy.spatial import cKDTree

PROJECTS = Path(__file__).resolve().parents[1]
ORB_ROOT = PROJECTS / "Stanford-ORB"
DATASET = PROJECTS / "dataset" / "DTC"
DEFAULT_OUT = PROJECTS / "benchmark_output"
GT_RES, PRED_RES = 2048, 512

OBJECTS = [
    "Birdhouse_Lighting001",
    "Cup_B08TWHJ33Q_Tan_Lighting001",
    "Figurine_B08FYFNYP4_LionKing_Lighting001",
    "Gargoyle_B08SQMBDXY_HandsOnKnees_Lighting001",
    "Mallard_B082D168CK_MintGreen_Lighting001",
    "Mouse_B0CHNVBBLF_Honeycomb_1_Lighting001",
]

NVDIFFREC_OUT = PROJECTS / "nvdiffrec" / "out" / "flexicubes"

METHODS: Dict[str, Dict] = {
    "recgen": {"mesh": str(PROJECTS / "recgen" / "thesis_output" / "{obj}" / "mesh.obj"), "frame": "align"},
    "sam3d": {"mesh": str(PROJECTS / "sam3d" / "sam-3d-objects" / "sam3d_output" / "{obj}" / "mesh.glb"), "frame": "align"},
    "trellis": {"mesh": str(PROJECTS / "trellis1" / "TRELLIS" / "trellis_output" / "{obj}" / "mesh.glb"), "frame": "align"},
    "trellis2": {"mesh": str(PROJECTS / "trellis2" / "TRELLIS.2" / "trellis2_output" / "{obj}" / "mesh.glb"), "frame": "align"},
    "nvdiffrec": {
        "mesh": str(NVDIFFREC_OUT / "{obj}" / "mesh.obj"), "frame": "align",
        "view": lambda obj, i: NVDIFFREC_OUT / obj / "validate" / f"val_{i:06d}_opt.png",
        "albedo": lambda obj, i: NVDIFFREC_OUT / obj / "validate" / f"val_{i:06d}_kd.png",
    },
}



def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((np.maximum(x, 0.04045) + 0.055) / 1.055) ** 2.4)


def write_exr(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = img.astype(np.float32)
    if img.ndim == 3:
        img = img[..., ::-1]
    assert cv2.imwrite(str(path), img), path


def write_png(path: Path, img: np.ndarray):
    """img: float in [0, 1] (H, W) or (H, W, 3) RGB."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    if img.ndim == 3:
        img = img[..., ::-1]
    assert cv2.imwrite(str(path), img), path


def load_dtc(obj: str) -> Dict:
    d = DATASET / obj
    train = json.load(open(d / "transforms_train.json"))
    test = json.load(open(d / "transforms_test.json"))
    intr = train["intrinsic"]
    w = h = 2000
    first = cv2.imread(str(d / (test["frames"][0]["file_path"] + ".png")), cv2.IMREAD_UNCHANGED)
    if first is not None:
        h, w = first.shape[:2]
    K = np.array([[intr["fx"] / w, 0, intr["cx"] / w], [0, intr["fy"] / h, intr["cy"] / h], [0, 0, 1]])
    return {
        "dir": d, "K": K, "object_pose": np.array(train["object_pose"], dtype=np.float64),
        "train": [(f["file_path"], np.array(f["transform_matrix"], dtype=np.float64)) for f in train["frames"]],
        "test": [(f["file_path"], np.array(f["transform_matrix"], dtype=np.float64)) for f in test["frames"]],
    }


def load_mesh_with_colors(path: Path):
    """(vertices [V,3] float64, faces [F,3], colors [V,3] in [0,1] or None)."""
    tm = trimesh.load(str(path), force="mesh", process=False)
    colors = None
    try:
        visual = tm.visual.to_color() if hasattr(tm.visual, "to_color") else tm.visual
        vc = np.asarray(visual.vertex_colors)
        if len(vc) == len(tm.vertices):
            colors = vc[:, :3] / 255.0
            if np.allclose(colors, colors[0]):
                colors = None
    except Exception:
        colors = None
    return np.asarray(tm.vertices, dtype=np.float64), np.asarray(tm.faces, dtype=np.int64), colors



class Renderer:
    def __init__(self, near: float = 0.01, far: float = 100.0):
        import nvdiffrast.torch as dr
        self.dr = dr
        self.ctx = dr.RasterizeCudaContext()
        self.near, self.far = near, far

    @torch.no_grad()
    def render(self, verts: torch.Tensor, faces: torch.Tensor, colors: Optional[torch.Tensor],
               E: np.ndarray, K: np.ndarray, res: int) -> Dict[str, np.ndarray]:
        """verts in the DTC world frame; E world->camera (OpenCV); K normalized intrinsics."""
        dev = verts.device
        E_t = torch.tensor(E, dtype=torch.float32, device=dev)
        vc = verts @ E_t[:3, :3].T + E_t[:3, 3]
        x, y, z = vc[:, 0], vc[:, 1], vc[:, 2]
        n, f = self.near, self.far
        clip = torch.stack([
            2 * K[0, 0] * x + (2 * K[0, 2] - 1) * z,
            2 * K[1, 1] * y + (2 * K[1, 2] - 1) * z,
            z * (f + n) / (f - n) - 2 * f * n / (f - n),
            z,
        ], dim=-1)[None].contiguous()
        tri = faces.int().contiguous()
        rast, _ = self.dr.rasterize(self.ctx, clip, tri, (res, res))
        tri_id = rast[0, ..., 3].long()
        mask = tri_id > 0

        fn = torch.cross(vc[faces[:, 1]] - vc[faces[:, 0]], vc[faces[:, 2]] - vc[faces[:, 0]], dim=-1)
        fn = torch.nn.functional.normalize(fn, dim=-1)
        fn = torch.where((fn * vc[faces[:, 0]]).sum(-1, keepdim=True) > 0, -fn, fn)
        normal = torch.zeros(res, res, 3, device=dev)
        normal[mask] = fn[tri_id[mask] - 1]

        depth = self.dr.interpolate(vc[None, :, 2:3].contiguous(), rast, tri)[0][0, ..., 0] * mask
        out = {"mask": mask.float().cpu().numpy(), "depth": depth.cpu().numpy(), "normal": normal.cpu().numpy()}
        if colors is not None:
            col = self.dr.interpolate(colors[None].contiguous(), rast, tri)[0][0] * mask[..., None]
            out["color"] = col.clamp(0, 1).cpu().numpy()
        return out


def to_torch_mesh(v: np.ndarray, f: np.ndarray, c: Optional[np.ndarray]):
    dev = torch.device("cuda")
    return (torch.tensor(v, dtype=torch.float32, device=dev), torch.tensor(f, dtype=torch.int64, device=dev),
            None if c is None else torch.tensor(c, dtype=torch.float32, device=dev))



def _rotations_24() -> List[np.ndarray]:
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
    return 0.5 * (cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean())


def align_to_world(v: np.ndarray, f: np.ndarray, gt_world: trimesh.Trimesh, dtc: Dict, renderer: Renderer,
                   n_views: int = 16, top_k: int = 4, n_points: int = 8000) -> Dict:
    """Similarity x_world = s R x + t: ICP with scale from the 24 axis-aligned rotations; the"""
    src = np.asarray(trimesh.Trimesh(v, f, process=False).sample(n_points))
    tgt = np.asarray(gt_world.sample(n_points))
    c_s, c_t = src.mean(0), tgt.mean(0)
    scale0 = np.sqrt(((tgt - c_t) ** 2).sum(1).mean() / ((src - c_s) ** 2).sum(1).mean())
    diag = np.linalg.norm(tgt.max(0) - tgt.min(0))
    src_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src))
    tgt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt))
    est = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
    crit = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    cands = []
    for k, R in enumerate(_rotations_24()):
        A = np.eye(4)
        A[:3, :3] = scale0 * R
        A[:3, 3] = c_t - scale0 * R @ c_s
        for thr in (0.1 * diag, 0.03 * diag):
            A = np.array(o3d.pipelines.registration.registration_icp(src_pcd, tgt_pcd, thr, A, est, crit).transformation)
        s = np.cbrt(np.linalg.det(A[:3, :3]))
        if 0.5 * scale0 < s < 2.0 * scale0:
            cands.append({"rot": k, "A": A, "chamfer": _chamfer(_apply(A, src), tgt)})
    cands = sorted(cands, key=lambda c: c["chamfer"])[:top_k]

    ids = np.linspace(0, len(dtc["train"]) - 1, n_views).round().astype(int)
    masks = []
    for i in ids:
        alpha = cv2.imread(str(dtc["dir"] / (dtc["train"][i][0] + ".png")), cv2.IMREAD_UNCHANGED)[..., 3]
        masks.append(cv2.resize(alpha, (256, 256), interpolation=cv2.INTER_AREA) > 127)
    for c in cands:
        vt, ft, _ = to_torch_mesh(_apply(c["A"], v), f, None)
        ious = []
        for i, m in zip(ids, masks):
            pm = renderer.render(vt, ft, None, dtc["train"][i][1], dtc["K"], 256)["mask"] > 0.5
            ious.append((pm & m).sum() / max((pm | m).sum(), 1))
        c["iou"] = float(np.mean(ious))
        print(f"    rot{c['rot']:02d}: scale {np.cbrt(np.linalg.det(c['A'][:3, :3])):.4f} "
              f"chamfer {c['chamfer'] * 1000:.2f} mm, silhouette IoU {c['iou']:.4f}")
    best = max(cands, key=lambda c: c["iou"])
    return {"A": best["A"], "iou": best["iou"], "chamfer_mm": best["chamfer"] * 1000}



def gt_dir(out_root: Path, obj: str) -> Path:
    return out_root / "_gt_orb" / obj


def gt_world_mesh(dtc: Dict):
    tm = trimesh.load_mesh(str(dtc["dir"] / "model.glb"), force="mesh")
    tm.apply_transform(dtc["object_pose"])
    visual = tm.visual.to_color()
    colors = np.asarray(visual.vertex_colors)[:, :3] / 255.0
    return tm, colors


def _todo(path: Path, overwrite: bool) -> bool:
    """True if `path` has to be (re)generated."""
    return overwrite or not path.exists()


def prepare_gt(objects: List[str], out_root: Path, renderer: Renderer, overwrite: bool):
    """Per file: existing outputs are skipped, the GT mesh is only loaded if something needs it."""
    for obj in objects:
        g = gt_dir(out_root, obj)
        dtc = load_dtc(obj)
        mesh = None

        def gt_mesh_gpu():
            nonlocal mesh
            if mesh is None:
                tm, colors = gt_world_mesh(dtc)
                mesh = to_torch_mesh(np.asarray(tm.vertices), np.asarray(tm.faces), colors)
                if _todo(g / "mesh.obj", overwrite):
                    g.mkdir(parents=True, exist_ok=True)
                    trimesh.Trimesh(tm.vertices, tm.faces, process=False).export(str(g / "mesh.obj"))
            return mesh

        if _todo(g / "mesh.obj", overwrite):
            gt_mesh_gpu()
        made = 0
        for name_path, E in dtc["test"]:
            name = Path(name_path).name
            photo_files = [g / "test" / f"{name}.exr", g / "test_mask" / f"{name}.png"]
            render_files = [g / "z_depth" / f"{name}.npy", g / "normal" / f"{name}.npy", g / "albedo" / f"{name}.png"]
            if not any(_todo(p, overwrite) for p in photo_files + render_files):
                continue
            made += 1
            photo = cv2.imread(str(dtc["dir"] / (name_path + ".png")), cv2.IMREAD_UNCHANGED)
            mask = cv2.resize(photo[..., 3], (GT_RES, GT_RES), interpolation=cv2.INTER_NEAREST) > 127
            if _todo(photo_files[0], overwrite):
                rgb = photo[..., 2::-1].astype(np.float32) / 255.0
                write_exr(photo_files[0], srgb_to_linear(cv2.resize(rgb, (GT_RES, GT_RES), interpolation=cv2.INTER_LINEAR)))
            if _todo(photo_files[1], overwrite):
                photo_files[1].parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(photo_files[1]), np.where(mask, 255, 0).astype(np.uint8))
            if any(_todo(p, overwrite) for p in render_files):
                vt, ft, ct = gt_mesh_gpu()
                r = renderer.render(vt, ft, ct, E, dtc["K"], GT_RES)
                for p in render_files[:2]:
                    p.parent.mkdir(parents=True, exist_ok=True)
                if _todo(render_files[0], overwrite):
                    np.save(render_files[0], r["depth"].astype(np.float32))
                if _todo(render_files[1], overwrite):
                    np.save(render_files[1], r["normal"].astype(np.float32))
                if _todo(render_files[2], overwrite):
                    write_png(render_files[2], r["color"])
                m_render = r["mask"] > 0.5
                iou = (m_render & mask).sum() / max((m_render | mask).sum(), 1)
                print(f"    {obj} {name}: GT-render vs photo mask IoU {iou:.4f}")
        print(f"[gt] {obj}: {made} view(s) generated, {len(dtc['test']) - made} already complete")


def resolve_methods(names: List[str], custom: Optional[Dict]) -> Dict[str, Dict]:
    methods = {}
    for n in names:
        if n not in METHODS:
            raise SystemExit(f"unknown method {n}; known: {list(METHODS)} (or use --custom-*)")
        methods[n] = METHODS[n]
    if custom:
        methods[custom["name"]] = custom
    return methods


def predict(method: str, spec: Dict, objects: List[str], out_root: Path, renderer: Renderer, overwrite: bool):
    """Per file: existing outputs are skipped; a saved alignment.npy is reused (the search is the"""
    for obj in objects:
        mesh_path = Path(spec["mesh"].format(obj=obj))
        if not mesh_path.is_absolute():
            mesh_path = PROJECTS / mesh_path
        if not mesh_path.exists():
            print(f"[{method}] {obj}: no mesh at {mesh_path}, skipped")
            continue
        o = out_root / method / obj
        dtc = load_dtc(obj)
        names = [Path(p).name for p, _ in dtc["test"]]
        per_view = {n: [o / "depth" / f"{n}.exr", o / "normal" / f"{n}.exr", o / "view" / f"{n}.png",
                        o / "albedo" / f"{n}.png"] for n in names}
        if not _todo(o / "mesh_world.obj", overwrite) and not any(_todo(p, overwrite) for ps in per_view.values() for p in ps):
            print(f"[{method}] {obj}: all files exist, skipped")
            continue
        print(f"[{method}] {obj}: {mesh_path}")
        o.mkdir(parents=True, exist_ok=True)
        v, f, c = load_mesh_with_colors(mesh_path)

        if spec.get("frame", "align") != "align":
            A = np.eye(4)
        elif not _todo(o / "alignment.npy", overwrite):
            A = np.load(o / "alignment.npy")
            print(f"    reusing {o / 'alignment.npy'}")
        else:
            gt_tm, _ = gt_world_mesh(dtc)
            res = align_to_world(v, f, trimesh.Trimesh(gt_tm.vertices, gt_tm.faces, process=False), dtc, renderer)
            A = res["A"]
            print(f"    alignment: IoU {res['iou']:.4f}, chamfer {res['chamfer_mm']:.2f} mm")
        if _todo(o / "alignment.npy", overwrite):
            np.save(o / "alignment.npy", A)
        v = _apply(A, v)
        if _todo(o / "mesh_world.obj", overwrite):
            trimesh.Trimesh(v, f, vertex_colors=None if c is None else (c * 255).astype(np.uint8),
                            process=False).export(str(o / "mesh_world.obj"))

        vt, ft, ct = to_torch_mesh(v, f, c)
        made = 0
        for i, (name_path, E) in enumerate(dtc["test"]):
            name = Path(name_path).name
            depth_p, normal_p, view_p, albedo_p = per_view[name]
            if not any(_todo(p, overwrite) for p in per_view[name]):
                continue
            made += 1
            own = {k: spec.get(k) for k in ("view", "albedo")}
            own = {k: fn(obj, i) for k, fn in own.items() if fn is not None and Path(fn(obj, i)).exists()}
            needs_render = (_todo(depth_p, overwrite) or _todo(normal_p, overwrite)
                            or (_todo(view_p, overwrite) and "view" not in own)
                            or (_todo(albedo_p, overwrite) and "albedo" not in own))
            r = renderer.render(vt, ft, ct, E, dtc["K"], PRED_RES) if needs_render else {}
            if _todo(depth_p, overwrite):
                write_exr(depth_p, r["depth"])
            if _todo(normal_p, overwrite):
                write_exr(normal_p, r["normal"])
            for kind, path in (("view", view_p), ("albedo", albedo_p)):
                if not _todo(path, overwrite):
                    continue
                if kind in own:
                    img = cv2.imread(str(own[kind]), cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255.0
                    img = cv2.resize(img, (PRED_RES, PRED_RES), interpolation=cv2.INTER_AREA)
                else:
                    img = r.get("color", np.full((PRED_RES, PRED_RES, 3), 0.5, dtype=np.float32))
                write_png(path, img)
        print(f"[{method}] {obj}: {made} view(s) generated, {len(names) - made} already complete")


def _complete(obj: str, o: Path, g: Path) -> bool:
    names = [Path(p).name for p, _ in load_dtc(obj)["test"]]
    need = [o / "mesh_world.obj", g / "mesh.obj"]
    for n in names:
        need += [o / "depth" / f"{n}.exr", o / "normal" / f"{n}.exr", o / "view" / f"{n}.png", o / "albedo" / f"{n}.png",
                 g / "test" / f"{n}.exr", g / "test_mask" / f"{n}.png", g / "z_depth" / f"{n}.npy",
                 g / "normal" / f"{n}.npy", g / "albedo" / f"{n}.png"]
    missing = [p for p in need if not p.exists()]
    if missing:
        print(f"    {obj}: {len(missing)} file(s) missing (e.g. {missing[0]}), not evaluated")
    return not missing


def results_json(method: str, objects: List[str], out_root: Path) -> Path:
    info = {}
    for obj in objects:
        o, g = out_root / method / obj, gt_dir(out_root, obj)
        if not _complete(obj, o, g):
            continue
        names = [Path(p).name for p, _ in load_dtc(obj)["test"]]
        info[obj] = {
            "view": [{"output_image": str(o / "view" / f"{n}.png"), "target_image": str(g / "test" / f"{n}.exr")} for n in names],
            "light": [{"output_image": str(o / "view" / f"{n}.png"), "target_image": str(g / "test" / f"{n}.exr")} for n in names],
            "geometry": [{"output_depth": str(o / "depth" / f"{n}.exr"), "target_depth": str(g / "z_depth" / f"{n}.npy"),
                          "output_normal": str(o / "normal" / f"{n}.exr"), "target_normal": str(g / "normal" / f"{n}.npy"),
                          "target_mask": str(g / "test_mask" / f"{n}.png")} for n in names],
            "material": [{"output_image": str(o / "albedo" / f"{n}.png"), "target_image": str(g / "albedo" / f"{n}.png"),
                          "target_mask": str(g / "test_mask" / f"{n}.png")} for n in names],
            "shape": {"output_mesh": str(o / "mesh_world.obj"), "target_mesh": str(g / "mesh.obj")},
        }
    path = out_root / method / "orb_input.json"
    path.write_text(json.dumps({"info": info}, indent=2))
    return path


def evaluate(method: str, objects: List[str], out_root: Path) -> Optional[Dict]:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "shim"))
    sys.path.insert(0, str(ORB_ROOT))
    from orb.utils.test import (compute_metrics_image_similarity, compute_metrics_geometry,
                                compute_metrics_material, compute_metrics_shape)

    info = json.loads(results_json(method, objects, out_root).read_text())["info"]
    if not info:
        print(f"[{method}] nothing to evaluate")
        return None
    scores = {}
    for obj, r in info.items():
        print(f"[{method}] evaluating {obj}")
        scores[obj] = {
            "view": compute_metrics_image_similarity(r["view"], scale_invariant=False),
            "light": compute_metrics_image_similarity(r["light"], scale_invariant=True),
            "geometry": compute_metrics_geometry(r["geometry"]),
            "material": compute_metrics_material(r["material"]),
            "shape": compute_metrics_shape(r["shape"]),
        }
    mean = {}
    for group in ("view", "light", "geometry", "material", "shape"):
        keys = sorted({k for s in scores.values() for k in s[group]})
        mean[group] = {k: [float(np.mean([s[group][k] for s in scores.values() if k in s[group]])),
                           float(np.std([s[group][k] for s in scores.values() if k in s[group]]))] for k in keys}
    out = {"method": method, "objects": list(scores), "scores_stats": mean, "scores": scores,
           "note": "ORB metrics; bidir_chamfer is the squared-distance chamfer in m^2 as in ORB's json "
                   "(ORB paper tables report 2e3 x this value). 'light' = the view renders scored "
                   "scale-invariantly (no per-view environment maps in DTC)."}
    (out_root / method / "scores.json").write_text(json.dumps(out, indent=2))
    print(f"[{method}] wrote {out_root / method / 'scores.json'}")
    return out


def write_summary(out_root: Path):
    rows = []
    for p in sorted(out_root.glob("*/scores.json")):
        s = json.loads(p.read_text())
        st = s["scores_stats"]
        get = lambda g, k: st.get(g, {}).get(k, [float("nan")])[0]
        rows.append((s["method"], len(s["objects"]), get("view", "psnr_hdr"), get("view", "psnr_ldr"), get("view", "ssim"),
                     get("view", "lpips"), get("geometry", "normal_angle"), get("geometry", "depth_mse_scene"),
                     get("light", "psnr_hdr"), get("light", "psnr_ldr"), get("light", "ssim"), get("light", "lpips"),
                     get("shape", "bidir_chamfer") * 2e3, get("material", "psnr_ldr")))
    header = ("| method | objects | view PSNR-H | view PSNR-L | view SSIM | view LPIPS | normal (1-cos) | depth SI-MSE "
              "| light PSNR-H | light PSNR-L | light SSIM | light LPIPS | chamfer (x2e3, ORB paper scale) | albedo PSNR |")
    lines = [header, "|" + "---|" * 14]
    lines += ["| " + " | ".join([r[0], str(r[1])] + [f"{x:.4f}" for x in r[2:]]) + " |" for r in rows]
    (out_root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Stanford-ORB benchmark on the DTC thesis objects")
    parser.add_argument("step", choices=["prepare-gt", "predict", "evaluate", "all", "summary"])
    parser.add_argument("--methods", nargs="*", default=[], help=f"built-in methods: {list(METHODS)}")
    parser.add_argument("--objects", nargs="*", default=OBJECTS)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--custom-method", default=None, help="name of an extra method (output folder name)")
    parser.add_argument("--custom-mesh", default=None, help="mesh path pattern with {obj}")
    parser.add_argument("--custom-frame", choices=["world", "align"], default="world")
    args = parser.parse_args()

    custom = None
    if args.custom_method:
        assert args.custom_mesh, "--custom-mesh is required with --custom-method"
        custom = {"name": args.custom_method, "mesh": args.custom_mesh, "frame": args.custom_frame}
    methods = resolve_methods(args.methods, custom)
    args.out_root.mkdir(parents=True, exist_ok=True)

    renderer = Renderer() if args.step in ("prepare-gt", "predict", "all") else None
    if args.step in ("prepare-gt", "all"):
        prepare_gt(args.objects, args.out_root, renderer, args.overwrite)
    if args.step in ("predict", "all"):
        for name, spec in methods.items():
            predict(name, spec, args.objects, args.out_root, renderer, args.overwrite)
    if args.step in ("evaluate", "all"):
        for name in methods:
            evaluate(name, args.objects, args.out_root)
    if args.step in ("evaluate", "all", "summary"):
        write_summary(args.out_root)


if __name__ == "__main__":
    main()
