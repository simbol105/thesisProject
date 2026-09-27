"""Shared pieces for the TRELLIS.2 topology check (slat_refinement.py / ovoxel_refinement.py)."""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import itertools
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import scipy.sparse
import scipy.sparse.csgraph
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder
from trellis2.renderers.mesh_renderer import MeshRenderer
from trellis2.representations import Mesh, MeshWithVoxel
from trellis2.utils.loss_utils import l1_loss, ssim, lpips
from o_voxel.convert import flexible_dual_grid_to_mesh
from opt_utils.utility import save_image

AABB = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
RENDER_RES = 512


def _patch_flex_gemm_frozen_weights() -> None:
    """flex_gemm's Triton sparse-conv backward kernels return grad_weight=None when the conv weight"""
    import flex_gemm.kernels.triton as triton_kernels
    for name in dir(triton_kernels):
        if not name.startswith("sparse_submanifold_conv_bwd"):
            continue
        fn = getattr(triton_kernels, name)
        if not callable(fn) or getattr(fn, "_frozen_weight_patch", False):
            continue

        def wrapped(*args, _fn=fn, **kwargs):
            grad_input, grad_weight, grad_bias = _fn(*args, **kwargs)
            if grad_weight is None:
                grad_weight = torch.zeros_like(args[2] if len(args) > 2 else kwargs["weight"])
            return grad_input, grad_weight, grad_bias
        wrapped._frozen_weight_patch = True
        setattr(triton_kernels, name, wrapped)


_patch_flex_gemm_frozen_weights()



def _try_image_path(base_path_no_ext: Path) -> Path:
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        cand = base_path_no_ext.with_suffix(ext)
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Could not find image for: {base_path_no_ext}")


def load_dtc(data_dir: Path, device) -> Dict:
    """Cameras (world -> camera, OpenCV), normalized intrinsics and image paths of a DTC object."""
    train = json.load(open(data_dir / "transforms_train.json"))
    test = json.load(open(data_dir / "transforms_test.json"))
    image_paths = [_try_image_path(data_dir / f["file_path"]) for f in train["frames"]]
    w, h = Image.open(image_paths[0]).size
    intr = train["intrinsic"]
    intrinsics = torch.tensor([[intr["fx"] / w, 0.0, intr["cx"] / w],
                               [0.0, intr["fy"] / h, intr["cy"] / h],
                               [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
    return {
        "image_paths": image_paths,
        "intrinsics": intrinsics,
        "extrinsics": torch.tensor([f["transform_matrix"] for f in train["frames"]], dtype=torch.float32, device=device),
        "test_extrinsics": torch.tensor([f["transform_matrix"] for f in test["frames"]], dtype=torch.float32, device=device),
        "object_pose": np.array(train["object_pose"], dtype=np.float64),
    }


def load_gt_mesh(path: Path, object_pose: np.ndarray, device) -> Tuple[Mesh, trimesh.Trimesh]:
    """GT mesh in the DTC world frame, with its texture baked to vertex colors."""
    tm = trimesh.load_mesh(str(path), force="mesh")
    tm.apply_transform(object_pose)
    tm.visual = tm.visual.to_color()
    colors = np.asarray(tm.visual.vertex_colors)[:, :3] / 255.0
    mesh = Mesh(
        vertices=torch.from_numpy(np.asarray(tm.vertices)).float().to(device),
        faces=torch.from_numpy(np.asarray(tm.faces)).int().to(device),
        vertex_attrs=torch.from_numpy(colors).float().to(device),
    )
    return mesh, trimesh.Trimesh(vertices=tm.vertices, faces=tm.faces, process=False)


def make_renderer(cfg: Dict, device) -> MeshRenderer:
    return MeshRenderer(rendering_options={
        "resolution": RENDER_RES, "near": float(cfg.get("near", 0.01)),
        "far": float(cfg.get("far", 100.0)), "ssaa": 1,
    }, device=device.type)



def _proper_rotations_24() -> List[np.ndarray]:
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1.0, -1.0], repeat=3):
            R = np.zeros((3, 3))
            R[range(3), perm] = signs
            if np.linalg.det(R) > 0:
                mats.append(R)
    return mats


def apply_similarity(A: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ A[:3, :3].T + A[:3, 3]


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * (cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean())


def split_similarity(A: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    s = float(np.cbrt(np.linalg.det(A[:3, :3])))
    return s, A[:3, :3] / s, A[:3, 3]


def to_trimesh(mesh) -> trimesh.Trimesh:
    return trimesh.Trimesh(vertices=mesh.vertices.detach().float().cpu().numpy(),
                           faces=mesh.faces.detach().long().cpu().numpy(), process=False)


@torch.no_grad()
def silhouette_iou(renderer, mesh_a, mesh_b, extrinsics, intrinsics) -> float:
    ious = []
    for E in extrinsics:
        ma = renderer.render(mesh_a, E, intrinsics, ["mask"])["mask"] > 0.5
        mb = renderer.render(mesh_b, E, intrinsics, ["mask"])["mask"] > 0.5
        ious.append(((ma & mb).sum() / (ma | mb).sum().clamp_min(1)).float())
    return torch.stack(ious).mean().item()


def search_alignment(pred_mesh: Mesh, gt_mesh: Mesh, gt_tm: trimesh.Trimesh, renderer, extrinsics, intrinsics,
                     n_views: int = 16, top_k: int = 4, n_points: int = 8000) -> np.ndarray:
    """ICP with scale from each of the 24 axis-aligned rotations; best silhouette IoU wins."""
    src = np.asarray(to_trimesh(pred_mesh).sample(n_points))
    tgt = np.asarray(gt_tm.sample(n_points))
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
        A = np.array(A)
        if 0.5 * scale0 < split_similarity(A)[0] < 2.0 * scale0:
            candidates.append({"name": f"rot{k:02d}", "A": A, "chamfer": chamfer(apply_similarity(A, src), tgt)})
    candidates = sorted(candidates, key=lambda c: c["chamfer"])[:top_k]

    view_ids = np.linspace(0, len(extrinsics) - 1, n_views).round().astype(int)
    print("[align] candidates (TRELLIS.2 -> world):")
    for c in candidates:
        A_t = torch.tensor(c["A"], dtype=torch.float32, device=pred_mesh.vertices.device)
        moved = Mesh(pred_mesh.vertices @ A_t[:3, :3].T + A_t[:3, 3], pred_mesh.faces)
        c["iou"] = silhouette_iou(renderer, moved, gt_mesh, [extrinsics[i] for i in view_ids], intrinsics)
        print(f"  {c['name']}  scale={split_similarity(c['A'])[0]:.4f}  chamfer={c['chamfer'] * 1000:.2f}mm  IoU={c['iou']:.4f}")
    best = max(candidates, key=lambda c: c["iou"])
    print(f"[align] selected {best['name']} (IoU={best['iou']:.4f})")
    return best["A"]


def to_model_frame(A: np.ndarray, gt_world: Mesh, extrinsics_world: torch.Tensor) -> Tuple[Mesh, torch.Tensor]:
    """GT mesh and rigid cameras in the TRELLIS.2 frame."""
    s, R, t = split_similarity(A)
    device = extrinsics_world.device
    R_t = torch.tensor(R, dtype=torch.float32, device=device)
    t_t = torch.tensor(t, dtype=torch.float32, device=device)
    gt = Mesh(((gt_world.vertices - t_t) @ R_t) / s, gt_world.faces, gt_world.vertex_attrs)
    extr = extrinsics_world @ torch.tensor(A, dtype=torch.float32, device=device)
    extr[..., :3, :] = extr[..., :3, :] / s
    return gt, extr



def decode_shape_raw(pipeline, shape_slat, resolution: int):
    """Raw decoder output h [N, 7] @ `resolution` and the (hard) subdivision logits per level."""
    decoder = pipeline.models['shape_slat_decoder']
    decoder.set_resolution(resolution)
    h, subs = SparseUnetVaeDecoder.forward(decoder, shape_slat, return_subs=True)
    return h, subs


def ovoxel_to_mesh(coords: torch.Tensor, h_feats: torch.Tensor, resolution: int, voxel_margin: float = 0.5,
                   train: bool = True):
    """o-voxel -> mesh, exactly the FlexiDualGridVaeDecoder mapping."""
    vertices = (1 + 2 * voxel_margin) * torch.sigmoid(h_feats[:, 0:3]) - voxel_margin
    intersected = h_feats[:, 3:6] > 0
    split_weight = F.softplus(h_feats[:, 6:7])
    return flexible_dual_grid_to_mesh(coords, vertices, intersected, split_weight,
                                      aabb=AABB, grid_size=resolution, train=train)


def with_texture(vertices, faces, tex_coords, tex_attrs, tex_shape, resolution, layout) -> MeshWithVoxel:
    return MeshWithVoxel(vertices, faces, origin=[-0.5, -0.5, -0.5], voxel_size=1.0 / resolution,
                         coords=tex_coords, attrs=tex_attrs, voxel_shape=tex_shape, layout=layout)



def _perceptual(a: torch.Tensor, b: torch.Tensor, lambda_ssim: float, lambda_lpips: float) -> torch.Tensor:
    a, b = a[None], b[None]
    return l1_loss(a, b) + lambda_ssim * (1.0 - ssim(a, b)) + lambda_lpips * lpips(a, b)


def view_losses(pred: Dict, gt: Dict, cfg: Dict) -> Dict[str, torch.Tensor]:
    """Unweighted loss terms for one view. pred/gt: mask [H,W], depth [H,W], normal [3,H,W], color [3,H,W]."""
    m = gt["mask"]
    return {
        "mask": l1_loss(pred["mask"], m),
        "depth": l1_loss(pred["depth"] * m, gt["depth"] * m),
        "normal": _perceptual(pred["normal"] * m, gt["normal"] * m,
                              float(cfg.get("lambda_ssim", 0.2)), float(cfg.get("lambda_lpips", 0.2))),
        "color": _perceptual(pred["color"] * m, gt["color"] * m,
                             float(cfg.get("lambda_ssim", 0.2)), float(cfg.get("lambda_lpips", 0.2))),
    }


def loss_weights(cfg: Dict) -> Dict[str, float]:
    return {k: float(cfg.get(f"lambda_{k}", d)) for k, d in
            {"mask": 1.0, "depth": 1.2, "normal": 1.0, "color": 0.9}.items()}


def render_pred(renderer, mesh: MeshWithVoxel, extr, intr) -> Dict:
    out = renderer.render(mesh, extr, intr, ["attr", "mask", "normal", "depth"])
    color = out["base_color"] if "base_color" in out else out["attr"][:3]
    return {"mask": out["mask"], "depth": out["depth"], "normal": out["normal"], "color": color}


@torch.no_grad()
def render_gt(renderer, gt_mesh: Mesh, extr, intr) -> Dict:
    out = renderer.render(gt_mesh, extr, intr, ["attr", "mask", "normal", "depth"])
    return {"mask": out["mask"], "depth": out["depth"], "normal": out["normal"], "color": out["attr"][:3]}


def grad_wrt(loss: torch.Tensor, tensors: Dict[str, torch.Tensor], scale: float = 1024.0) -> Dict[str, Tuple[float, float]]:
    """{name: (grad norm, fraction of non-zero entries)} of `loss` w.r.t. each tensor."""
    names = [k for k, t in tensors.items() if t.requires_grad]
    if not loss.requires_grad or not names:
        return {k: (0.0, 0.0) for k in tensors}
    grads = torch.autograd.grad(loss * scale, [tensors[k] for k in names], retain_graph=True, allow_unused=True)
    out = {k: (0.0, 0.0) for k in tensors}
    for k, g in zip(names, grads):
        if g is not None:
            g = g.float() / scale
            out[k] = (g.norm().item(), (g != 0).float().mean().item())
    return out



def topology_stats(vertices: torch.Tensor, faces: torch.Tensor) -> Dict[str, float]:
    """Topology of a triangle mesh after welding coincident vertices."""
    v = vertices.detach().float().cpu().numpy()
    f = faces.detach().long().cpu().numpy()
    if len(f) == 0:
        return {"V": 0, "E": 0, "F": 0, "euler": 0, "components": 0, "boundary_edges": 0,
                "nonmanifold_edges": 0, "genus_closed": 0.0}
    _, inverse = np.unique(np.round(v, 7), axis=0, return_inverse=True)
    f = inverse.reshape(-1)[f]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    used = np.unique(f)
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    uniq_edges, counts = np.unique(edges, axis=0, return_counts=True)
    V, E, Fn = len(used), len(uniq_edges), len(f)
    n = int(used.max()) + 1
    adj = scipy.sparse.coo_matrix((np.ones(len(uniq_edges)), (uniq_edges[:, 0], uniq_edges[:, 1])), shape=(n, n))
    n_comp, labels = scipy.sparse.csgraph.connected_components(adj, directed=False)
    C = len(np.unique(labels[used]))
    euler = V - E + Fn
    return {"V": V, "E": E, "F": Fn, "euler": euler, "components": C,
            "boundary_edges": int((counts == 1).sum()), "nonmanifold_edges": int((counts > 2).sum()),
            "genus_closed": (2 * C - euler) / 2}


def _coord_keys(coords: torch.Tensor, res: int) -> torch.Tensor:
    c = coords[:, -3:].long()
    return (c[:, 0] * res + c[:, 1]) * res + c[:, 2]


@torch.no_grad()
def gate_changes(coords0, inter0, coords1, inter1, res: int) -> Dict[str, int]:
    """How many discrete decisions flipped between two o-voxels."""
    k0, k1 = _coord_keys(coords0, res), _coord_keys(coords1, res)
    common, i0, i1 = np.intersect1d(k0.cpu().numpy(), k1.cpu().numpy(), return_indices=True)
    flips = (inter0[torch.from_numpy(i0).to(inter0.device)] != inter1[torch.from_numpy(i1).to(inter1.device)]).sum().item()
    return {"voxels_added": int(len(k1) - len(common)), "voxels_removed": int(len(k0) - len(common)),
            "intersected_flips": int(flips), "active_quads": int(inter1.sum().item())}


@torch.no_grad()
def logit_margin(logits: torch.Tensor, band: float = 0.5) -> float:
    """Fraction of gate logits within `band` of the threshold (how close gates are to flipping)."""
    return (logits.abs() < band).float().mean().item()



@torch.no_grad()
def panel(renderer, mesh: MeshWithVoxel, gt_mesh: Mesh, extr, intr) -> np.ndarray:
    """pred color | gt color | pred normal | gt normal | overlay (red = pred only, green = gt only)."""
    p, g = render_pred(renderer, mesh, extr, intr), render_gt(renderer, gt_mesh, extr, intr)
    overlay = torch.stack([p["mask"], g["mask"], torch.zeros_like(g["mask"])])
    vis = torch.cat([p["color"] * p["mask"], g["color"] * g["mask"], p["normal"], g["normal"], overlay], dim=2)
    return vis.permute(1, 2, 0).clamp(0, 1).cpu().numpy()


@torch.no_grad()
def evaluate(renderer, mesh, gt_mesh, extrinsics, intr, view_ids, out_fmt: Optional[str] = None) -> Dict[str, float]:
    psnrs, ious = [], []
    for v in view_ids:
        p, g = render_pred(renderer, mesh, extrinsics[v], intr), render_gt(renderer, gt_mesh, extrinsics[v], intr)
        pc, gc = (p["color"] * p["mask"]).clamp(0, 1), g["color"] * g["mask"]
        psnrs.append(float(-10 * torch.log10(F.mse_loss(pc, gc).clamp_min(1e-10))))
        pm, gm = p["mask"] > 0.5, g["mask"] > 0.5
        ious.append(float((pm & gm).sum() / (pm | gm).sum().clamp_min(1)))
        if out_fmt is not None:
            save_image(out_fmt.format(v), panel(renderer, mesh, gt_mesh, extrinsics[v], intr))
    return {"psnr": float(np.mean(psnrs)), "mask_iou": float(np.mean(ious))}


def chamfer_to_gt(mesh, A: np.ndarray, gt_tm: trimesh.Trimesh, n: int = 50000) -> float:
    tm = to_trimesh(mesh)
    tm.vertices = apply_similarity(A, np.asarray(tm.vertices))
    return chamfer(np.asarray(tm.sample(n)), np.asarray(gt_tm.sample(n)))


@torch.no_grad()
def voxel_vertex_colors(mesh) -> Optional[np.ndarray]:
    """Base color of a MeshWithVoxel at its vertices (value of the voxel each vertex lies in; vertices"""
    if not isinstance(mesh, MeshWithVoxel) or mesh.attrs is None:
        return None
    res = int(round(1.0 / float(mesh.voxel_size)))
    coords = mesh.coords[:, -3:].long()
    keys = (coords[:, 0] * res + coords[:, 1]) * res + coords[:, 2]
    order = torch.argsort(keys)
    keys = keys[order]
    idx = ((mesh.vertices.detach().float() + 0.5) * res).floor().long().clamp(0, res - 1)
    vkeys = (idx[:, 0] * res + idx[:, 1]) * res + idx[:, 2]
    pos = torch.searchsorted(keys, vkeys).clamp(max=len(keys) - 1)
    hit = keys[pos] == vkeys
    voxel = order[pos]
    if (~hit).any():
        centers = ((coords.float() + 0.5) / res - 0.5).cpu().numpy()
        _, nn = cKDTree(centers).query(mesh.vertices[~hit].detach().float().cpu().numpy())
        voxel[~hit] = torch.as_tensor(nn, device=voxel.device)
    base = mesh.attrs[voxel][:, mesh.layout.get("base_color", slice(0, 3))].float().clamp(0, 1)
    return (base.cpu().numpy() * 255 + 0.5).astype(np.uint8)


def export_mesh(mesh, path: Path, A: Optional[np.ndarray] = None):
    tm = to_trimesh(mesh)
    colors = voxel_vertex_colors(mesh)
    if colors is not None:
        tm.visual = trimesh.visual.ColorVisuals(tm, vertex_colors=colors)
    if A is not None:
        tm.vertices = apply_similarity(A, np.asarray(tm.vertices))
    path.parent.mkdir(parents=True, exist_ok=True)
    tm.export(str(path))
    print(f"[io] wrote {path}")


def append_jsonl(path: Path, record: Dict):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")



def load_pipeline(cfg: Dict):
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    print("Loading TRELLIS.2 pipeline...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(cfg.get("model_name", "microsoft/TRELLIS.2-4B")))
    pipeline.low_vram = False
    pipeline.cuda()
    for module in pipeline.models.values():
        if isinstance(module, torch.nn.Module):
            module.eval()
            module.requires_grad_(False)
    return pipeline


def get_init_state(cfg: Dict, out_root: Path, pipeline, data: Dict, gt_world: Mesh, gt_tm: trimesh.Trimesh,
                   renderer, device) -> Dict:
    """Latents from pipeline.run + the alignment, cached in <out_root>/init_state.pt."""
    from trellis2.modules.sparse import SparseTensor
    path = out_root / "init_state.pt"
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=False)
        print(f"[init] loaded {path}")
    else:
        prompt_idx = int(cfg.get("prompt_idx", 0))
        print(f"[init] pipeline.run on {data['image_paths'][prompt_idx]} (pipeline_type={cfg.get('pipeline_type', '512')})")
        _, (shape_slat, tex_slat, res) = pipeline.run(
            Image.open(data["image_paths"][prompt_idx]).convert("RGBA"),
            seed=int(cfg.get("seed", 42)), return_latent=True,
            pipeline_type=str(cfg.get("pipeline_type", "512")),
        )
        state = {
            "shape_coords": shape_slat.coords.cpu(), "shape_feats": shape_slat.feats.cpu(),
            "tex_coords": tex_slat.coords.cpu(), "tex_feats": tex_slat.feats.cpu(), "res": int(res),
        }
    shape_slat = SparseTensor(feats=state["shape_feats"].to(device), coords=state["shape_coords"].to(device))
    tex_slat = SparseTensor(feats=state["tex_feats"].to(device), coords=state["tex_coords"].to(device))
    res = int(state["res"])

    if "alignment" not in state:
        with torch.no_grad():
            h, _ = decode_shape_raw(pipeline, shape_slat, res)
            v, f = ovoxel_to_mesh(h.coords[:, 1:], h.feats.float(), res, train=False)
        path_a = cfg.get("alignment_path", "")
        if path_a and Path(path_a).exists():
            state["alignment"] = np.array(np.load(path_a), dtype=np.float64)
        else:
            state["alignment"] = search_alignment(Mesh(v, f), gt_world, gt_tm, renderer, data["extrinsics"],
                                                  data["intrinsics"], int(cfg.get("align_iou_views", 16)),
                                                  int(cfg.get("align_top_k", 4)))
    out_root.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"[init] shape SLAT {tuple(shape_slat.feats.shape)}, tex SLAT {tuple(tex_slat.feats.shape)}, "
          f"res {res}, alignment scale {split_similarity(state['alignment'])[0]:.4f}")
    return {"shape_slat": shape_slat, "tex_slat": tex_slat, "res": res, "alignment": state["alignment"]}
