"""Images for the placeholders of the pipeline diagram, as single tiles and as one labelled slate."""
import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree

from dtc_orb_benchmark import DATASET, load_dtc
from make_tiles import font

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parent
BENCH = PROJECTS / "benchmark_output_v2"
OPT = PROJECTS / "trellis1" / "TRELLIS" / "optimization_output"
RES, TILE = 512, 320


def crop_box(mask: np.ndarray, pad: float = 0.12):
    ys, xs = np.nonzero(mask)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    half = max(ys.max() - ys.min(), xs.max() - xs.min()) / 2 * (1 + pad) + 4
    return (int(max(cy - half, 0)), int(min(cy + half, mask.shape[0])),
            int(max(cx - half, 0)), int(min(cx + half, mask.shape[1])))


def tile(img: np.ndarray, box) -> Image.Image:
    y0, y1, x0, x1 = box
    crop = (np.clip(img[y0:y1, x0:x1], 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(crop).resize((TILE, TILE), Image.LANCZOS)


def on_white(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.float32)[..., None]
    return img * m + (1 - m)


def buffers(folder: Path, name: str, gt: bool):
    """mask, depth, camera-space normal and unlit colour of one benchmarked view, at RES."""
    if gt:
        area = lambda x: cv2.resize(x, (RES, RES), interpolation=cv2.INTER_AREA)
        depth, normal = area(np.load(folder / "z_depth" / f"{name}.npy")), area(np.load(folder / "normal" / f"{name}.npy"))
    else:
        depth = cv2.imread(str(folder / "depth" / f"{name}.exr"), cv2.IMREAD_UNCHANGED)
        normal = cv2.imread(str(folder / "normal" / f"{name}.exr"), cv2.IMREAD_UNCHANGED)[..., ::-1]
    color = cv2.imread(str(folder / "albedo" / f"{name}.png"), cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255
    color = cv2.resize(color, (RES, RES), interpolation=cv2.INTER_AREA)
    return depth > 0, depth, normal, color


def shaded(normal: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Grey clay render from the camera-space normals: a head light plus a little fill."""
    n = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-8)
    light = np.array([0.35, -0.45, -0.82]) / np.linalg.norm([0.35, -0.45, -0.82])
    lam = np.clip((n * light).sum(-1), 0, 1) * 0.75 + 0.25
    return on_white(np.repeat(lam[..., None], 3, -1) * np.array([0.82, 0.84, 0.88]), mask)


def depth_range(mask, depth):
    inner = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0     # edges mix in background
    return np.percentile(depth[inner], 1), np.percentile(depth[inner], 99)


def four(mask, depth, normal, color, lo, hi):
    d = (1 - np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)) * 0.8 + 0.15    # near is bright
    n = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-8)
    nvis = np.stack([n[..., 0], -n[..., 1], -n[..., 2]], -1) * 0.5 + 0.5
    m3 = np.repeat(mask[..., None].astype(np.float32), 3, -1)
    return [1 - m3 * 0.85, on_white(np.repeat(d[..., None], 3, -1), mask), on_white(nvis, mask), on_white(color, mask)]


def voxels(coords: np.ndarray, colors: np.ndarray, grid: int, A: np.ndarray, E: np.ndarray, K: np.ndarray,
           res: int = 1024) -> np.ndarray:
    """Splat the active voxels at the DTC camera: one shaded square per voxel, nearest wins."""
    occ = np.zeros((grid,) * 3, np.float32)
    occ[tuple(coords.T)] = 1
    blur = ndimage.gaussian_filter(occ, 1.5)
    grad = np.stack(np.gradient(blur), -1)[tuple(coords.T)]               # points into the object
    world = ((coords + 0.5) / grid - 0.5) @ A[:3, :3].T + A[:3, 3]
    cam = world @ E[:3, :3].T + E[:3, 3]
    n = -(grad @ A[:3, :3].T) @ E[:3, :3].T
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8
    light = np.array([0.35, -0.45, -0.82]) / np.linalg.norm([0.35, -0.45, -0.82])
    shade = np.clip(n @ light, 0, 1) * 0.6 + 0.4
    u = (K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]) * res
    v = (K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]) * res
    size = np.cbrt(abs(np.linalg.det(A[:3, :3]))) / grid * K[0, 0] * res / cam[:, 2].mean()
    half = max(int(np.ceil(size * 0.5)), 1)
    order = np.argsort(-cam[:, 2])                                         # far first, near overwrites
    u, v, rgb = u[order].round().astype(int), v[order].round().astype(int), (colors * shade[:, None])[order]
    img = np.ones((res, res, 3), np.float32)
    zbuf = np.full((res, res), np.inf, np.float32)
    z = cam[order, 2].astype(np.float32)
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            edge = 0.72 if grid <= 64 and (abs(dy) == half or abs(dx) == half) else 1.0     # voxel outline
            yy, xx = v + dy, u + dx
            ok = (yy >= 0) & (yy < res) & (xx >= 0) & (xx < res)
            yy, xx, zz, cc = yy[ok], xx[ok], z[ok], rgb[ok] * edge
            closer = zz <= zbuf[yy, xx]
            img[yy[closer], xx[closer]] = cc[closer]
            np.minimum.at(zbuf, (yy, xx), zz)
    return cv2.resize(img, (RES, RES), interpolation=cv2.INTER_AREA)


def pca_colors(feats: np.ndarray) -> np.ndarray:
    x = feats - feats.mean(0)
    _, _, vt = np.linalg.svd(x[:: max(len(x) // 20000, 1)], full_matrices=False)
    p = x @ vt[:3].T
    lo, hi = np.percentile(p, 2, axis=0), np.percentile(p, 98, axis=0)
    return np.clip((p - lo) / (hi - lo + 1e-8), 0, 1) * 0.85 + 0.1


def main():
    parser = argparse.ArgumentParser(description="Images for the pipeline diagram placeholders")
    parser.add_argument("--object", default="Birdhouse_Lighting001")
    parser.add_argument("--source", default="recgen", choices=["recgen", "sam3d", "trellis", "trellis2"])
    parser.add_argument("--view", type=int, default=2, help="index into the test views")
    parser.add_argument("--out", type=Path, default=PROJECTS / "benchmark_images_v2" / "pipeline")
    args = parser.parse_args()

    obj, src = args.object, args.source
    dtc = load_dtc(obj)
    name_path, E = dtc["test"][args.view]
    name, K = Path(name_path).name, dtc["K"]
    gt = buffers(BENCH / "_gt_orb" / obj, name, gt=True)
    pred = buffers(BENCH / f"trellis1_slat_{src}_v2" / obj, name, gt=False)
    source = buffers(BENCH / f"{src}_v2" / obj, name, gt=False)
    box = crop_box(gt[0])
    lo, hi = depth_range(gt[0], gt[1])            # one depth scale for both rows

    prompt = yaml.safe_load((HERE / "prompts.yaml").read_text())["prompts"][obj]
    photo_path = next(DATASET / obj / (p + ".png") for p, _ in dtc["train"] if Path(p).name == prompt)
    rgba = cv2.imread(str(photo_path), cv2.IMREAD_UNCHANGED)
    pbox = crop_box(rgba[..., 3] > 127, pad=0.25)
    photo = Image.fromarray(rgba[pbox[0]:pbox[1], pbox[2]:pbox[3], 2::-1]).resize((TILE, TILE), Image.LANCZOS)

    slat = torch.load(OPT / "slat_opt" / "v2" / "slat" / src / obj / "final_mesh" / "slat.pt", map_location="cpu", weights_only=False)
    cube = torch.load(OPT / "cubefeats_opt" / "v2" / src / obj / "final_mesh" / "cubefeats.pt", map_location="cpu", weights_only=False)
    s_xyz, s_feat = slat["coords"][:, 1:].numpy(), slat["feats"].numpy()
    c_xyz, c_feat = cube["coords"][:, 1:].numpy(), cube["feats"].numpy()
    # the colour channels only mean something where they are interpolated onto the surface
    crossing = (c_feat[:, :8].min(1) < 0) & (c_feat[:, :8].max(1) > 0)
    c_xyz = c_xyz[crossing]
    surf = trimesh.load(str(OPT / "cubefeats_opt" / "v2" / src / obj / "final_mesh" / "cubefeats_opt_world.glb"),
                        force="mesh", process=False)
    A = cube["alignment"]
    world = ((c_xyz + 0.5) / 256 - 0.5) @ A[:3, :3].T + A[:3, 3]
    nearest = cKDTree(np.asarray(surf.vertices)).query(world)[1]
    c_rgb = np.asarray(surf.visual.vertex_colors)[nearest, :3] / 255.0
    slat_img = voxels(s_xyz, pca_colors(s_feat), 64, slat["alignment"], E, K)
    cube_img = voxels(c_xyz, c_rgb, 256, cube["alignment"], E, K)

    label = {"recgen": "RecGen", "sam3d": "SAM 3D", "trellis": "TRELLIS", "trellis2": "TRELLIS.2"}[src]
    rows = [
        ("input_image", "Input image", f"one photograph ({prompt})", [photo]),
        ("source_mesh", "Source mesh", f"{label}, aligned to the scene", [tile(shaded(source[2], source[0]), box)]),
        ("structured_latent", "Structured latent z (SLAT)", f"{len(s_xyz):,} active voxels at 64³, 8 channels\n(colour = first 3 principal components)",
         [tile(slat_img, box)]),
        ("explicit_representation", "Decoded explicit representation", f"cubefeats at 256³, 101 channels: {len(c_feat):,} voxels,\n{len(c_xyz):,} crossed by the surface (shown, coloured\nwith the decoded surface colour)",
         [tile(cube_img, box)]),
        ("decoded_mesh", "Decoded mesh", "FlexiCubes surface from the decoder", [tile(shaded(pred[2], pred[0]), box)]),
        ("renders", "Renders", "mask, depth, normal, unlit colour", [tile(x, box) for x in four(*pred, lo, hi)]),
        ("gt_mesh", "GT mesh", "DTC scan in the same frame", [tile(shaded(gt[2], gt[0]), box)]),
        ("reference_renders", "Reference renders", "mask, depth, normal, unlit colour", [tile(x, box) for x in four(*gt, lo, hi)]),
    ]

    out = args.out / obj
    out.mkdir(parents=True, exist_ok=True)
    sub = ["mask", "depth", "normal", "colour"]
    for i, (key, _t, _c, tiles) in enumerate(rows, 1):
        for j, im in enumerate(tiles):
            im.save(out / (f"{i:02d}_{key}.png" if len(tiles) == 1 else f"{i:02d}_{key}_{sub[j]}.png"))

    label_w, gap, top = 470, 10, 64
    W, H = label_w + 4 * (TILE + gap) + gap, top + len(rows) * (TILE + gap + 26)
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 16), f"Pipeline diagram images: {obj.split('_')[0]}, {label} → TRELLIS SLAT, test view {name}",
              fill="black", font=font(24))
    for r, (_key, title, caption, tiles) in enumerate(rows):
        y = top + r * (TILE + gap + 26)
        draw.text((16, y + TILE // 2 - 44), title, fill="black", font=font(24))
        draw.multiline_text((16, y + TILE // 2 - 8), caption, fill=(90, 90, 90), font=font(17), spacing=6)
        for j, im in enumerate(tiles):
            x = label_w + j * (TILE + gap)
            canvas.paste(im, (x, y))
            draw.rectangle([x, y, x + TILE - 1, y + TILE - 1], outline=(200, 200, 200))
            if len(tiles) > 1:
                draw.text((x + TILE // 2, y + TILE + 12), sub[j], fill=(90, 90, 90), font=font(16), anchor="mm")
    canvas.save(out / "pipeline_slate.png")
    print(f"wrote {out / 'pipeline_slate.png'} and {sum(len(t) for *_x, t in rows)} tiles in {out}")


if __name__ == "__main__":
    main()
