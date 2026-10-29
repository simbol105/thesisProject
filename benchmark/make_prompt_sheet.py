"""Comparison sheet of the prompt images each method was given, v1 against v2."""
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prompts import DATASET, PROJECTS, train_frame_index          # noqa: E402

OBJECTS = ["Birdhouse_Lighting001", "Cup_B08TWHJ33Q_Tan_Lighting001",
           "Figurine_B08FYFNYP4_LionKing_Lighting001", "Gargoyle_B08SQMBDXY_HandsOnKnees_Lighting001",
           "Mallard_B082D168CK_MintGreen_Lighting001", "Mouse_B0CHNVBBLF_Honeycomb_1_Lighting001"]
BG = 0.92


def font(size: int):
    for p in [Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
              Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]:
        if p.exists():
            return ImageFont.truetype(str(p), size)
    try:
        import matplotlib
        return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"), size)
    except Exception:
        return ImageFont.load_default()


def v1_frames(obj: str):
    """The v1 prompt of each method: RecGen's frame index (also used by TRELLIS) and SAM3D's image."""
    recgen = yaml.safe_load(open(PROJECTS / "recgen" / "thesis_configs" / f"{obj}.yaml"))
    sam3d = yaml.safe_load(open(PROJECTS / "sam3d" / "sam-3d-objects" / "sam3d_configs" / f"{obj}.yaml"))
    return int(recgen["prompt_id"]), train_frame_index(obj, Path(sam3d["image_path"]).stem)


def v2_frame(obj: str, prompts_file: Path) -> int:
    name = yaml.safe_load(prompts_file.read_text())["prompts"][obj]
    return train_frame_index(obj, name)


def photo(obj: str, frame_idx: int, res: int = 800):
    """The dataset photograph of a training frame, composited on the sheet background."""
    frames = json.load(open(DATASET / obj / "transforms_train.json"))["frames"]
    path = (DATASET / obj / frames[frame_idx]["file_path"]).with_suffix(".png")
    im = Image.open(path).convert("RGBA").resize((res, res), Image.LANCZOS)
    img = np.asarray(im).astype(np.float32) / 255.0
    rgb, alpha = img[..., :3], img[..., 3:]
    return rgb * alpha + BG * (1 - alpha), alpha[..., 0]


_MESH_CACHE = {}


def gt_mesh(obj: str):
    """The posed GT mesh, loaded once per object (the GLB files are tens of megabytes)."""
    import trimesh
    if obj not in _MESH_CACHE:
        transforms = json.load(open(DATASET / obj / "transforms_train.json"))
        mesh = trimesh.load_mesh(DATASET / obj / "model.glb")
        mesh.apply_transform(np.array(transforms["object_pose"]))
        _MESH_CACHE[obj] = mesh
    return _MESH_CACHE[obj]


def render_gt(obj: str, frame_idx: int, res: int = 800):
    """Colour and depth of the GT mesh from a training camera: RecGen's exact input."""
    import pyrender
    transforms = json.load(open(DATASET / obj / "transforms_train.json"))
    intr = transforms["intrinsic"]
    first = Image.open((DATASET / obj / transforms["frames"][frame_idx]["file_path"]).with_suffix(".png"))
    w, h = first.size
    s = res / max(w, h)
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
    scene.add(pyrender.Mesh.from_trimesh(gt_mesh(obj), smooth=False))
    camera = pyrender.IntrinsicsCamera(fx=intr["fx"] * s, fy=intr["fy"] * s,
                                       cx=intr["cx"] * s, cy=intr["cy"] * s, znear=0.001, zfar=100.0)
    cv_to_gl = np.diag([1, -1, -1, 1]).astype(np.float32)
    scene.add(camera, pose=np.linalg.inv(np.array(transforms["frames"][frame_idx]["transform_matrix"])) @ cv_to_gl)
    renderer = pyrender.OffscreenRenderer(viewport_width=int(w * s), viewport_height=int(h * s))
    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()
    rgba = color.astype(np.float32) / 255.0
    rgb, alpha = rgba[..., :3], rgba[..., 3:]
    return rgb * alpha + BG * (1 - alpha), depth.astype(np.float32)


def depth_vis(depth: np.ndarray):
    """Depth map coloured over the object only, near bright, far dark."""
    mask = depth > 0
    out = np.full((*depth.shape, 3), BG, dtype=np.float32)
    if not mask.any():
        return out, mask
    d = depth[mask]
    lo, hi = np.percentile(d, 1), np.percentile(d, 99)
    s = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    try:
        import matplotlib
        colored = matplotlib.colormaps["viridis"](1.0 - s)[..., :3]
    except Exception:
        colored = np.stack([1.0 - s, 1.0 - s, np.ones_like(s)], axis=-1)
    out[mask] = colored[mask]
    return out, mask


def crop(images, masks, pad: int = 24):
    """Crop every panel of a column to the union of its silhouettes, keeping them aligned."""
    ys, xs = [], []
    for m in masks:
        if m is None or not m.any():
            continue
        y, x = np.where(m > 0.5)
        ys += [y.min(), y.max()]
        xs += [x.min(), x.max()]
    if not ys:
        return images
    h, w = images[0].shape[:2]
    y0, y1 = max(0, min(ys) - pad), min(h, max(ys) + pad)
    x0, x1 = max(0, min(xs) - pad), min(w, max(xs) + pad)
    side = max(y1 - y0, x1 - x0)                       # square crop, centred
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    y0, x0 = max(0, cy - side // 2), max(0, cx - side // 2)
    y1, x1 = min(h, y0 + side), min(w, x0 + side)
    return [im[y0:y1, x0:x1] for im in images]


def main():
    parser = argparse.ArgumentParser(description="Prompt images per object, v1 against v2")
    parser.add_argument("--objects", nargs="*", default=OBJECTS)
    parser.add_argument("--prompts", type=Path, default=HERE / "prompts.yaml")
    parser.add_argument("--out", type=Path, default=PROJECTS / "benchmark_images" / "prompts_v1_v2.png")
    parser.add_argument("--tile", type=int, default=256, help="output cell size in pixels")
    parser.add_argument("--res", type=int, default=800, help="render/photo working resolution")
    args = parser.parse_args()

    rows = ["v1   prompt image",
            "v2   prompt image",
            "v1   depth input (RecGen + D)",
            "v2   depth input (RecGen + D)"]

    cells = {}
    for obj in args.objects:
        r_idx, s_idx = v1_frames(obj)
        v2_idx = v2_frame(obj, args.prompts)
        print(f"{obj}: v1 frame {r_idx} (SAM3D used {s_idx}), v2 frame {v2_idx}", flush=True)

        p_v1, m_v1 = photo(obj, r_idx, args.res)             # TRELLIS, TRELLIS.2 and RecGen's camera
        p_v2, m_v2 = photo(obj, v2_idx, args.res)
        _, depth_v1 = render_gt(obj, r_idx, args.res)        # the depth RecGen was given
        _, depth_v2 = render_gt(obj, v2_idx, args.res)
        _MESH_CACHE.pop(obj, None)                           # free the mesh before the next object
        d_v1, dm_v1 = depth_vis(depth_v1)
        d_v2, dm_v2 = depth_vis(depth_v2)

        imgs = [p_v1, p_v2, d_v1, d_v2]
        masks = [m_v1, m_v2, dm_v1, dm_v2]
        imgs = crop(imgs, masks)
        for label, im in zip(rows, imgs):
            arr = (np.clip(im, 0, 1) * 255).astype(np.uint8)
            cells[(label, obj)] = Image.fromarray(arr).resize((args.tile, args.tile), Image.LANCZOS)

    label_w, header_h, gap, top = 330, 44, 6, 40
    W = label_w + len(args.objects) * (args.tile + gap)
    H = top + header_h + len(rows) * (args.tile + gap)
    canvas = Image.new("RGB", (W, H + 30), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), "Prompt images per object: v1 against v2.  + D marks the RGB-D method: RecGen "
              "also receives a depth map rendered from the GT mesh.", fill="black", font=font(19))
    f_head, f_row = font(20), font(17)
    for c, obj in enumerate(args.objects):
        draw.text((label_w + c * (args.tile + gap) + args.tile // 2, top + header_h // 2),
                  obj.split("_")[0], fill="black", font=f_head, anchor="mm")
    for r, label in enumerate(rows):
        y = top + header_h + r * (args.tile + gap)
        draw.text((label_w // 2, y + args.tile // 2), label, fill="black", font=f_row, anchor="mm")
        for c, obj in enumerate(args.objects):
            cell = cells.get((label, obj))
            if cell is not None:
                canvas.paste(cell, (label_w + c * (args.tile + gap), y))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"wrote {args.out} ({len(rows)} rows x {len(args.objects)} columns)")


if __name__ == "__main__":
    main()
