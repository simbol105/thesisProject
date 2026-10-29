"""Comparison tiles of the benchmarked methods, one row per test view, GT in the first column."""
import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECTS = Path(__file__).resolve().parents[1]
BENCH = PROJECTS / "benchmark_output"
STYLE = "sheet"        # "thesis": lay out to the thesis figure specification (figure_layout.py), PDF and PNG
CROP = "union"         # "gt": frame on the ground truth only, so stray fragments cannot shrink the object
GT_COLOR = "photo"     # "albedo": the unlit render of the GT mesh instead of the photograph
LABELS = {"nvdiffrec": "nvdiffrec (multi-view)", "nvdiffrec_gtsup": "nvdiffrec (albedo only)",
          "nvdiffrec_gtsup_pass1": "nvdiffrec (5000 it, pass 1)", "recgen_v2": "RecGen", "sam3d_v2": "SAM3D",
          "trellis_v2": "TRELLIS", "trellis2_v2": "TRELLIS.2", "recgen": "RecGen (v1)", "sam3d": "SAM3D (v1)",
          "trellis": "TRELLIS (v1)", "trellis2": "TRELLIS.2 (v1)"}
DEFAULT_METHODS = ["nvdiffrec", "recgen_v2", "sam3d_v2", "trellis_v2", "trellis2_v2"]

SOURCE_LABEL = {"recgen": "RecGen", "sam3d": "SAM3D", "trellis": "TRELLIS", "trellis2": "TRELLIS.2"}
EXPERIMENTS = {"trellis1_slat": "SLAT (T1)", "trellis1_cubefeats": "cubefeats (T1)",
               "trellis2_slat": "SLAT (T2)", "trellis2_ovoxel": "o-voxel (T2)"}
EXPERIMENT_TITLE = {"trellis1_slat": "TRELLIS1 SLAT optimization",
                    "trellis1_cubefeats": "TRELLIS1 cubefeats optimization",
                    "trellis2_slat": "TRELLIS.2 SLAT optimization",
                    "trellis2_ovoxel": "TRELLIS.2 o-voxel optimization"}
for _exp, _elab in EXPERIMENTS.items():
    for _src, _slab in SOURCE_LABEL.items():
        LABELS[f"{_exp}_{_src}_v2"] = f"{_slab} -> {_elab}"
for _it in (500, 1000, 2000):
    LABELS[f"nvdiffrec_gtsup_it{_it}"] = f"nvdiffrec ({_it} it, final)"
    LABELS[f"nvdiffrec_gtsup_it{_it}_pass1"] = f"nvdiffrec ({_it} it, pass 1)"
RES = 512


def font(size: int):
    for path in [Path(cv2.__file__).parent / "qt" / "fonts" / "DejaVuSans.ttf",
                 Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
                 Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    try:
        import matplotlib
        return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"), size)
    except Exception:
        return ImageFont.load_default()


def to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def read_rgb(path: Path) -> np.ndarray:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255.0


def gt_images(obj: str, name: str):
    g = BENCH / "_gt_orb" / obj
    mask = cv2.resize(cv2.imread(str(g / "test_mask" / f"{name}.png"), 0), (RES, RES), interpolation=cv2.INTER_AREA) / 255.0
    if GT_COLOR == "albedo":
        photo = cv2.resize(read_rgb(g / "albedo" / f"{name}.png"), (RES, RES), interpolation=cv2.INTER_AREA)
    else:
        photo = cv2.imread(str(g / "test" / f"{name}.exr"), cv2.IMREAD_UNCHANGED)[..., ::-1]
        photo = to_srgb(cv2.resize(photo, (RES, RES), interpolation=cv2.INTER_AREA))
    normal = cv2.resize(np.load(g / "normal" / f"{name}.npy"), (RES, RES), interpolation=cv2.INTER_AREA)
    return mask, photo, normal


def method_images(method: str, obj: str, name: str):
    o = BENCH / method / obj
    if not (o / "view" / f"{name}.png").exists():
        return None
    depth = cv2.imread(str(o / "depth" / f"{name}.exr"), cv2.IMREAD_UNCHANGED)
    mask = (depth > 0).astype(np.float32)
    color = read_rgb(o / "view" / f"{name}.png")
    normal = cv2.imread(str(o / "normal" / f"{name}.exr"), cv2.IMREAD_UNCHANGED)[..., ::-1]
    return mask, color, normal


COLOR_BG = 0.92


def gt_label(kind: str) -> str:
    return "Ground truth (albedo)" if kind == "color" and GT_COLOR == "albedo" else "Ground truth"


def on_white(img: np.ndarray, mask: np.ndarray, bg: float = 1.0) -> np.ndarray:
    return img * mask[..., None] + bg * (1 - mask[..., None])


def normal_vis(n: np.ndarray, mask: np.ndarray) -> np.ndarray:
    n = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8)
    vis = np.stack([n[..., 0], -n[..., 1], -n[..., 2]], -1) * 0.5 + 0.5
    return on_white(vis, mask)


def crop_box(masks, pad: float = 0.08):
    if CROP == "gt":
        masks, pad = masks[:1], 0.3
    union = np.max(np.stack(masks), 0) > 0.5
    ys, xs = np.nonzero(union)
    if len(ys) == 0:
        return 0, RES, 0, RES
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    half = max(ys.max() - ys.min(), xs.max() - xs.min()) / 2 * (1 + pad) + 4
    y0, x0 = int(max(cy - half, 0)), int(max(cx - half, 0))
    y1, x1 = int(min(cy + half, RES)), int(min(cx + half, RES))
    return y0, y1, x0, x1


def thesis_label(text: str) -> str:
    """Row and column titles for the thesis layout: short, and without the arrow of 'source -> experiment'."""
    return text.split(" -> ")[0].replace("Ground truth (albedo)", "GT albedo")


def build(obj: str, methods, views, kind: str, tile: int, out: Path):
    names = sorted(p.stem for p in (BENCH / "_gt_orb" / obj / "test").glob("*.exr"))
    names = [names[i] for i in np.linspace(0, len(names) - 1, views).round().astype(int)]
    present = [m for m in methods if (BENCH / m / obj / "view").exists()]
    header_h, label_w, gap = 44, 110, 6
    cols = [gt_label(kind)] + [LABELS.get(m, m) for m in present]
    W = label_w + len(cols) * (tile + gap)
    H = header_h + len(names) * (tile + gap)
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    f_head, f_row = font(20), font(16)
    for c, text in enumerate(cols):
        x = label_w + c * (tile + gap) + tile // 2
        draw.text((x, header_h // 2), text, fill="black", font=f_head, anchor="mm")
    matrix = []
    for r, name in enumerate(names):
        gmask, gphoto, gnormal = gt_images(obj, name)
        imgs, masks = [], [gmask]
        imgs.append(on_white(gphoto, gmask, COLOR_BG) if kind == "color" else normal_vis(gnormal, gmask > 0.5))
        for m in present:
            res = method_images(m, obj, name)
            if res is None:
                imgs.append(np.ones((RES, RES, 3), np.float32))
                continue
            mask, color, normal = res
            masks.append(mask)
            imgs.append(on_white(color, mask, COLOR_BG) if kind == "color" else normal_vis(normal, mask > 0.5))
        y0, y1, x0, x1 = crop_box(masks)
        y = header_h + r * (tile + gap)
        draw.text((label_w // 2, y + tile // 2), name.replace("im_", "view "), fill="black", font=f_row, anchor="mm")
        matrix.append([np.clip(img[y0:y1, x0:x1], 0, 1) for img in imgs])
        for c, img in enumerate(imgs):
            crop = (np.clip(img[y0:y1, x0:x1], 0, 1) * 255).astype(np.uint8)
            canvas.paste(Image.fromarray(crop).resize((tile, tile), Image.LANCZOS), (label_w + c * (tile + gap), y))
    if STYLE == "thesis":
        from figure_layout import figure_grid
        figure_grid([(thesis_label(text), [row[c] for row in matrix]) for c, text in enumerate(cols)],
                    out / f"{obj}_{kind}.png", col_labels=[n.replace("im_", "View ") for n in names])
        return
    title = obj.replace("_Lighting001", "") + ("  (base color)" if kind == "color" else "  (normals)")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{obj}_{kind}.png"
    full = Image.new("RGB", (W, H + 36), "white")
    ImageDraw.Draw(full).text((12, 8), title, fill="black", font=font(22))
    full.paste(canvas, (0, 36))
    full.save(path)
    print(f"wrote {path} ({len(names)} views x {len(cols)} columns)")


def build_overview(objects, methods, view_index: int, view_override: dict, kind: str, tile: int,
                   transpose: bool, out: Path, stem: str = "overview", title: str = ""):
    """One image: rows = GT + methods, columns = objects (or transposed), one test view per object."""
    rows = [("gt", gt_label(kind))] + [(m, LABELS.get(m, m)) for m in methods]
    cells = {}
    for obj in objects:
        names = sorted(p.stem for p in (BENCH / "_gt_orb" / obj / "test").glob("*.exr"))
        name = view_override.get(obj) or names[min(view_index, len(names) - 1)]
        gmask, gphoto, gnormal = gt_images(obj, name)
        imgs = {"gt": on_white(gphoto, gmask, COLOR_BG) if kind == "color" else normal_vis(gnormal, gmask > 0.5)}
        masks = [gmask]
        for m in methods:
            res = method_images(m, obj, name)
            if res is None:
                continue
            mask, color, normal = res
            masks.append(mask)
            imgs[m] = on_white(color, mask, COLOR_BG) if kind == "color" else normal_vis(normal, mask > 0.5)
        y0, y1, x0, x1 = crop_box(masks)
        for key, img in imgs.items():
            crop = (np.clip(img[y0:y1, x0:x1], 0, 1) * 255).astype(np.uint8)
            cells[(key, obj)] = Image.fromarray(crop).resize((tile, tile), Image.LANCZOS)
    col_labels = [obj.split("_")[0].replace("Lighting001", "") for obj in objects]

    grid_rows, grid_cols = rows, list(zip(objects, col_labels))
    if transpose:
        grid_rows, grid_cols = [(o, l) for o, l in zip(objects, col_labels)], rows
    if STYLE == "thesis":
        from figure_layout import figure_grid
        as_array = lambda cell: None if cell is None else np.asarray(cell, dtype=np.float32) / 255.0
        figure_grid([(thesis_label(rtext), [as_array(cells.get((rkey, ckey) if not transpose else (ckey, rkey)))
                                            for ckey, _ in grid_cols]) for rkey, rtext in grid_rows],
                    out / f"{stem}_{kind}.png", col_labels=[thesis_label(text) for _, text in grid_cols])
        return
    label_w = 260 if not transpose else 130
    header_h, gap = 44, 6
    W = label_w + len(grid_cols) * (tile + gap)
    H = header_h + len(grid_rows) * (tile + gap)
    canvas = Image.new("RGB", (W, H + 36), "white")
    draw = ImageDraw.Draw(canvas)
    heading = "Base color / RGB" if kind == "color" else "Normals"
    draw.text((12, 8), f"{heading} - {title}" if title else heading, fill="black", font=font(22))
    f_head, f_row = font(20), font(18)
    for c, (_, text) in enumerate(grid_cols):
        draw.text((label_w + c * (tile + gap) + tile // 2, 36 + header_h // 2), text, fill="black", font=f_head, anchor="mm")
    for r, (rkey, rtext) in enumerate(grid_rows):
        y = 36 + header_h + r * (tile + gap)
        draw.text((label_w // 2, y + tile // 2), rtext, fill="black", font=f_row, anchor="mm")
        for c, (ckey, _) in enumerate(grid_cols):
            key, obj = (rkey, ckey) if not transpose else (ckey, rkey)
            cell = cells.get((key, obj))
            if cell is None:
                draw.text((label_w + c * (tile + gap) + tile // 2, y + tile // 2), "n/a", fill="gray", font=f_row, anchor="mm")
            else:
                canvas.paste(cell, (label_w + c * (tile + gap), y))
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{stem}_{kind}.png"
    canvas.save(path)
    print(f"wrote {path} ({len(grid_rows)} rows x {len(grid_cols)} columns)")


def main():
    global BENCH, GT_COLOR, CROP, STYLE
    parser = argparse.ArgumentParser(description="Comparison tiles of the benchmarked methods")
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--objects", nargs="*", default=None)
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--out", type=Path, default=PROJECTS / "benchmark_images")
    parser.add_argument("--bench", type=Path, default=BENCH, help="benchmark folder to read")
    parser.add_argument("--style", choices=["sheet", "thesis"], default="sheet",
                        help="thesis: 5.49 in wide, at most 8 in high, 9 pt Computer Modern, PDF and 300 dpi PNG")
    parser.add_argument("--crop", choices=["union", "gt"], default="union",
                        help="frame each cell on all masks, or on the ground truth only")
    parser.add_argument("--gt", choices=["photo", "albedo"], default="photo",
                        help="ground-truth colour: the photograph, or the unlit render of the GT mesh")
    parser.add_argument("--overview", action="store_true", help="one image for all objects and methods")
    parser.add_argument("--view-index", type=int, default=2, help="overview: index into the sorted test views")
    parser.add_argument("--view", nargs="*", default=[], help="overview: per-object view, e.g. Mallard_B082D168CK_MintGreen_Lighting001=im_0060")
    parser.add_argument("--transpose", action="store_true", help="overview: objects on the rows, methods on the columns")
    parser.add_argument("--name", default="overview", help="overview: output file stem (<name>_color.png)")
    parser.add_argument("--title", default="", help="overview: extra text in the heading")
    parser.add_argument("--experiments", nargs="*", choices=list(EXPERIMENTS), default=None,
                        help="overview: one image pair per latent-optimization experiment "
                             "(rows = GT + the four mesh sources it was encoded from)")
    args = parser.parse_args()
    BENCH, GT_COLOR, CROP, STYLE = args.bench, args.gt, args.crop, args.style
    objects = args.objects or sorted(p.name for p in (BENCH / "_gt_orb").iterdir() if p.is_dir())
    if args.overview:
        override = dict(v.split("=", 1) for v in args.view)
        for kind in ("color", "normals"):
            if args.experiments:
                for exp in args.experiments:
                    build_overview(objects, [f"{exp}_{s}_v2" for s in SOURCE_LABEL], args.view_index,
                                   override, kind, args.tile, args.transpose, args.out,
                                   stem=f"overview_{exp}", title=EXPERIMENT_TITLE[exp])
            else:
                build_overview(objects, args.methods, args.view_index, override, kind, args.tile,
                               args.transpose, args.out, stem=args.name, title=args.title)
        return
    for obj in objects:
        for kind in ("color", "normals"):
            build(obj, args.methods, args.views, kind, args.tile, args.out)


if __name__ == "__main__":
    main()
