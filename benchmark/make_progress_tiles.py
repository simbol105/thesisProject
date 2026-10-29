"""Progress tiles from images that earlier runs already saved; nothing is rendered or trained."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from figure_layout import figure_grid

PROJECTS = Path(__file__).resolve().parents[1]
NVDIFFREC = PROJECTS / "nvdiffrec" / "out" / "flexicubes_gtsup_cluster"
SLAT = PROJECTS / "trellis1" / "TRELLIS" / "optimization_output" / "slat_opt" / "v2" / "slat"
DILATION = PROJECTS / "trellis1" / "TRELLIS" / "optimization_output" / "slat_dilation" / "Birdhouse_chimney"
DATASET = PROJECTS / "dataset" / "DTC"
OBJECTS = ["Birdhouse_Lighting001", "Cup_B08TWHJ33Q_Tan_Lighting001", "Figurine_B08FYFNYP4_LionKing_Lighting001",
           "Gargoyle_B08SQMBDXY_HandsOnKnees_Lighting001", "Mallard_B082D168CK_MintGreen_Lighting001",
           "Mouse_B0CHNVBBLF_Honeycomb_1_Lighting001"]
PANEL = 512


def panel(path: Path, index: int) -> np.ndarray:
    """The index-th 512 px square of a saved strip, float RGB."""
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return img[:, index * PANEL:(index + 1) * PANEL]


def nearest(wanted: int, available) -> int:
    return min(available, key=lambda a: (abs(a - wanted), a))


def step_label(wanted: int, got: int) -> str:
    """The step that is shown; when the wanted one was not saved this is the nearest saved step."""
    if got != wanted:
        print(f"  step {wanted} was not saved, showing {got}")
    return "Initialization" if got == 0 else f"{got} steps"


def grid(cells, row_labels, col_labels, title: str, path: Path, **_unused):
    """cells[r][c]: float RGB array or None."""
    figure_grid([(label, row) for label, row in zip(row_labels, cells)], path, col_labels=col_labels)


def nvdiffrec_tiles(out: Path, steps):
    """img_dmtet_pass1_<k>.png is written every 200 steps: opt | ref | light | kd | ks | normal."""
    interval = json.loads((PROJECTS / "nvdiffrec" / "configs" / "flexicubes_gtsup_cluster" / f"{OBJECTS[0]}.json").read_text())["save_interval"]
    for kind, index in (("color", 3), ("normals", 5)):
        cells, cols = [], None
        for obj in OBJECTS:
            saved = {int(p.stem.rsplit("_", 1)[1]) * interval: p for p in (NVDIFFREC / obj).glob("img_dmtet_pass1_*.png")}
            got = [nearest(s, saved) for s in steps]
            cols = [step_label(s, g) for s, g in zip(steps, got)]
            row = []
            for g in got:
                img = panel(saved[g], index)
                if kind == "normals":
                    img = img * 2 - 1          # train.py encodes the display normal twice; undo one
                row.append(img)
            cells.append(row)
        grid(cells, [o.split("_")[0] for o in OBJECTS], cols,
             f"nvdiffrec + FlexiCubes, GT-supervised: {'base colour' if kind == 'color' else 'normals'} during the FlexiCubes pass",
             out / f"progress_nvdiffrec_{kind}.png")


def nvdiffrec_fixed_view_tiles(out: Path, steps):
    """The fixed-view progress renders of run_gtsup_progress_dtc.sh"""
    root = PROJECTS / "nvdiffrec" / "out" / "flexicubes_gtsup_progress"
    for kind, index in (("color", 3), ("normals", 5)):
        rows, cols = [], None
        for obj in OBJECTS:
            saved = {int(q.stem.rsplit("_", 1)[1]): q for q in (root / obj / "progress").glob("dmtet_pass1_*.png")}
            if not saved:
                print(f"  {obj}: no progress renders in {root / obj / 'progress'}")
                rows.append((obj.split("_")[0], [None] * len(steps)))
                continue
            got = [nearest(st, saved) for st in steps]
            cols = [step_label(st, g) for st, g in zip(steps, got)]
            # frame the row on the object (the reference panel, white outside it), with room for fragments
            ref = panel(saved[got[-1]], 1)
            ys, xs = np.nonzero(ref.min(axis=2) < 0.97)
            half = int(max(np.ptp(ys), np.ptp(xs)) * 0.75) + 8
            cy, cx = int((ys.min() + ys.max()) / 2), int((xs.min() + xs.max()) / 2)
            y0, x0 = max(cy - half, 0), max(cx - half, 0)
            cells = []
            for g in got:
                img = panel(saved[g], index)
                if kind == "normals":
                    img = np.clip(img * 2 - 1, 0, 1)        # train.py encodes the display normal twice; undo one
                cells.append(img[y0:min(cy + half, PANEL), x0:min(cx + half, PANEL)])
            rows.append((obj.split("_")[0], cells))
        figure_grid(rows, out / f"progress_nvdiffrec_fixedview_{kind}.png", col_labels=cols)


def slat_tiles(out: Path, steps, source: str, view: int):
    """test_<iter>_<view>.png: pred colour | GT colour | pred normal | GT normal | overlay."""
    for kind, pred, gt in (("color", 0, 1), ("normals", 2, 3)):
        cells, cols = [], None
        for obj in OBJECTS:
            folder = SLAT / source / obj / "test"
            saved = {int(p.stem.split("_")[1]): p for p in folder.glob(f"test_*_{view:03d}.png")}
            if not saved:
                cells.append([None] * (len(steps) + 1))
                continue
            got = [nearest(s, saved) for s in steps]
            cols = [step_label(s, g) for s, g in zip(steps, got)] + ["Ground truth"]
            row = [panel(saved[g], pred) for g in got] + [panel(saved[got[-1]], gt)]
            # frame the row on the object: the ground-truth colour panel is black outside it
            ys, xs = np.nonzero(panel(saved[got[-1]], 1).max(axis=2) > 0.03)
            if len(ys):
                half = int(max(np.ptp(ys), np.ptp(xs)) * 0.62) + 8
                cy, cx = int((ys.min() + ys.max()) / 2), int((xs.min() + xs.max()) / 2)
                y0, x0 = max(cy - half, 0), max(cx - half, 0)
                row = [img[y0:min(cy + half, PANEL), x0:min(cx + half, PANEL)] for img in row]
            cells.append(row)
        label = {"recgen": "RecGen", "sam3d": "SAM 3D", "trellis": "TRELLIS", "trellis2": "TRELLIS.2"}[source]
        grid(cells, [o.split("_")[0] for o in OBJECTS], cols,
             f"TRELLIS SLAT refinement from the {label} mesh: {'base colour' if kind == 'color' else 'normals'}",
             out / f"progress_slat_{kind}.png")


def chimney_tile(out: Path, arm: str, steps, view: int):
    """region panels: pred colour | GT colour | pred normal | GT normal | overlay"""
    summary = json.loads((DILATION / "summary.json").read_text())
    views = sorted(int(p.stem.rsplit("_", 1)[1]) for p in (DILATION / "prompt").glob("region_view_*.png"))
    view = view if view in views else views[len(views) // 2]
    zoom = lambda img: img[0:330, 20:350]                    # the roof and the chimney

    rgba = Image.open(DATASET / OBJECTS[0] / "train" / "im_0005.png").convert("RGBA")
    alpha = np.asarray(rgba)[..., 3] > 127
    ys, xs = np.nonzero(alpha)
    half = int(max(np.ptp(ys), np.ptp(xs)) * 0.58)
    cy, cx = int(ys.mean()), int(xs.mean())
    photo = np.asarray(rgba.convert("RGB").crop((cx - half, cy - half, cx + half, cy + half)), dtype=np.float32) / 255.0

    saved = {int(p.stem.split("_")[1]): p for p in (DILATION / arm / "region").glob(f"iter_[0-9]*_view_{view:03d}.png")}
    got = [nearest(s, saved) for s in steps]
    prompt_panel = DILATION / "prompt" / f"region_view_{view:03d}.png"
    pct = lambda r: f"{r.get('region_recall_2vox', 0) * 100:.0f}%"

    strategy = {"baseline": "No dilation", "shell_r1": "Shell (r=1)", "shell_r2": "Shell (r=2)",
                "random_r2": "Random", "grow": "Progressive", "guided": "Guided"}
    rows = [
        ("Prompt and initialization",
         [photo, panel(prompt_panel, 0), panel(prompt_panel, 2), zoom(panel(prompt_panel, 0)), zoom(panel(prompt_panel, 2)),
          zoom(panel(prompt_panel, 1))],
         ["Prompt image", "Sampled mesh", "Normals", "Roof", "Roof normals", "Ground truth"]),
        (f"Refinement ({strategy.get(arm, arm).lower()})",
         [zoom(panel(saved[g], 0)) for g in got], [f"{g} steps" for g in got]),
        ("Normals", [zoom(panel(saved[g], 2)) for g in got], None),
        ("After 500 steps",
         [zoom(panel(DILATION / a / "final" / f"region_view_{view:03d}.png", 0)) for a in summary],
         [f"{strategy.get(a, a)}\n{pct(summary[a]['final'])} covered" for a in summary]),
    ]
    figure_grid(rows, out / "chimney_growth.png")
    print(f"  chimney: view {view}, arm {arm}")


FAMILIES = {      # explicit representation and latent of each model, as benchmarked
    "t1": [("trellis1_cubefeats", "Cubefeats (T1)"), ("trellis1_slat", "SLAT (T1)")],
    "t2": [("trellis2_ovoxel", "O-voxel (T2)"), ("trellis2_slat", "SLAT (T2)")],
}
FAMILIES["both"] = FAMILIES["t1"] + FAMILIES["t2"]
FAMILY_TITLE = {"t1": "TRELLIS", "t2": "TRELLIS.2", "both": "TRELLIS and TRELLIS.2"}


def explicit_vs_latent_tile(out: Path, source: str, family: str, view_index: int):
    """Same source mesh, same test camera: initial mesh | explicit optimized | latent optimized | GT."""
    import make_tiles as mt

    mt.BENCH, mt.GT_COLOR, mt.CROP = PROJECTS / "benchmark_output_v2", "albedo", "gt"
    label = {"recgen": "RecGen", "sam3d": "SAM 3D", "trellis": "TRELLIS", "trellis2": "TRELLIS.2"}[source]
    methods = [(f"{source}_v2", f"Before ({label})")]
    methods += [(f"{exp}_{source}_v2", text) for exp, text in FAMILIES[family]]
    for kind in ("color", "normals"):
        cells = []
        for obj in OBJECTS:
            names = sorted(q.stem for q in (mt.BENCH / "_gt_orb" / obj / "test").glob("*.exr"))
            name = names[min(view_index, len(names) - 1)]
            gmask, gcolor, gnormal = mt.gt_images(obj, name)
            show = lambda color, normal, mask: (mt.on_white(color, mask, mt.COLOR_BG) if kind == "color"
                                                else mt.on_white(mt.normal_vis(normal, mask > 0.5), mask, mt.COLOR_BG))
            y0, y1, x0, x1 = mt.crop_box([gmask])
            row = []
            for method, _ in methods:
                res = mt.method_images(method, obj, name)
                row.append(None if res is None else show(res[1], res[2], res[0])[y0:y1, x0:x1])
            row.append(show(gcolor, gnormal, gmask)[y0:y1, x0:x1])
            cells.append(row)
        gt_label = "GT albedo" if kind == "color" else "GT normals"
        grid(cells, [o.split("_")[0] for o in OBJECTS], [m[1] for m in methods] + [gt_label],
             f"{FAMILY_TITLE[family]}: explicit representation against structured latent, "
             f"{'base colour' if kind == 'color' else 'normals'}",
             out / f"compare_explicit_latent_{family}_{source}_{kind}.png")


def prompt_sheet(out: Path):
    """The v1 / v2 prompt images and RecGen depth inputs"""
    src = PROJECTS / "benchmark_images" / "prompts_v1_v2.png"
    if not src.exists():
        print(f"no {src}; run benchmark/make_prompt_sheet.py first")
        return
    sheet = np.asarray(Image.open(src).convert("RGB"), dtype=np.float32) / 255.0
    label_w, header_h, gap, top, tile = 330, 44, 6, 40, 256          # the layout of make_prompt_sheet.py
    labels = ["v1 prompt", "v2 prompt", "v1 depth (RecGen)", "v2 depth (RecGen)"]
    rows = []
    for r, label in enumerate(labels):
        y = top + header_h + r * (tile + gap)
        rows.append((label, [sheet[y:y + tile, label_w + c * (tile + gap):label_w + c * (tile + gap) + tile]
                             for c in range(len(OBJECTS))]))
    figure_grid(rows, out / "prompts_v1_v2.png", col_labels=[o.split("_")[0] for o in OBJECTS])


def main():
    parser = argparse.ArgumentParser(description="Progress tiles from images earlier runs saved")
    parser.add_argument("--out", type=Path, default=PROJECTS / "thesis_figures")
    parser.add_argument("--nvdiffrec-steps", nargs="+", type=int, default=[0, 200, 500, 1000, 2000, 3000, 5000])
    parser.add_argument("--nvdiffrec-fixed-steps", nargs="+", type=int, default=[0, 20, 50, 100, 200, 500, 1000, 5000],
                        help="columns of the fixed-view nvdiffrec figure (run_gtsup_progress_dtc.sh)")
    parser.add_argument("--slat-steps", nargs="+", type=int, default=[0, 100, 200, 500])
    parser.add_argument("--slat-source", default="recgen", choices=["recgen", "sam3d", "trellis", "trellis2"])
    parser.add_argument("--slat-view", type=int, default=2, help="index of the test view")
    parser.add_argument("--chimney-arm", default="guided")
    parser.add_argument("--chimney-steps", nargs="+", type=int, default=[0, 50, 100, 200, 300, 500])
    parser.add_argument("--chimney-view", type=int, default=70, help="training view of the region panels")
    parser.add_argument("--compare-sources", nargs="+", default=["recgen"],
                        choices=["recgen", "sam3d", "trellis", "trellis2"])
    parser.add_argument("--compare-families", nargs="+", default=["both", "t1", "t2"], choices=list(FAMILIES),
                        help="t1: cubefeats and SLAT of TRELLIS; t2: o-voxel and SLAT of TRELLIS.2; both: one image")
    parser.add_argument("--compare-view", type=int, default=2, help="index of the test view")
    parser.add_argument("--only", nargs="*", default=None, choices=["nvdiffrec", "nvdiffrec_fixed", "slat", "chimney", "compare", "prompts"])
    args = parser.parse_args()
    todo = args.only or ["nvdiffrec", "slat", "chimney", "compare", "prompts"]   # nvdiffrec_fixed: after its run
    if "prompts" in todo:
        prompt_sheet(args.out)
    if "compare" in todo:
        for source in args.compare_sources:
            for family in args.compare_families:
                explicit_vs_latent_tile(args.out, source, family, args.compare_view)
    if "nvdiffrec_fixed" in todo:
        nvdiffrec_fixed_view_tiles(args.out, args.nvdiffrec_fixed_steps)
    if "nvdiffrec" in todo:
        nvdiffrec_tiles(args.out, args.nvdiffrec_steps)
    if "slat" in todo:
        slat_tiles(args.out, args.slat_steps, args.slat_source, args.slat_view)
    if "chimney" in todo:
        chimney_tile(args.out, args.chimney_arm, args.chimney_steps, args.chimney_view)


if __name__ == "__main__":
    main()
