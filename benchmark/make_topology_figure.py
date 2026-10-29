"""Thesis figure for the TRELLIS.2 topology experiment"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from figure_layout import BOLD, DPI, GAP_IN, LABEL_PT, REGULAR, SMALL_PT, WIDTH_IN   # noqa: E402

import matplotlib.patheffects as pe                                                  # noqa: E402
import matplotlib.pyplot as plt                                                      # noqa: E402
from matplotlib.patches import Rectangle                                             # noqa: E402
from PIL import Image                                                                # noqa: E402

plt.rcParams["axes.unicode_minus"] = False          # Computer Modern (cmr10) has no U+2212 minus sign

PROJECTS = HERE.parent
RUN = PROJECTS / "trellis2" / "TRELLIS.2" / "topology_study" / "output" / "Birdhouse_Lighting001"
OBJ = "Birdhouse_Lighting001"
MESHES = {"init": "slat/mesh_init_world.ply", "slat": "slat/mesh_final_world.ply", "ovoxel": "ovoxel/mesh_final_world.ply"}
PANEL = 512


def chimney_points(p_obj: np.ndarray, margin: float = 0.004) -> np.ndarray:
    """Points above the roof plane on the chimney's side of the ridge (object frame, metres)."""
    keep = (p_obj[:, 1] > 0.17) & ((-p_obj[:, 0] + 1.05 * p_obj[:, 1]) > 0.256 + margin) & (p_obj[:, 0] < 0.0)
    return p_obj[keep]


def chimney_positions(object_pose: np.ndarray, dataset_dir: Path) -> dict:
    cache = RUN / "chimney_position.json"
    if cache.exists():
        return json.loads(cache.read_text())
    import trimesh
    inv = np.linalg.inv(object_pose)
    out = {}
    gt = chimney_points(np.asarray(trimesh.load_mesh(str(dataset_dir / "model.glb"), force="mesh").vertices))
    out["gt"] = {"centroid": gt.mean(0).tolist(), "lo": gt.min(0).tolist(), "hi": gt.max(0).tolist()}
    for key, rel in MESHES.items():
        v = np.asarray(trimesh.load(str(RUN / rel), force="mesh", process=False).vertices)
        c = chimney_points(v @ inv[:3, :3].T + inv[:3, 3])
        out[key] = {"centroid": c.mean(0).tolist(), "lo": c.min(0).tolist(), "hi": c.max(0).tolist()}
        print(f"  {key}: {len(c)} chimney vertices")
    cache.write_text(json.dumps(out, indent=1))
    return out


def main():
    from dtc_orb_benchmark import load_dtc

    dtc = load_dtc(OBJ)
    pose, K = dtc["object_pose"], dtc["K"]
    pos = chimney_positions(pose, dtc["dir"])
    gt_c = np.array(pos["gt"]["centroid"])
    dist = {k: float(np.linalg.norm(np.array(pos[k]["centroid"]) - gt_c) * 1000) for k in MESHES}
    moved = {k: float(np.linalg.norm(np.array(pos[k]["centroid"]) - np.array(pos["init"]["centroid"])) * 1000)
             for k in ("slat", "ovoxel")}
    print("distance to the true chimney (mm):", {k: round(v, 1) for k, v in dist.items()},
          "| moved by refinement (mm):", {k: round(v, 1) for k, v in moved.items()})

    view = 6                                                    # test/im_0111: the roof from above
    E = dtc["test"][view][1]

    def project(p_obj):
        w = pose[:3, :3] @ np.asarray(p_obj) + pose[:3, 3]
        c = E[:3, :3] @ w + E[:3, 3]
        return np.array([K[0, 0] * c[0] / c[2] + K[0, 2], K[1, 1] * c[1] / c[2] + K[1, 2]]) * PANEL

    def strip(path, index):
        img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        return img[:, index * PANEL:(index + 1) * PANEL]

    panels = {"init": RUN / "slat" / "test" / f"init_{view:03d}.png",
              "slat": RUN / "slat" / "test" / f"test_0300_{view:03d}.png",
              "ovoxel": RUN / "ovoxel" / "test" / f"test_0300_{view:03d}.png"}
    gt_px = project(gt_c)
    pred_px = {k: project(pos[k]["centroid"]) for k in MESHES}
    centre = (gt_px + pred_px["init"]) / 2
    half = 150
    x0, y0 = int(max(centre[0] - half, 0)), int(max(centre[1] - half, 0))
    crop = lambda img: img[y0:y0 + 2 * half, x0:x0 + 2 * half]

    columns = [("Ground truth", None, 1, 3), ("TRELLIS.2 sample", "init", 0, 2),
               ("SLAT, 300 steps", "slat", 0, 2), ("O-voxel, 300 steps", "ovoxel", 0, 2)]

    # ---- layout, in inches -----------------------------------------------------------------------
    label_w, line = 0.22, LABEL_PT * 1.2 / 72
    tile = (WIDTH_IN - label_w - 3 * GAP_IN) / 4
    title_h, head_h = line + 0.05, line + 0.05
    plot_h = 1.9
    H = title_h + head_h + 2 * tile + GAP_IN + 0.12 + title_h + plot_h + 0.45
    fig = plt.figure(figsize=(WIDTH_IN, H))
    W = WIDTH_IN
    y = 0.0
    fig.text(0, 1 - (y + title_h / 2) / H, "(a) The roof from above (test view)", fontproperties=BOLD,
             fontsize=LABEL_PT, ha="left", va="center")
    y += title_h
    for c, (head, *_rest) in enumerate(columns):
        fig.text((label_w + c * (tile + GAP_IN) + tile / 2) / W, 1 - (y + head_h / 2) / H, head,
                 fontproperties=REGULAR, fontsize=LABEL_PT, ha="center", va="center")
    y += head_h
    outline = [pe.withStroke(linewidth=2.2, foreground="black")]
    for r, row_name in enumerate(["Colour", "Normals"]):
        fig.text((label_w / 2) / W, 1 - (y + tile / 2) / H, row_name, fontproperties=REGULAR, fontsize=LABEL_PT,
                 rotation=90, ha="center", va="center")
        for c, (head, key, colour_idx, normal_idx) in enumerate(columns):
            ax = fig.add_axes([(label_w + c * (tile + GAP_IN)) / W, 1 - (y + tile) / H, tile / W, tile / H])
            ax.axis("off")
            src = panels["init"] if key is None else panels[key]
            img = crop(strip(src, colour_idx if r == 0 else normal_idx))
            px = int(round(tile * DPI))
            img = np.asarray(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((px, px), Image.LANCZOS))
            ax.imshow(img, extent=(x0, x0 + 2 * half, y0 + 2 * half, y0), interpolation="none")
            ax.set_xlim(x0, x0 + 2 * half)
            ax.set_ylim(y0 + 2 * half, y0)
            ax.plot(*gt_px, marker="+", markersize=9, mew=1.6, color="#2ca02c")
            if key is not None:
                p = pred_px[key]
                ax.plot(*p, marker="o", markersize=4, color="#d62728")
                ax.annotate("", xy=gt_px, xytext=p, arrowprops=dict(arrowstyle="-|>", color="white", lw=1.4,
                                                                    shrinkA=3, shrinkB=5, mutation_scale=9))
                if r == 0:
                    mid = (gt_px + p) / 2 + np.array([0, -18])
                    ax.text(*mid, f"{dist[key]:.0f} mm", fontproperties=REGULAR, fontsize=SMALL_PT, color="white",
                            ha="center", va="bottom", path_effects=outline)
        y += tile + GAP_IN
    y += 0.12

    # ---- (b) footprints in the object frame ---------------------------------------------------------
    fig.text(0, 1 - (y + title_h / 2) / H, "(b) Chimney position, top view of the object", fontproperties=BOLD,
             fontsize=LABEL_PT, ha="left", va="center")
    y += title_h
    ax = fig.add_axes([0.62 / W, 1 - (y + plot_h) / H, 2.6 / W, plot_h / H])
    mm = lambda v: np.asarray(v) * 1000
    ax.add_patch(Rectangle((-86, -66), 172, 130, fill=False, ec="0.6", lw=0.8))           # roof outline
    ax.plot([-3, -3], [-66, 64], color="0.6", lw=0.8, ls="--")                           # ridge
    ax.text(-1, 58, "ridge", fontproperties=REGULAR, fontsize=SMALL_PT, color="0.45", ha="left", va="center")
    styles = {"gt": ("Ground truth", "#2ca02c", "-"), "init": ("TRELLIS.2 sample", "0.25", "--"),
              "slat": ("SLAT, 300 steps", "#d62728", "-"), "ovoxel": ("O-voxel, 300 steps", "#1f77b4", ":")}
    for key, (name, col, ls) in styles.items():
        lo, hi = mm(pos[key]["lo"]), mm(pos[key]["hi"])
        ax.add_patch(Rectangle((lo[0], lo[2]), hi[0] - lo[0], hi[2] - lo[2], fill=False, ec=col, lw=1.3, ls=ls, label=name))
    g, i = mm(gt_c), mm(pos["init"]["centroid"])
    ax.annotate("", xy=(g[0], g[2]), xytext=(i[0], i[2]),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0, shrinkA=0, shrinkB=0, mutation_scale=9))
    left = min(mm(pos[k]["lo"])[0] for k in styles)
    ax.text(left - 3, (g[2] + i[2]) / 2, f"{dist['init']:.0f} mm", fontproperties=REGULAR, fontsize=SMALL_PT,
            ha="right", va="center")
    ax.set_xlim(-90, 90)
    ax.set_ylim(-70, 68)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm), across the ridge", fontproperties=REGULAR, fontsize=SMALL_PT)
    ax.set_ylabel("z (mm), along the ridge", fontproperties=REGULAR, fontsize=SMALL_PT)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontproperties(REGULAR)
        t.set_fontsize(SMALL_PT)
    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.02), frameon=False, prop=REGULAR, fontsize=SMALL_PT,
                    handlelength=1.6, borderaxespad=0.0)
    for t in leg.get_texts():
        t.set_fontsize(SMALL_PT)
    lines = ["Distance to the true chimney:",
             f"  sample {dist['init']:.1f} mm",
             f"  SLAT {dist['slat']:.1f} mm",
             f"  o-voxel {dist['ovoxel']:.1f} mm",
             "Moved by the refinement:",
             f"  SLAT {moved['slat']:.1f} mm",
             f"  o-voxel {moved['ovoxel']:.1f} mm"]
    fig.text(3.30 / W, 1 - (y + plot_h * 0.50) / H, "\n".join(lines), fontproperties=REGULAR, fontsize=SMALL_PT,
             ha="left", va="top", linespacing=1.25)

    out = PROJECTS / "thesis_figures" / "topology_chimney"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), dpi=DPI)
    fig.savefig(out.with_suffix(".png"), dpi=DPI)
    plt.close(fig)
    print(f"wrote {out.with_suffix('.pdf')} and .png: {W:.2f} x {H:.2f} in")


if __name__ == "__main__":
    main()
