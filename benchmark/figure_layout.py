"""Image grids laid out to the thesis figure specification."""
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from PIL import Image

WIDTH_IN, MAX_HEIGHT_IN, DPI = 5.49, 8.0, 300
LABEL_PT, SMALL_PT = 9.0, 8.0
GAP_IN = 0.03
_FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
REGULAR = FontProperties(fname=str(_FONT_DIR / "cmr10.ttf"))
BOLD = FontProperties(fname=str(_FONT_DIR / "cmb10.ttf"))
matplotlib.rcParams["pdf.fonttype"] = 42          # embed the font, keep the text selectable
CHAR_EM = 0.55                                   # mean glyph width of Computer Modern, in em


def _wrap(text: str, width_in: float, size_pt: float):
    """Wrap to a width; explicit newlines are kept."""
    per_line = max(int(width_in * 72 / (CHAR_EM * size_pt)), 1)
    lines, fits = [], True
    for part in str(text).split("\n"):
        wrapped = textwrap.wrap(part, per_line, break_long_words=False) or [""]
        fits &= all(len(w) <= per_line for w in wrapped)
        lines += wrapped
    return lines, fits


def _fit(text: str, width_in: float):
    """9 pt when it fits, otherwise 8 pt (never smaller)."""
    lines, fits = _wrap(text, width_in, LABEL_PT)
    if fits:
        return lines, LABEL_PT
    return _wrap(text, width_in, SMALL_PT)[0], SMALL_PT


def figure_grid(rows, path: Path, col_labels=None, width_in: float = WIDTH_IN, max_height_in: float = MAX_HEIGHT_IN,
                panel_title: str = "", note: str = ""):
    """rows: list of (row label, cells, headers)"""
    rows = [(r[0], list(r[1]), (list(r[2]) if len(r) > 2 and r[2] else None)) for r in rows]
    if col_labels:
        rows[0] = (rows[0][0], rows[0][1], list(col_labels))
    n_cols, n_rows = max(len(r[1]) for r in rows), len(rows)
    line = lambda pt: pt * 1.2 / 72                                  # line height in inches
    title_h = line(LABEL_PT) + 0.04 if panel_title else 0.0
    note_h = line(SMALL_PT) + 0.03 if note else 0.0

    label_w, tile = line(LABEL_PT) + 0.04, 0.0
    for _ in range(3):                                              # label width and tile size depend on each other
        tile_w = (width_in - label_w - (n_cols - 1) * GAP_IN) / n_cols
        heads = []
        for _label, cells, hs in rows:
            fitted = [_fit(h, tile_w) for h in hs] if hs else []
            pt = min([f[1] for f in fitted], default=LABEL_PT)
            fitted = [(_wrap(h, tile_w, pt)[0], pt) for h in hs] if hs else []
            heads.append((fitted, max([len(f[0]) for f in fitted], default=0) * line(pt) + (0.05 if fitted else 0.0)))
        fixed = title_h + note_h + sum(h[1] for h in heads) + (n_rows - 1) * GAP_IN
        tile = min(tile_w, (max_height_in - fixed) / n_rows)
        labels = [_fit(r[0], tile) for r in rows]
        label_w = max([len(l[0]) for l in labels], default=1) * line(LABEL_PT) + 0.04
    W = label_w + n_cols * tile + (n_cols - 1) * GAP_IN
    H = fixed + n_rows * tile

    fig = plt.figure(figsize=(W, H))
    px = max(int(round(tile * DPI)), 1)
    y = title_h
    if panel_title:
        fig.text(0.0, 1 - (title_h / 2) / H, panel_title, fontproperties=BOLD, fontsize=LABEL_PT, ha="left", va="center")
    for (label, cells, _hs), (fitted, head_h), (label_lines, label_pt) in zip(rows, heads, labels):
        for c, (lines, pt) in enumerate(fitted):
            x = label_w + c * (tile + GAP_IN) + tile / 2
            fig.text(x / W, 1 - (y + head_h / 2) / H, "\n".join(lines), fontproperties=REGULAR, fontsize=pt,
                     ha="center", va="center", multialignment="center", linespacing=1.1)
        y += head_h
        fig.text((label_w - 0.04) / 2 / W, 1 - (y + tile / 2) / H, "\n".join(label_lines), fontproperties=REGULAR,
                 fontsize=label_pt, ha="center", va="center", rotation=90, multialignment="center", linespacing=1.1)
        for c, cell in enumerate(cells):
            x = label_w + c * (tile + GAP_IN)
            ax = fig.add_axes([x / W, 1 - (y + tile) / H, tile / W, tile / H])
            ax.axis("off")
            if cell is None:
                ax.text(0.5, 0.5, "n/a", fontproperties=REGULAR, fontsize=SMALL_PT, color="0.5", ha="center",
                        va="center", transform=ax.transAxes)
                continue
            img = Image.fromarray((np.clip(np.asarray(cell), 0, 1) * 255).astype(np.uint8)).resize((px, px), Image.LANCZOS)
            ax.imshow(np.asarray(img), interpolation="none", aspect="auto")
        y += tile + GAP_IN
    if note:
        fig.text(0.0, (note_h / 2) / H, note, fontproperties=REGULAR, fontsize=SMALL_PT, ha="left", va="center")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), dpi=DPI)
    fig.savefig(path.with_suffix(".png"), dpi=DPI)
    plt.close(fig)
    print(f"wrote {path.with_suffix('.pdf')} and .png: {W:.2f} x {H:.2f} in "
          f"({int(round(W * DPI))} x {int(round(H * DPI))} px), {n_rows} rows x {n_cols} columns")
    return W, H
