"""Read scores.json files and print result tables."""
import json
from pathlib import Path
from typing import Dict, List, Optional

from . import config as C


def load(method: str, live: bool = False) -> Optional[Dict]:
    p = C.results_dir(live) / method / "scores.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def available(live: bool = False) -> List[str]:
    root = C.results_dir(live)
    return sorted(p.parent.name for p in root.glob("*/scores.json")) if root.is_dir() else []


def value(scores: Dict, obj: str, metric: str) -> Optional[float]:
    _, group, field, scale, _, _ = C.METRICS[metric]
    try:
        return scores["scores"][obj][group][field] * scale
    except KeyError:
        return None


def mean(scores: Dict, metric: str, objects: Optional[List[str]] = None) -> Optional[float]:
    vals = [value(scores, o, metric) for o in (objects or scores["scores"])]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def label(method: str) -> str:
    if method in C.BASELINES:
        return C.BASELINES[method]
    for variant, key in C.EXPERIMENT_KEY.items():
        for src in C.SOURCES:
            if method == f"{key}_{src}_{C.TAG}":
                return f"{C.BASELINES.get(src + '_' + C.TAG, src)} + {C.VARIANTS[variant]['label']}"
    return method


def fmt(v: Optional[float], decimals: int = 4) -> str:
    return "--" if v is None else f"{v:.{decimals}f}"


def render_table(header: List[str], rows: List[List[str]], widths: Optional[List[int]] = None) -> str:
    """Plain text table, aligned; markdown-free so it reads well in a terminal."""
    cols = len(header)
    widths = widths or [max(len(str(r[i])) for r in [header] + rows) for i in range(cols)]

    def numeric(i):                      # right-align a column only when it holds numbers
        vals = [str(r[i]).rstrip(" +-") for r in rows]
        return all(v.replace(".", "", 1).replace("--", "").lstrip("-").isdigit() or v in ("", "--")
                   for v in vals)

    align = [numeric(i) for i in range(cols)]
    def cell(text, i):
        return str(text).rjust(widths[i]) if align[i] else str(text).ljust(widths[i])

    out = ["  ".join(cell(h, i) for i, h in enumerate(header)),
           "  ".join("-" * widths[i] for i in range(cols))]
    for r in rows:
        out.append("  ".join(cell(c, i) for i, c in enumerate(r)))
    return "\n".join(out)


def summary_table(methods: List[str], metrics: List[str], objects: Optional[List[str]] = None,
                  live: bool = False) -> str:
    rows = []
    for m in methods:
        s = load(m, live)
        if s is None:
            continue
        objs = objects or list(s["scores"])
        n = len([o for o in objs if o in s["scores"]])
        rows.append([label(m), str(n)] + [fmt(mean(s, k, objs), C.METRICS[k][5]) for k in metrics])
    if not rows:
        return "no results found"
    return render_table(["method", "obj"] + [C.METRICS[k][0] for k in metrics], rows)


def object_table(obj: str, methods: List[str], metrics: List[str], live: bool = False) -> str:
    rows = []
    for m in methods:
        s = load(m, live)
        if s is None or obj not in s.get("scores", {}):
            continue
        rows.append([label(m)] + [fmt(value(s, obj, k), C.METRICS[k][5]) for k in metrics])
    if not rows:
        return f"no results for {C.short_name(obj)}"
    return render_table(["method"] + [C.METRICS[k][0] for k in metrics], rows)


def per_object_table(method: str, metrics: List[str], live: bool = False) -> str:
    s = load(method, live)
    if s is None:
        return f"no results for {method}"
    rows = []
    for obj in sorted(s["scores"]):
        rows.append([C.short_name(obj)] + [fmt(value(s, obj, k), C.METRICS[k][5]) for k in metrics])
    rows.append(["mean"] + [fmt(mean(s, k), C.METRICS[k][5]) for k in metrics])
    return render_table(["object"] + [C.METRICS[k][0] for k in metrics], rows)


def compare_table(source: str, metrics: List[str], live: bool = False) -> str:
    """One source: its baseline mesh and the four refinements, with the delta."""
    base_key = f"{source}_{C.TAG}"
    base = load(base_key, live)
    if base is None:
        return f"no baseline results for {base_key}"
    rows = [[label(base_key)] + [fmt(mean(base, k), C.METRICS[k][5]) for k in metrics]]
    for variant in C.VARIANTS:
        m = C.refined_method(variant, source)
        s = load(m, live)
        if s is None:
            continue
        objs = list(s["scores"])
        cells = []
        for k in metrics:
            v, b = mean(s, k, objs), mean(base, k, objs)
            lower_better = C.METRICS[k][4]
            arrow = "" if v is None or b is None else (
                " +" if (v < b) == lower_better else " -")
            cells.append(fmt(v, C.METRICS[k][5]) + arrow)
        rows.append([f"  + {C.VARIANTS[variant]['label']} ({len(objs)} obj)"] + cells)
    return render_table(["method"] + [C.METRICS[k][0] for k in metrics], rows) + \
        "\n\n'+' = better than the baseline on the objects that variant covers, '-' = worse."
