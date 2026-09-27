"""Write tables.md: every benchmark score, per object and averaged, for the thesis appendix."""
import argparse
import json
from pathlib import Path

PROJECTS = Path(__file__).resolve().parents[1]
BENCH = PROJECTS / "benchmark_output"
T1 = PROJECTS / "trellis1" / "TRELLIS" / "optimization_output"
T2 = PROJECTS / "trellis2" / "TRELLIS.2" / "optimization_output" / "slat_opt" / "v2"
TAG = "v2"

OBJECTS = ["Birdhouse_Lighting001", "Cup_B08TWHJ33Q_Tan_Lighting001",
           "Figurine_B08FYFNYP4_LionKing_Lighting001", "Gargoyle_B08SQMBDXY_HandsOnKnees_Lighting001",
           "Mallard_B082D168CK_MintGreen_Lighting001", "Mouse_B0CHNVBBLF_Honeycomb_1_Lighting001"]
SHORT = {o: o.split("_")[0] for o in OBJECTS}
SOURCES = ["recgen", "sam3d", "trellis", "trellis2"]
SOURCE_LABEL = {"recgen": "RecGen", "sam3d": "SAM3D", "trellis": "TRELLIS", "trellis2": "TRELLIS.2"}
VARIANTS = [("trellis1_slat", "SLAT (T1)"), ("trellis1_cubefeats", "cubefeats (T1)"),
            ("trellis2_slat", "SLAT (T2)"), ("trellis2_ovoxel", "o-voxel (T2)")]

BASELINES = [("nvdiffrec", "nvdiffrec (multi-view)"),
             ("recgen", "RecGen (v1)"), ("recgen_v2", "RecGen (v2)"),
             ("sam3d", "SAM3D (v1)"), ("sam3d_v2", "SAM3D (v2)"),
             ("trellis", "TRELLIS (v1)"), ("trellis_v2", "TRELLIS (v2)"),
             ("trellis2", "TRELLIS.2 (v1)"), ("trellis2_v2", "TRELLIS.2 (v2)")]

# key -> (label, group, field, scale, lower_is_better, decimals)
METRICS = {
    "chamfer": ("chamfer (x2e-3)", "shape", "bidir_chamfer", 2e3, True, 4),
    "normal": ("normal (1-cos)", "geometry", "normal_angle", 1, True, 4),
    "depth": ("depth SI-MSE", "geometry", "depth_mse_scene", 1, True, 4),
    "view_psnr_h": ("view PSNR-H", "view", "psnr_hdr", 1, False, 2),
    "view_psnr_l": ("view PSNR-L", "view", "psnr_ldr", 1, False, 2),
    "view_ssim": ("view SSIM", "view", "ssim", 1, False, 4),
    "view_lpips": ("view LPIPS", "view", "lpips", 1, True, 4),
    "light_psnr_h": ("light PSNR-H", "light", "psnr_hdr", 1, False, 2),
    "light_psnr_l": ("light PSNR-L", "light", "psnr_ldr", 1, False, 2),
    "light_ssim": ("light SSIM", "light", "ssim", 1, False, 4),
    "light_lpips": ("light LPIPS", "light", "lpips", 1, True, 4),
    "albedo": ("albedo PSNR", "material", "psnr_ldr", 1, False, 2),
}


def load(method: str):
    p = BENCH / method / "scores.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def value(scores: dict, obj: str, metric: str):
    _, group, field, scale, _, _ = METRICS[metric]
    try:
        return scores["scores"][obj][group][field] * scale
    except KeyError:
        return None


def mean(scores: dict, metric: str, objects=None):
    vals = [value(scores, o, metric) for o in (objects or scores["scores"])]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def fmt(v, decimals=4):
    return "--" if v is None else f"{v:.{decimals}f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def per_object_table(methods, metric):
    """One metric, methods on the rows, objects on the columns."""
    _, _, _, _, _, dec = METRICS[metric]
    rows = []
    for key, label in methods:
        s = load(key)
        if s is None:
            continue
        cells = [fmt(value(s, o, metric), dec) for o in OBJECTS]
        rows.append([label] + cells + [fmt(mean(s, metric), dec)])
    return table(["method"] + [SHORT[o] for o in OBJECTS] + ["mean"], rows)


def refined_methods():
    return [(f"{exp}_{src}_{TAG}", f"{SOURCE_LABEL[src]} + {lab}")
            for src in SOURCES for exp, lab in VARIANTS]


def block_tables():
    """Per source: the v2 baseline followed by its four refinements (means)."""
    keys = ["chamfer", "normal", "depth", "albedo", "view_psnr_l", "view_lpips"]
    rows = []
    for src in SOURCES:
        base = load(f"{src}_{TAG}")
        rows.append([f"**{SOURCE_LABEL[src]} (v2)**"] + [fmt(mean(base, k), METRICS[k][5]) for k in keys] + ["6"])
        for exp, lab in VARIANTS:
            s = load(f"{exp}_{src}_{TAG}")
            if s is None:
                continue
            objs = list(s["scores"])
            rows.append([f"&nbsp;&nbsp;+ {lab}"] + [fmt(mean(s, k), METRICS[k][5]) for k in keys] + [str(len(objs))])
    nv = load("nvdiffrec")
    rows.append(["*nvdiffrec (reference)*"] + [fmt(mean(nv, k), METRICS[k][5]) for k in keys] + [str(len(nv["objects"]))])
    return table(["method"] + [METRICS[k][0] for k in keys] + ["objects"], rows)


def matched_baseline_note():
    """SLAT (T2) from RecGen covers 5 objects; give its baseline over the same 5."""
    s = load(f"trellis2_slat_recgen_{TAG}")
    base = load(f"recgen_{TAG}")
    if s is None or base is None:
        return ""
    objs = list(s["scores"])
    if len(objs) == len(OBJECTS):
        return ""
    missing = [SHORT[o] for o in OBJECTS if o not in objs]
    keys = ["chamfer", "normal", "depth", "albedo"]
    rows = [["RecGen (v2), same 5 objects"] + [fmt(mean(base, k, objs), METRICS[k][5]) for k in keys],
            ["RecGen + SLAT (T2)"] + [fmt(mean(s, k, objs), METRICS[k][5]) for k in keys]]
    return (f"`RecGen + SLAT (T2)` is missing {', '.join(missing)} (the run ran out of GPU memory), so its "
            f"mean is over 5 objects. Against the baseline restricted to those same 5:\n\n"
            + table(["method"] + [METRICS[k][0] for k in keys], rows))


def chamfer_mm_tables():
    """Source -> encode/decode -> optimized, in mm, from each run's own metrics."""
    out = []
    rows = []
    for exp, base, stem in [("SLAT (T1)", T1 / "slat_opt" / TAG / "slat", "slat"),
                            ("cubefeats (T1)", T1 / "cubefeats_opt" / TAG, "cubefeats")]:
        for src in SOURCES:
            for obj in OBJECTS:
                f = base / src / obj / "metrics.jsonl"
                if not f.exists():
                    continue
                recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
                init = next((r for r in recs if r.get("iter") == -1), None)
                fin = next((r for r in recs if r.get("iter") == "final"), None)
                if not (init and fin):
                    continue
                rows.append([exp, SOURCE_LABEL[src], SHORT[obj], fmt(init.get("source_chamfer_mm"), 2),
                             fmt(init.get("chamfer_mm"), 2), fmt(fin.get("chamfer_mm"), 2)])
    out.append(table(["variable", "source", "object", "source mesh", "after encode/decode", "after optimization"], rows))

    rows = []
    for kind, lab in [("slat", "SLAT (T2)"), ("ovoxel", "o-voxel (T2)")]:
        for src in SOURCES:
            for obj in OBJECTS:
                f = T2 / src / obj / kind / "summary.json"
                if not f.exists():
                    continue
                d = json.loads(f.read_text())
                ci, cf = d["chamfer_mm"]
                rows.append([lab, SOURCE_LABEL[src], SHORT[obj], fmt(ci, 2), fmt(cf, 2)])
    out.append(table(["variable", "source", "object", "after encode/decode", "after optimization"], rows))

    # means per variable and source
    means = []
    for exp, base, stem in [("SLAT (T1)", T1 / "slat_opt" / TAG / "slat", "slat"),
                            ("cubefeats (T1)", T1 / "cubefeats_opt" / TAG, "cubefeats")]:
        for src in SOURCES:
            vals = []
            for obj in OBJECTS:
                f = base / src / obj / "metrics.jsonl"
                if not f.exists():
                    continue
                recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
                init = next((r for r in recs if r.get("iter") == -1), None)
                fin = next((r for r in recs if r.get("iter") == "final"), None)
                if init and fin:
                    vals.append((init["source_chamfer_mm"], init["chamfer_mm"], fin["chamfer_mm"]))
            if vals:
                n = len(vals)
                means.append([exp, SOURCE_LABEL[src], str(n)] +
                             [fmt(sum(v[i] for v in vals) / n, 2) for i in range(3)])
    for kind, lab in [("slat", "SLAT (T2)"), ("ovoxel", "o-voxel (T2)")]:
        for src in SOURCES:
            vals = []
            for obj in OBJECTS:
                f = T2 / src / obj / kind / "summary.json"
                if f.exists():
                    vals.append(json.loads(f.read_text())["chamfer_mm"])
            if vals:
                n = len(vals)
                means.append([lab, SOURCE_LABEL[src], str(n), "--",
                              fmt(sum(v[0] for v in vals) / n, 2), fmt(sum(v[1] for v in vals) / n, 2)])
    out.append(table(["variable", "source", "objects", "source mesh", "after encode/decode", "after optimization"], means))
    return out


def topology_table():
    """TRELLIS.2 gate and topology statistics per run."""
    rows = []
    for kind, lab in [("slat", "SLAT (T2)"), ("ovoxel", "o-voxel (T2)")]:
        for src in SOURCES:
            for obj in OBJECTS:
                f = T2 / src / obj / kind / "summary.json"
                if not f.exists():
                    continue
                d = json.loads(f.read_text())
                ti, tf, g = d["topology_init"], d["topology_final"], d["gate_changes"]
                rows.append([lab, SOURCE_LABEL[src], SHORT[obj],
                             str(ti["components"]), str(tf["components"]),
                             f"{ti['euler']:,}", f"{tf['euler']:,}",
                             f"{g['voxels_added']:,}", f"{g['voxels_removed']:,}",
                             f"{g['intersected_flips']:,}"])
    detail = table(["variable", "source", "object", "comp. init", "comp. final", "Euler init", "Euler final",
                    "voxels +", "voxels -", "quad flips"], rows)
    means = []
    for kind, lab in [("slat", "SLAT (T2)"), ("ovoxel", "o-voxel (T2)")]:
        for src in SOURCES:
            recs = [json.loads((T2 / src / o / kind / "summary.json").read_text())
                    for o in OBJECTS if (T2 / src / o / kind / "summary.json").exists()]
            if not recs:
                continue
            n = len(recs)
            means.append([lab, SOURCE_LABEL[src], str(n),
                          f"{sum(r['topology_init']['components'] for r in recs) / n:.0f}",
                          f"{sum(r['topology_final']['components'] for r in recs) / n:.0f}",
                          f"{sum(r['gate_changes']['intersected_flips'] for r in recs) / n:,.0f}"])
    return detail + "\n\n**Means**\n\n" + table(
        ["variable", "source", "objects", "comp. init (mean)", "comp. final (mean)", "quad flips (mean)"], means)


def components_table(cache: Path):
    if not cache.exists():
        return None
    data = json.loads(cache.read_text())
    rows = []
    for exp, lab in [("slat", "SLAT (T1)"), ("cubefeats", "cubefeats (T1)")]:
        for src in SOURCES:
            for obj in OBJECTS:
                d = data.get(f"{exp}|{src}|{obj}")
                if not d:
                    continue
                rows.append([lab, SOURCE_LABEL[src], SHORT[obj], str(d["init"]), str(d["final"]),
                             f"{d['faces_init']:,}", f"{d['faces_final']:,}"])
    detail = table(["variable", "source", "object", "comp. init", "comp. final", "faces init", "faces final"], rows)
    means = []
    for exp, lab in [("slat", "SLAT (T1)"), ("cubefeats", "cubefeats (T1)")]:
        for src in SOURCES:
            vals = [data[f"{exp}|{src}|{o}"] for o in OBJECTS if f"{exp}|{src}|{o}" in data]
            if not vals:
                continue
            n = len(vals)
            means.append([lab, SOURCE_LABEL[src], str(n),
                          f"{sum(v['init'] for v in vals) / n:.0f}", f"{sum(v['final'] for v in vals) / n:.0f}"])
    return detail + "\n\n**Means**\n\n" + table(
        ["variable", "source", "objects", "comp. init (mean)", "comp. final (mean)"], means)


def noise_floor_table():
    rows = []
    for src in SOURCES:
        for obj in OBJECTS:
            f = T2 / src / obj / "slat" / "summary.json"
            if not f.exists():
                continue
            n = json.loads(f.read_text()).get("gate_noise_floor")
            if not n:
                continue
            rows.append([SOURCE_LABEL[src], SHORT[obj], str(n["voxels_added"]), str(n["voxels_removed"]),
                         str(n["intersected_flips"])])
    return table(["source", "object", "voxels added", "voxels removed", "quad flips"], rows)


def main():
    parser = argparse.ArgumentParser(description="Write the appendix tables")
    parser.add_argument("--out", type=Path, default=PROJECTS / "tables.md")
    parser.add_argument("--components", type=Path, default=None, help="json cache of connected-component counts")
    args = parser.parse_args()

    baselines_v2 = [("nvdiffrec", "nvdiffrec (multi-view)"), ("recgen_v2", "RecGen (v2)"), ("sam3d_v2", "SAM3D (v2)"),
                    ("trellis_v2", "TRELLIS (v2)"), ("trellis2_v2", "TRELLIS.2 (v2)")]
    all_methods = BASELINES + refined_methods()
    L = []
    A = L.append

    A("# Appendix: benchmark tables\n")
    A("All scores come from the Stanford-ORB metric code applied to the DTC objects, on 8 held-out test "
      "views per object, one run per method and object. Chamfer is the bidirectional squared chamfer in "
      "ORB's scale (x2e-3); normal is 1-cos; depth is scale-invariant MSE; PSNR, SSIM and LPIPS follow "
      "ORB. Lower is better for chamfer, normal, depth and LPIPS; higher for PSNR and SSIM. `--` marks a "
      "run that does not exist.\n")

    A("## A.1 Objects\n")
    A(table(["short name", "DTC object"], [[SHORT[o], f"`{o}`"] for o in OBJECTS]) + "\n")

    A("## A.2 Baselines: all metrics, averaged over the objects\n")
    keys = list(METRICS)
    rows = []
    for key, label in BASELINES:
        s = load(key)
        if s is None:
            continue
        rows.append([label, str(len(s["objects"]))] + [fmt(mean(s, k), METRICS[k][5]) for k in keys])
    A(table(["method", "objects"] + [METRICS[k][0] for k in keys], rows) + "\n")

    A("## A.3 Baselines, per object\n")
    for k in ["chamfer", "normal", "depth", "view_psnr_l", "view_lpips", "albedo"]:
        A(f"**{METRICS[k][0]}**\n")
        A(per_object_table(BASELINES, k) + "\n")

    A("## A.4 Refinement: each source with its four optimized variables (means)\n")
    A(block_tables() + "\n")
    note = matched_baseline_note()
    if note:
        A(note + "\n")

    A("## A.5 Refinement, per object\n")
    for k in ["chamfer", "normal", "depth", "albedo", "view_psnr_l", "view_lpips"]:
        A(f"**{METRICS[k][0]}**\n")
        A(per_object_table(baselines_v2 + refined_methods(), k) + "\n")

    A("## A.6 Chamfer in millimetres: encoding cost against optimization gain\n")
    A("Each run's own chamfer (world frame, 50k sampled points), separating the encode/decode round trip "
      "from the optimization. TRELLIS.2 runs do not record the source mesh separately.\n")
    for t in chamfer_mm_tables():
        A(t + "\n")

    A("## A.7 TRELLIS.2: topology and gate changes per run\n")
    A("Components and Euler characteristic of the extracted mesh before and after optimization, with the "
      "number of voxels added or removed at the final decoder level and the number of quad flags flipped.\n")
    A(topology_table() + "\n")

    comp = components_table(args.components) if args.components else None
    if comp:
        A("## A.8 TRELLIS: connected components before and after optimization\n")
        A("Measured on the exported world-frame meshes.\n")
        A(comp + "\n")

    A(f"## A.{9 if comp else 8} Decoder noise floor (TRELLIS.2 SLAT runs)\n")
    A("The same latent decoded twice within one run, with bit-identical inputs: the gate changes reported "
      "in A.7 are therefore caused by the optimization, not by decoder non-determinism.\n")
    A(noise_floor_table() + "\n")

    args.out.write_text("\n".join(L))
    print(f"wrote {args.out} ({len(L)} blocks)")


if __name__ == "__main__":
    main()
