"""Benchmark the GT-supervised nvdiffrec + FlexiCubes run into its own folder, benchmark_output_v2."""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parent
sys.path.insert(0, str(HERE))

from run_all import BENCH, run, shell_command

OLD = PROJECTS / "benchmark_output"
OUT = PROJECTS / "benchmark_output_v2"
RUNS = PROJECTS / "nvdiffrec" / "out"
WORLD = OUT / "_nvdiffrec_world"
OBJECT_SCALE = 5.0      # dataset/dataset_nerf_copy.py

# stage -> (run folder, mesh folder, validate folder, image used as view, image used as albedo)
STAGES = {}
for _suffix in ("", "_it500", "_it1000", "_it2000"):       # 5000 iterations, then the short budgets
    _run = RUNS / f"flexicubes_gtsup_cluster{_suffix}"
    STAGES[f"nvdiffrec_gtsup{_suffix}_pass1"] = (_run, "dmtet_mesh", "dmtet_validate", "kd", "kd")
    STAGES[f"nvdiffrec_gtsup{_suffix}"] = (_run, "mesh", "validate", "opt", "kd")


def ready(method: str, obj: str) -> bool:
    """The training run of this object exited cleanly (runtime.json is written after train.py)."""
    runtime = STAGES[method][0] / obj / "runtime.json"
    try:
        return json.loads(runtime.read_text()).get("exit_code") == 0
    except (OSError, json.JSONDecodeError):
        return False


def link_existing():
    """GT renders and the other methods' results, linked so nothing is recomputed or rewritten."""
    OUT.mkdir(parents=True, exist_ok=True)
    for src in sorted(OLD.iterdir()):
        dst = OUT / src.name
        if src.is_dir() and not dst.exists() and src.name not in STAGES:
            dst.symlink_to(src, target_is_directory=True)


def export_world_meshes(objects, overwrite: bool):
    """nvdiffrec trains in p = S * diag(1,-1,1) * object space; undo that and apply object_pose."""
    import numpy as np
    import trimesh
    from scipy.spatial import cKDTree
    from dtc_orb_benchmark import load_dtc

    for obj in objects:
        dtc = load_dtc(obj)
        to_world = dtc["object_pose"] @ np.diag([1.0, -1.0, 1.0, 1.0]) @ np.diag([1 / OBJECT_SCALE] * 3 + [1.0])
        for method, (run_dir, mesh_dir, *_rest) in STAGES.items():
            src, dst = run_dir / obj / mesh_dir / "mesh.obj", WORLD / method / obj / "mesh.obj"
            if not ready(method, obj) or not src.exists():
                continue
            if dst.exists() and not overwrite:
                continue
            tm = trimesh.load(str(src), force="mesh", process=False)
            tm.apply_transform(to_world)
            dst.parent.mkdir(parents=True, exist_ok=True)
            tm.export(str(dst))
            gt = OLD / "_gt_orb" / obj / "mesh.obj"
            note = ""
            if gt.exists():
                g = trimesh.load(str(gt), force="mesh", process=False)
                a, b = np.asarray(tm.sample(20000)), np.asarray(g.sample(20000))
                d = 0.5 * (cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean())
                note = f", chamfer to GT {d * 1000:.2f} mm"
            print(f"[{method}] {obj}: {len(tm.vertices)} verts -> {dst}{note}", flush=True)


def spec(method: str) -> dict:
    run_dir, _mesh, val, view, albedo = STAGES[method]
    return {"mesh": str(WORLD / method / "{obj}" / "mesh.obj"), "frame": "world",
            "view": lambda obj, i: run_dir / obj / val / f"val_{i:06d}_{view}.png",
            "albedo": lambda obj, i: run_dir / obj / val / f"val_{i:06d}_{albedo}.png"}


def inner(args) -> int:
    import dtc_orb_benchmark as bench

    link_existing()
    export_world_meshes(args.objects, args.overwrite)
    if args.prepare_only:
        return 0
    renderer = bench.Renderer()
    bench.prepare_gt(args.objects, OUT, renderer, overwrite=False)      # all linked: a no-op check
    for method in args.stages:
        objects = [o for o in args.objects if ready(method, o)]
        if len(objects) < len(args.objects):
            print(f"[{method}] {len(objects)}/{len(args.objects)} object(s) trained"
                  + ("" if objects else ", skipped"), flush=True)
        if not objects:
            continue
        bench.predict(method, spec(method), objects, OUT, renderer, args.overwrite)
        bench.evaluate(method, objects, OUT)
    bench.write_summary(OUT)
    print(f"table: {OUT / 'summary.md'}")
    return 0


def main() -> int:
    from dtc_orb_benchmark import OBJECTS

    parser = argparse.ArgumentParser(description="Benchmark the GT-supervised nvdiffrec run into benchmark_output_v2")
    parser.add_argument("--stages", nargs="*", default=list(STAGES), choices=list(STAGES))
    parser.add_argument("--objects", nargs="*", default=OBJECTS)
    parser.add_argument("--overwrite", action="store_true", help="redo the nvdiffrec_gtsup* files")
    parser.add_argument("--prepare-only", action="store_true", help="meshes and links only (CPU)")
    parser.add_argument("--inner", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.inner:
        return inner(args)
    passed = [a for a in sys.argv[1:]]
    ok = run(shell_command(BENCH, [Path(__file__).resolve(), "--inner", *passed]))
    print("\nbenchmark finished" if ok else "\nBENCHMARK FAILED: nothing was scored after the error above; "
          f"{OUT / 'summary.md'} is unchanged")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
