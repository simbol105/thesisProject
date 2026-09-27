"""Benchmark every latent-optimization result, the way run_all.py benchmarks the generative methods."""
import argparse
from pathlib import Path

import yaml

from run_all import BENCH, run, shell_command

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parent
SOURCES = ["recgen", "sam3d", "trellis", "trellis2"]

EXPERIMENTS = {
    "trellis1_slat": PROJECTS / "trellis1" / "TRELLIS" / "optimization_output" / "slat_opt" / "{tag}" / "slat"
                     / "{src}" / "{obj}" / "final_mesh" / "slat_opt_world.glb",
    "trellis1_cubefeats": PROJECTS / "trellis1" / "TRELLIS" / "optimization_output" / "cubefeats_opt" / "{tag}"
                          / "{src}" / "{obj}" / "final_mesh" / "cubefeats_opt_world.glb",
    "trellis2_slat": PROJECTS / "trellis2" / "TRELLIS.2" / "optimization_output" / "slat_opt" / "{tag}"
                     / "{src}" / "{obj}" / "slat" / "mesh_final_world.ply",
    "trellis2_ovoxel": PROJECTS / "trellis2" / "TRELLIS.2" / "optimization_output" / "slat_opt" / "{tag}"
                       / "{src}" / "{obj}" / "ovoxel" / "mesh_final_world.ply",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark every latent-optimization result")
    parser.add_argument("--prompts", type=Path, default=HERE / "prompts.yaml", help="its tag selects the runs")
    parser.add_argument("--experiments", nargs="*", default=list(EXPERIMENTS), choices=list(EXPERIMENTS))
    parser.add_argument("--sources", nargs="*", default=SOURCES, choices=SOURCES)
    parser.add_argument("--objects", nargs="*", default=None, help="default: every object in the prompts file")
    parser.add_argument("--overwrite", action="store_true", help="redo existing benchmark files")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--list", action="store_true", help="only print the methods and their meshes")
    args = parser.parse_args()

    spec = yaml.safe_load(args.prompts.resolve().read_text())
    tag = str(spec["tag"])
    objects = args.objects or list(spec["prompts"])

    methods = []
    for exp in args.experiments:
        for src in args.sources:
            mesh = str(EXPERIMENTS[exp]).format(tag=tag, src=src, obj="{obj}")
            found = [o for o in objects if Path(mesh.format(obj=o)).exists()]
            name = f"{exp}_{src}_{tag}"
            methods.append((name, mesh, found))
            print(f"{name:<34s} {len(found)}/{len(objects)} mesh(es)  {mesh}")
    if args.list:
        return 0

    failed = []
    bench = [HERE / "dtc_orb_benchmark.py"]
    for name, mesh, found in methods:
        if not found:
            print(f"\nskip {name}: no meshes")
            continue
        cmd = bench + ["all", "--objects", *found, "--custom-method", name,
                       "--custom-mesh", mesh, "--custom-frame", "world"]
        if not run(shell_command(BENCH, cmd + (["--overwrite"] if args.overwrite else []))):
            failed.append(name)
    if not args.skip_summary:
        run(shell_command(BENCH, bench + ["summary"]))

    print("\n" + ("all methods finished" if not failed else "failed: " + ", ".join(failed)))
    print(f"table: {PROJECTS / 'benchmark_output' / 'summary.md'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
