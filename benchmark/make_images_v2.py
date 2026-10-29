"""Recreate every comparison image into benchmark_images_v2, from benchmark_output_v2."""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parent
BENCH = PROJECTS / "benchmark_output_v2"
OUT = PROJECTS / "benchmark_images_v2"
EXPERIMENTS = ["trellis1_slat", "trellis1_cubefeats", "trellis2_slat", "trellis2_ovoxel"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Recreate all comparison images for the v2 benchmark")
    parser.add_argument("--nvdiffrec", default="nvdiffrec_gtsup",
                        choices=["nvdiffrec_gtsup", "nvdiffrec_gtsup_pass1"], help="row used as nvdiffrec")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--thesis", action="store_true",
                        help="thesis figure specification (5.49 in wide, 9 pt Computer Modern, PDF and 300 dpi PNG) "
                             "into thesis_figures/ instead of the working sheets")
    args = parser.parse_args()

    missing = [o.name for o in (BENCH / "_gt_orb").iterdir() if o.is_dir() and not any((o / "albedo").glob("*.png"))]
    if missing:
        print(f"no unlit GT render for {missing}; run: python benchmark/dtc_orb_benchmark.py prepare-gt")
        return 1

    nv = args.nvdiffrec
    args.out = args.out or (PROJECTS / "thesis_figures" if args.thesis else OUT)
    its = [it for it in ("_it500", "_it1000", "_it2000") if (BENCH / f"nvdiffrec_gtsup{it}_pass1").exists()]
    short = [f"nvdiffrec_gtsup{it}_pass1" for it in its]                         # short-budget rows
    budget = [f"nvdiffrec_gtsup{it}{stage}" for it in (*its, "") for stage in ("_pass1", "")]
    base = [sys.executable, HERE / "make_tiles.py", "--bench", BENCH, "--out", args.out, "--gt", "albedo", "--crop", "gt"]
    if args.thesis:
        base += ["--style", "thesis"]
    jobs = [
        ["--methods", *short, nv, "recgen_v2", "sam3d_v2", "trellis_v2", "trellis2_v2"],              # per object
        ["--overview", "--methods", *short, nv, "recgen_v2", "sam3d_v2", "trellis_v2", "trellis2_v2"],
        ["--overview", "--methods", *budget, "--name", "overview_nvdiffrec_budget",
         "--title", "nvdiffrec at increasing iteration budgets"],
        ["--overview", "--methods", nv, "recgen", "sam3d", "trellis", "trellis2", "--name", "overview_v1"],
        ["--overview", "--experiments", *EXPERIMENTS],
    ]
    failed = sum(subprocess.run([str(c) for c in base + job], check=False).returncode != 0 for job in jobs)
    prompts = PROJECTS / "benchmark_images" / "prompts_v1_v2.png"          # no GT or nvdiffrec in it
    if prompts.exists() and not args.thesis:
        args.out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(prompts, args.out / prompts.name)
    print(f"\nimages: {args.out}" + (f"  ({failed} step(s) failed)" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
