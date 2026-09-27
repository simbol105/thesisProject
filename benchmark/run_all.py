"""Regenerate every generative method from the prompt images in a prompts file, then benchmark them."""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parent

GENERATORS = {
    "recgen": {"env": "recgen", "cuda": "11.8", "cwd": PROJECTS / "recgen",
               "exports": ['PIXI_CACHE_DIR="/scratch/pixi-cache-$USER"', 'UV_CACHE_DIR="/scratch/uv-cache-$USER"'],
               "python": "pixi run python",
               "script": HERE / "generate_recgen.py", "mesh": "mesh.obj"},
    "sam3d": {"env": "sam3d", "cuda": "12.8", "cwd": PROJECTS / "sam3d" / "sam-3d-objects",
              "script": HERE / "generate_sam3d.py", "mesh": "mesh.glb"},
    "trellis": {"env": "trellis", "cuda": "12.8", "cwd": PROJECTS / "trellis1" / "TRELLIS",
                "exports": ['PATH="/opt/bwhpc/common/devel/cuda/12.8/bin:$PATH"'],
                "script": HERE / "generate_trellis.py", "mesh": "mesh.glb"},
    "trellis2": {"env": "trellis2", "cuda": "12.8", "cwd": PROJECTS / "trellis2" / "TRELLIS.2",
                 "script": HERE / "generate_trellis2.py", "mesh": "mesh.glb"},
}
BENCH = {"env": "trellis2", "cuda": "12.8", "cwd": PROJECTS}


def shell_command(step: dict, args: list) -> list:
    """bash command that activates the step's environment like run_script.sh and runs `args`."""
    q = lambda x: shlex.quote(str(x))
    python = step.get("python", '"$CONDA_PREFIX/bin/python"')
    lines = [
        "source ~/miniforge3/etc/profile.d/conda.sh",
        "source ~/.bashrc >/dev/null 2>&1 || true",
        "type module >/dev/null 2>&1 || source /etc/profile >/dev/null 2>&1 || true",
        f"module load devel/cuda/{step['cuda']}",
        f"conda activate {step['env']}",
        *[f"export {e}" for e in step.get("exports", [])],
        f"cd {q(step['cwd'])}",
        'echo "[env] CONDA_PREFIX=$CONDA_PREFIX CUDA_HOME=$CUDA_HOME"',
        f"{python} " + " ".join(q(a) for a in args),
    ]
    return ["bash", "-c", "\n".join(lines)]


def run(cmd, cwd=None) -> bool:
    print("\n$ " + (cmd[-1] if cmd[:2] == ["bash", "-c"] else " ".join(str(c) for c in cmd)), flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=cwd, check=False).returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate all generative methods and benchmark them")
    parser.add_argument("--prompts", type=Path, default=HERE / "prompts.yaml")
    parser.add_argument("--methods", nargs="*", default=list(GENERATORS), choices=list(GENERATORS))
    parser.add_argument("--objects", nargs="*", default=None, help="default: every object in the prompts file")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--no-nvdiffrec", action="store_true", help="do not (re)score the nvdiffrec baseline")
    parser.add_argument("--overwrite", action="store_true", help="regenerate even if the outputs exist")
    args = parser.parse_args()

    prompts_file = args.prompts.resolve()
    spec = yaml.safe_load(prompts_file.read_text())
    tag = str(spec["tag"])
    objects = args.objects or list(spec["prompts"])
    runs = PROJECTS / "generative_runs" / tag
    print(f"prompt set '{tag}' from {prompts_file}")
    for obj in objects:
        print(f"  {obj}: {spec['prompts'][obj]}")

    failed = []
    if not args.skip_generate:
        for method in args.methods:
            g = GENERATORS[method]
            cmd = shell_command(g, [g["script"], "--prompt-source", prompts_file, "--out-root", runs / method,
                                    "--objects", *objects] + (["--overwrite"] if args.overwrite else []))
            if not run(cmd):
                failed.append(f"generate {method}")

    if not args.skip_benchmark:
        bench = [HERE / "dtc_orb_benchmark.py"]
        for method in args.methods:
            mesh = runs / method / "{obj}" / GENERATORS[method]["mesh"]
            if not run(shell_command(BENCH, bench + ["all", "--objects", *objects, "--custom-method", f"{method}_{tag}",
                                                     "--custom-mesh", mesh, "--custom-frame", "align"])):
                failed.append(f"benchmark {method}")
        if not args.no_nvdiffrec and not run(shell_command(BENCH, bench + ["all", "--methods", "nvdiffrec"])):
            failed.append("benchmark nvdiffrec")
        run(shell_command(BENCH, bench + ["summary"]))

    print("\n" + ("all steps finished" if not failed else "failed steps: " + ", ".join(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
