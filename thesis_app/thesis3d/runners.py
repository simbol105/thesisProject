"""Generation, refinement and benchmarking: build the commands and run them."""
from pathlib import Path
from typing import List, Optional

from . import config as C
from . import prompts as P
from .envs import run, shell_command

GENERATORS = {
    "recgen": {"env": "recgen", "cwd": C.PROJECT / "recgen", "python": "pixi run python",
               "script": C.BENCHMARK_DIR / "generate_recgen.py", "mesh": "mesh.obj",
               "exports": ['PIXI_CACHE_DIR="/scratch/pixi-cache-$USER"', 'UV_CACHE_DIR="/scratch/uv-cache-$USER"']},
    "sam3d": {"env": "sam3d", "cwd": C.PROJECT / "sam3d" / "sam-3d-objects",
              "script": C.BENCHMARK_DIR / "generate_sam3d.py", "mesh": "mesh.glb"},
    "trellis": {"env": "trellis", "cwd": C.PROJECT / "trellis1" / "TRELLIS",
                "script": C.BENCHMARK_DIR / "generate_trellis.py", "mesh": "mesh.glb",
                "exports": ['PATH="/opt/bwhpc/common/devel/cuda/12.8/bin:$PATH"']},
    "trellis2": {"env": "trellis2", "cwd": C.PROJECT / "trellis2" / "TRELLIS.2",
                 "script": C.BENCHMARK_DIR / "generate_trellis2.py", "mesh": "mesh.glb"},
}
BENCH_ENV = {"env": "trellis2", "cwd": C.PROJECT}


def generate(sources: List[str], objects: List[str], overwrite: bool, dry_run: bool) -> List[str]:
    """Run each generative method on the given objects (benchmark/generate_*.py)."""
    runs_root = C.PROJECT / "generative_runs" / C.TAG
    P.load()                                   # make sure the app's prompt file exists
    missing = [o for o in objects if not P.has_prompt(o)]
    if missing:
        names = ", ".join(C.short_name(o) for o in missing)
        raise RuntimeError(f"no prompt image for {names}; set one with "
                           f"`thesis3d prompts auto {C.short_name(missing[0])}` or `prompts set ...`")
    failed = []
    for src in sources:
        g = GENERATORS[src]
        args = [g["script"], "--prompt-source", C.APP_PROMPTS,
                "--out-root", runs_root / src, "--objects", *objects]
        args += ["--overwrite"] if overwrite else []
        cmd = shell_command(g["env"], g["cwd"], args, g.get("python"), g.get("exports"))
        if run(cmd, dry_run) != 0:
            failed.append(src)
    return failed


def refine(variants: List[str], sources: List[str], objects: List[str], check: bool,
           force: bool, dry_run: bool) -> List[str]:
    """Run the latent/explicit refinements (the slat_opt and cubefeats_opt runners)."""
    failed = []
    for variant in variants:
        v = C.VARIANTS[variant]
        stems = [Path(o).name for o in objects]
        cfg_args = [v["configs"], "--prompts", C.APP_PROMPTS, "--objects", *stems, "--overwrite"]
        run(shell_command(v["env"], v["cwd"], cfg_args), dry_run)
        if v["multi_source"]:
            args = [v["runner"], "--mesh-source", *sources, "--objects", *stems]
            args += ["--only", v["only"]] if v.get("only") else []
            args += ["--check"] if check else []
            args += ["--force"] if force else []
            if run(shell_command(v["env"], v["cwd"], args), dry_run) != 0:
                failed.append(variant)
        else:
            for src in sources:                       # this runner takes one source per call
                args = [v["runner"], "--mesh-source", src, "--objects", *stems]
                args += ["--check"] if check else []
                args += ["--force"] if force else []
                if run(shell_command(v["env"], v["cwd"], args), dry_run) != 0:
                    failed.append(f"{variant}/{src}")
    return failed


def benchmark_baselines(methods: List[str], objects: List[str], overwrite: bool, dry_run: bool) -> List[str]:
    """Score un-refined meshes: built-in methods (nvdiffrec) or a generated prompt set."""
    bench = C.BENCHMARK_DIR / "dtc_orb_benchmark.py"
    runs_root = C.PROJECT / "generative_runs" / C.TAG
    failed = []
    for m in methods:
        if m == "nvdiffrec":
            args = [bench, "all", "--methods", "nvdiffrec", "--objects", *objects]
        else:
            src = m.replace(f"_{C.TAG}", "")
            mesh = runs_root / src / "{obj}" / GENERATORS[src]["mesh"]
            args = [bench, "all", "--objects", *objects, "--custom-method", m,
                    "--custom-mesh", mesh, "--custom-frame", "align"]
        args += ["--overwrite"] if overwrite else []
        if run(shell_command(BENCH_ENV["env"], BENCH_ENV["cwd"], args), dry_run) != 0:
            failed.append(m)
    return failed


def benchmark_refined(variants: List[str], sources: List[str], objects: List[str],
                      overwrite: bool, dry_run: bool) -> List[str]:
    """Score refined meshes (already in the DTC world frame)."""
    bench = C.BENCHMARK_DIR / "dtc_orb_benchmark.py"
    failed = []
    for variant in variants:
        pattern = C.VARIANTS[variant]["mesh"]
        for src in sources:
            name = C.refined_method(variant, src)
            mesh = C.VARIANTS[variant]["cwd"] / pattern.format(tag=C.TAG, src=src, obj="{obj}")
            present = [o for o in objects if Path(str(mesh).format(obj=o)).exists()]
            if not present:
                print(f"skip {name}: no refined meshes yet")
                continue
            args = [bench, "all", "--objects", *present, "--custom-method", name,
                    "--custom-mesh", mesh, "--custom-frame", "world"]
            args += ["--overwrite"] if overwrite else []
            if run(shell_command(BENCH_ENV["env"], BENCH_ENV["cwd"], args), dry_run) != 0:
                failed.append(name)
    return failed


def summary(dry_run: bool = False) -> int:
    return run(shell_command(BENCH_ENV["env"], BENCH_ENV["cwd"],
                             [C.BENCHMARK_DIR / "dtc_orb_benchmark.py", "summary"]), dry_run)


def make_tables(out: Path, components: Optional[Path], dry_run: bool) -> int:
    args = [C.BENCHMARK_DIR / "make_tables.py", "--out", out]
    args += ["--components", components] if components else []
    return run(shell_command(BENCH_ENV["env"], BENCH_ENV["cwd"], args), dry_run)


def make_figures(experiments: Optional[List[str]], out: Path, dry_run: bool) -> int:
    args = [C.BENCHMARK_DIR / "make_tiles.py", "--overview", "--out", out]
    if experiments:
        args += ["--experiments", *[C.EXPERIMENT_KEY[e] for e in experiments]]
    return run(shell_command(BENCH_ENV["env"], BENCH_ENV["cwd"], args), dry_run)
