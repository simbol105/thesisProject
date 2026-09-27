"""thesis3d: run and inspect the latent-refinement experiments from one command line."""
from pathlib import Path
from typing import List, Optional

import typer

from . import config as C
from . import prompts as P
from . import report, runners
from .envs import ENVS, env_exists

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Latent refinement of 3D reconstructions: generate, refine, benchmark, inspect.")
results_app = typer.Typer(no_args_is_help=True, help="Inspect benchmark results.")
app.add_typer(results_app, name="results")
prompts_app = typer.Typer(no_args_is_help=True, help="Choose the prompt image each object is generated from.")
app.add_typer(prompts_app, name="prompts")

SOURCE_HELP = "recgen | sam3d | trellis | trellis2 | all"
OBJECT_HELP = ("a short name (birdhouse), a full DTC name, a unique prefix, "
               "'all' (the six thesis objects) or 'all-dataset'")
VARIANT_HELP = "slat-t1 | cubefeats-t1 | slat-t2 | ovoxel-t2 | all"


def _objects(values: Optional[List[str]]) -> List[str]:
    try:
        return C.resolve_objects(values)
    except ValueError as e:
        raise typer.BadParameter(str(e))


def _pick(values: Optional[List[str]], known: List[str], what: str) -> List[str]:
    if not values or "all" in values:
        return list(known)
    bad = [v for v in values if v not in known]
    if bad:
        raise typer.BadParameter(f"unknown {what}: {', '.join(bad)}; known: {', '.join(known)}")
    return list(values)


def _done(failed: List[str], what: str):
    if failed:
        typer.secho(f"\n{what}: {len(failed)} failed -> {', '.join(failed)}", fg="red")
        raise typer.Exit(1)
    typer.secho(f"\n{what}: finished", fg="green")


# ---------------------------------------------------------------------------------- information
@app.command()
def objects(
    dataset: bool = typer.Option(False, "--dataset", help="every DTC object present, not just the six"),
):
    """List the objects that can be used, and whether they have a prompt image."""
    if not dataset:
        print(report.render_table(["short name", "DTC object"],
                                  [[k, v] for k, v in C.THESIS_OBJECTS.items()]))
        print(f"\n{len(C.dataset_objects())} objects are available in total; --dataset lists them.")
        return
    thesis = set(C.THESIS_OBJECTS.values())
    rows = []
    for obj in C.dataset_objects():
        img = P.image_path(obj)
        rows.append([obj, "thesis" if obj in thesis else "",
                     img.stem if img else "-", str(len(P.frames(obj)))])
    print(report.render_table(["DTC object", "set", "prompt", "train views"], rows))
    print(f"\n{len(rows)} objects. Objects without a prompt need one before `generate`: "
          f"`thesis3d prompts auto <object>`.")


@app.command()
def methods():
    """List the generative sources, the optimized variables and the benchmarked methods."""
    print("sources (generative methods, also usable as refinement starting points):")
    for s in C.SOURCES:
        print(f"  {s:<10s} {C.BASELINES.get(s + '_' + C.TAG, s)}")
    print("\noptimized variables:")
    for k, v in C.VARIANTS.items():
        print(f"  {k:<14s} {v['label']:<16s} env={v['env']}")
    print("\nbenchmarked method names in the results folder:")
    for m in report.available():
        print(f"  {m:<32s} {report.label(m)}")


@app.command()
def status(
    object: List[str] = typer.Option(None, "--object", "-o", help=OBJECT_HELP),
    live: bool = typer.Option(False, help="read the live project instead of the app snapshot"),
):
    """Show what has been generated, refined and benchmarked, for the given objects."""
    objs = _objects(object)
    runs = C.PROJECT / "generative_runs" / C.TAG
    total = len(objs)
    rows = []
    for src in C.SOURCES:
        meshes = len([o for o in objs if list((runs / src / o).glob("mesh.*"))])
        cells = [src, f"{meshes}/{total}"]
        for variant, v in C.VARIANTS.items():
            pattern = str(v["cwd"] / v["mesh"])
            n = len([o for o in objs
                     if Path(pattern.format(tag=C.TAG, src=src, obj=o)).exists()])
            s = report.load(C.refined_method(variant, src), live)
            scored = len([o for o in objs if s and o in s.get("scores", {})])
            cells.append(f"{n}/{total}" + (f" ({scored} scored)" if scored else ""))
        rows.append(cells)
    print(report.render_table(["source", "generated"] + [C.VARIANTS[v]["label"] for v in C.VARIANTS], rows))
    print("\n(N scored) = objects with benchmark scores.  Results folder:", C.results_dir(live))


@app.command("env")
def env_check():
    """Check that the conda environments the experiments need exist."""
    rows = [[name, "found" if env_exists(name) else "MISSING", f"CUDA {cfg['cuda']}", cfg["note"]]
            for name, cfg in ENVS.items()]
    print(report.render_table(["env", "state", "cuda", "used for"], rows))
    print(f"\nproject: {C.PROJECT}")
    if not all(env_exists(n) for n in ENVS):
        raise typer.Exit(1)


# ---------------------------------------------------------------------------------- experiments
@app.command()
def generate(
    source: List[str] = typer.Option(None, "--source", "-s", help=SOURCE_HELP),
    object: List[str] = typer.Option(None, "--object", "-o", help=OBJECT_HELP),
    overwrite: bool = typer.Option(False, help="regenerate meshes that already exist"),
    dry_run: bool = typer.Option(False, "--dry-run", help="print the commands only"),
):
    """Run the generative methods on the prompt images (writes generative_runs/)."""
    try:
        failed = runners.generate(_pick(source, C.SOURCES, "source"), _objects(object), overwrite, dry_run)
    except RuntimeError as e:
        raise typer.BadParameter(str(e))
    _done(failed, "generate")


@app.command()
def refine(
    variable: List[str] = typer.Option(None, "--variable", "-v", help=VARIANT_HELP),
    source: List[str] = typer.Option(None, "--source", "-s", help=SOURCE_HELP),
    object: List[str] = typer.Option(None, "--object", "-o", help=OBJECT_HELP),
    check: bool = typer.Option(False, help="encode and decode only, no optimization"),
    force: bool = typer.Option(False, help="rerun runs that already finished"),
    dry_run: bool = typer.Option(False, "--dry-run", help="print the commands only"),
):
    """Refine encoded reconstructions (writes optimization_output/)."""
    _done(runners.refine(_pick(variable, list(C.VARIANTS), "variable"),
                         _pick(source, C.SOURCES, "source"), _objects(object), check, force, dry_run),
          "refine")


@app.command()
def benchmark(
    source: List[str] = typer.Option(None, "--source", "-s", help=SOURCE_HELP),
    variable: List[str] = typer.Option(None, "--variable", "-v",
                                       help=f"{VARIANT_HELP} | none (baselines only)"),
    object: List[str] = typer.Option(None, "--object", "-o", help=OBJECT_HELP),
    baselines: bool = typer.Option(True, help="also score the un-refined meshes"),
    nvdiffrec: bool = typer.Option(False, help="also score the nvdiffrec baseline"),
    overwrite: bool = typer.Option(False, help="redo existing benchmark files"),
    dry_run: bool = typer.Option(False, "--dry-run", help="print the commands only"),
):
    """Score meshes with the Stanford-ORB protocol (writes benchmark_output/)."""
    sources = _pick(source, C.SOURCES, "source")
    objs = _objects(object)
    failed = []
    if baselines:
        names = [f"{s}_{C.TAG}" for s in sources] + (["nvdiffrec"] if nvdiffrec else [])
        failed += runners.benchmark_baselines(names, objs, overwrite, dry_run)
    if not variable or "none" not in variable:
        failed += runners.benchmark_refined(_pick(variable, list(C.VARIANTS), "variable"),
                                            sources, objs, overwrite, dry_run)
    runners.summary(dry_run)
    _done(failed, "benchmark")


# ---------------------------------------------------------------------------------- prompts
@prompts_app.command("list")
def prompts_list(
    missing: bool = typer.Option(False, help="only objects that still need a prompt image"),
):
    """Show the prompt image chosen for each object."""
    data = P.load()
    rows = []
    for obj in C.dataset_objects():
        name = data.get("prompts", {}).get(obj)
        if missing and name:
            continue
        if not missing and not name:
            continue
        rows.append([obj, name or "-"])
    print(report.render_table(["DTC object", "prompt image"], rows) if rows else
          ("every object has a prompt" if missing else "no prompts set yet"))
    print(f"\nfile: {C.APP_PROMPTS}  (tag: {data.get('tag', C.TAG)})")


@prompts_app.command("show")
def prompts_show(name: str = typer.Argument(..., help=OBJECT_HELP)):
    """Print the prompt image of one object, and the views available for it."""
    obj = _objects([name])[0]
    img = P.image_path(obj)
    print(f"{obj}\n  prompt: {img if img else 'not set'}")
    views = P.camera_angles(obj)
    print(f"  training views: {len(views)} "
          f"(elevation {min(v[2] for v in views):.0f}..{max(v[2] for v in views):.0f} deg)")
    if img:
        cur = [v for v in views if v[1] == img.stem]
        if cur:
            print(f"  chosen view: frame {cur[0][0]}, elevation {cur[0][2]:.0f} deg, "
                  f"azimuth {cur[0][3]:.0f} deg")


@prompts_app.command("set")
def prompts_set(
    name: str = typer.Argument(..., help=OBJECT_HELP),
    image: str = typer.Argument(..., help="image stem (im_0077), file name, or frame index"),
):
    """Choose the prompt image for an object by name or frame index."""
    obj = _objects([name])[0]
    try:
        stem = P.resolve(obj, image)
    except ValueError as e:
        raise typer.BadParameter(str(e))
    P.set_prompt(obj, stem)
    typer.secho(f"{obj}: prompt set to {stem}  ({P.image_path(obj)})", fg="green")


@prompts_app.command("auto")
def prompts_auto(
    name: List[str] = typer.Argument(..., help=OBJECT_HELP),
    elevation: float = typer.Option(30.0, help="target camera elevation in degrees"),
    overwrite: bool = typer.Option(False, help="replace a prompt that is already set"),
):
    """Pick a raised three-quarter view automatically, as the thesis prompt set did."""
    for obj in _objects(name):
        if P.has_prompt(obj) and not overwrite:
            typer.echo(f"{obj}: keeping {P.load()['prompts'][obj]} (use --overwrite to replace)")
            continue
        idx, stem, elev, azim = P.pick_view(obj, elevation)
        P.set_prompt(obj, stem)
        typer.secho(f"{obj}: {stem} (frame {idx}, elevation {elev:.0f} deg, azimuth {azim:.0f} deg)",
                    fg="green")


@prompts_app.command("views")
def prompts_views(
    name: str = typer.Argument(..., help=OBJECT_HELP),
    limit: int = typer.Option(15, help="how many views to print, sorted by elevation"),
):
    """List an object's training views with their camera angles, to choose from."""
    obj = _objects([name])[0]
    views = sorted(P.camera_angles(obj), key=lambda v: -v[2])[:limit]
    print(report.render_table(["frame", "image", "elevation", "azimuth"],
                              [[str(i), s, f"{e:.0f}", f"{a:.0f}"] for i, s, e, a in views]))


# ---------------------------------------------------------------------------------- results
@results_app.command("summary")
def results_summary(
    metric: List[str] = typer.Option(None, "--metric", "-m", help="default: the six headline metrics"),
    source: List[str] = typer.Option(None, "--source", "-s", help=SOURCE_HELP),
    refined_only: bool = typer.Option(False, help="omit the un-refined baselines"),
    live: bool = typer.Option(False, help="read the live project instead of the app snapshot"),
):
    """One row per benchmarked method, averaged over its objects."""
    metrics = _pick(metric, list(C.METRICS), "metric") if metric else C.DEFAULT_METRICS
    sources = _pick(source, C.SOURCES, "source")
    names = [] if refined_only else [m for m in ["nvdiffrec"] + [f"{s}_{C.TAG}" for s in sources]
                                     if report.load(m, live)]
    names += [C.refined_method(v, s) for s in sources for v in C.VARIANTS
              if report.load(C.refined_method(v, s), live)]
    print(report.summary_table(names, metrics, live=live))


@results_app.command("object")
def results_object(
    name: str = typer.Argument(..., help=OBJECT_HELP),
    metric: List[str] = typer.Option(None, "--metric", "-m"),
    live: bool = typer.Option(False),
):
    """Every method's scores for one object."""
    obj = _objects([name])[0]
    metrics = _pick(metric, list(C.METRICS), "metric") if metric else C.DEFAULT_METRICS
    print(f"{C.short_name(obj)}  ({obj})\n")
    print(report.object_table(obj, report.available(live), metrics, live))


@results_app.command("method")
def results_method(
    name: str = typer.Argument(..., help="a method name from `thesis3d methods`"),
    metric: List[str] = typer.Option(None, "--metric", "-m"),
    live: bool = typer.Option(False),
):
    """One method, per object."""
    metrics = _pick(metric, list(C.METRICS), "metric") if metric else C.DEFAULT_METRICS
    print(f"{report.label(name)}  ({name})\n")
    print(report.per_object_table(name, metrics, live))


@results_app.command("compare")
def results_compare(
    source: str = typer.Argument(..., help=SOURCE_HELP),
    metric: List[str] = typer.Option(None, "--metric", "-m"),
    live: bool = typer.Option(False),
):
    """One source mesh against its four refinements."""
    src = _pick([source], C.SOURCES, "source")[0]
    metrics = _pick(metric, list(C.METRICS), "metric") if metric else C.DEFAULT_METRICS
    print(report.compare_table(src, metrics, live))


@results_app.command("tables")
def results_tables(
    out: Path = typer.Option(C.APP_ROOT / "results" / "tables.md", help="output markdown file"),
    components: Optional[Path] = typer.Option(None, help="cached connected-component counts (json)"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Regenerate the full appendix tables (markdown)."""
    raise typer.Exit(runners.make_tables(out, components, dry_run))


@results_app.command("figures")
def results_figures(
    variable: List[str] = typer.Option(None, "--variable", "-v", help=f"{VARIANT_HELP}; omit for baselines"),
    out: Path = typer.Option(C.APP_ROOT / "results" / "figures"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Regenerate the comparison images (ground truth on top, one column per object)."""
    variants = _pick(variable, list(C.VARIANTS), "variable") if variable else None
    raise typer.Exit(runners.make_figures(variants, out, dry_run))


@results_app.command("path")
def results_path(live: bool = typer.Option(False)):
    """Print where results, figures and tables are read from."""
    print(f"scores : {C.results_dir(live)}")
    print(f"figures: {C.SNAPSHOT_FIGURES if not live else C.LIVE_FIGURES}")
    print(f"tables : {C.APP_ROOT / 'results' / 'tables.md'}")


def main():
    app()


if __name__ == "__main__":
    main()
