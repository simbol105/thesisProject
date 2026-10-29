"""Write one cubefeats-optimization YAML per DTC object (TRELLIS1)."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

_spec = importlib.util.spec_from_file_location("slat_gen", REPO_ROOT / "slat_opt" / "generate_slat_opt_configs.py")
slat_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(slat_gen)

DEFAULT_CONFIG_DIR = REPO_ROOT / "cubefeats_opt_configs"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "optimization_output" / "cubefeats_opt"

CUBEFEATS_BLOCK = """\
optimize: cubefeats                     # the 101-channel decoder output @ 256^3
cubefeats_sdf_voxel_units: true         # optimize the SDF channels in voxel units
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TRELLIS1 cubefeats optimization configs")
    parser.add_argument("--dataset-root", type=Path, default=slat_gen.DEFAULT_DATASET_ROOT)
    parser.add_argument("--prompts", type=Path, default=slat_gen.DEFAULT_PROMPTS)
    parser.add_argument("--runs-root", type=Path, default=slat_gen.DEFAULT_RUNS_ROOT)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--objects", nargs="*", default=None)
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="Adam lr; SDF in voxel units, deform pre-tanh, weights / color raw")
    parser.add_argument("--num-iters", type=int, default=500)
    parser.add_argument("--frames-per-iter", type=int, default=10)
    parser.add_argument("--display-interval", type=int, default=25)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    prompts = yaml.safe_load(args.prompts.read_text())
    tag = str(prompts["tag"])
    runs = args.runs_root.expanduser().resolve() / tag
    output_root = args.output_root.expanduser().resolve() / tag
    config_dir = args.config_dir.expanduser().resolve()
    objects = args.objects or slat_gen.THESIS_OBJECTS
    written = skipped = 0

    for category in objects:
        data_dir = dataset_root / category
        config_path = config_dir / f"{category}.yaml"
        if not data_dir.is_dir() or category not in prompts["prompts"]:
            print(f"skip {category}: no dataset directory or no prompt image")
            skipped += 1
            continue
        if config_path.exists() and not args.overwrite:
            print(f"skip {category}: config already exists (use --overwrite)")
            skipped += 1
            continue

        meshes = {m: runs / m / category / f for m, f in slat_gen.MESH_FILES.items()}
        for label, path in meshes.items():
            if not path.exists():
                print(f"note  {category}: {label} mesh not there yet ({path})")
        frames = json.loads((data_dir / "transforms_train.json").read_text())["frames"]
        prompt_name = str(prompts["prompts"][category])
        prompt_idx = next(i for i, f in enumerate(frames) if Path(f["file_path"]).name == Path(prompt_name).stem)

        text = slat_gen.TEMPLATE.format(
            category=category, config_name=f"cubefeats_opt_configs/{category}.yaml",
            data_dir=data_dir, gt_mesh=data_dir / "model.glb",
            recgen_mesh=meshes["recgen"], sam3d_mesh=meshes["sam3d"],
            trellis_mesh=meshes["trellis"], trellis2_mesh=meshes["trellis2"],
            tag=tag, prompt_image=frames[prompt_idx]["file_path"], prompt_idx=prompt_idx,
            out_dir=output_root / "{mesh_source}" / category,
            lr=args.lr, num_iters=args.num_iters, frames_per_iter=args.frames_per_iter,
            display_interval=args.display_interval, eval_interval=args.eval_interval,
        )
        head, rest = text.split("optimize: slat", 1)
        rest = rest.split("\n\n", 1)[1]
        text = (head + CUBEFEATS_BLOCK + "\n" + rest)
        text = text.replace("# SLAT optimization for", "# cubefeats optimization (TRELLIS1) for")
        text = text.replace("-> [optimized] -> mesh decoder -> flexicubes",
                            "-> mesh decoder (once) -> cubefeats @ 256^3 [optimized] -> flexicubes")
        text = text.replace("backprop into the SLAT", "backprop into the cubefeats")
        text = text.replace("slat_opt/hypothesis_mesh_slat.py", "cubefeats_opt/hypothesis_mesh_cubefeats.py")
        text = text.replace("slat_opt/run_all_slat_opt.py", "cubefeats_opt/run_all_cubefeats_opt.py")
        text = text.replace("# slat: in normalized units (feats std ~ 1)", "# cubefeats: sdf in voxel units, the rest raw")
        text = text.replace("# {optimize} / {mesh_source} are substituted", "# {mesh_source} is substituted")

        config_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(text, encoding="utf-8")
        print(f"write {config_path}")
        written += 1

    print(f"created {written} config(s); skipped {skipped}")


if __name__ == "__main__":
    sys.exit(main())
