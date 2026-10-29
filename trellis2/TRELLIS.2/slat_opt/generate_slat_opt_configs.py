"""Write one TRELLIS.2 SLAT-optimization YAML per DTC object"""
import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT.parents[1]
DATASET = PROJECTS / "dataset" / "DTC"
PROMPTS = PROJECTS / "benchmark" / "prompts.yaml"
RUNS = PROJECTS / "generative_runs"
MESH_FILES = {"recgen": "mesh.obj", "sam3d": "mesh.glb", "trellis": "mesh.glb", "trellis2": "mesh.glb"}

TEMPLATE = """\
# TRELLIS.2 SLAT / o-voxel optimization for {category}

category: "{category}"
data_dir: "{data_dir}"
gt_mesh: "{gt_mesh}"

# --- source meshes: prompt set "{tag}" of benchmark/run_all.py ({prompt})
mesh_source: recgen                    # recgen | sam3d | trellis | trellis2   (CLI --mesh-source wins)
recgen_mesh_path: "{recgen}"
sam3d_mesh_path: "{sam3d}"
trellis_mesh_path: "{trellis}"
trellis2_mesh_path: "{trellis2}"

# --- mesh -> TRELLIS.2 SLAT ------------------------------------------------------------------
encode_res: {encode_res}                       # o-voxel grid; the SLAT is at encode_res / 16 (512 -> 32^3, as the 512 pipeline)
optimize: [slat, ovoxel]               # which refinements to run, in this order

# --- alignment (TRELLIS.2 frame -> world) -----------------------------------------------------
model_name: "microsoft/TRELLIS.2-4B"
seed: 42
align_iou_views: 16
align_top_k: 4
alignment_path: ""

# --- rendering ---------------------------------------------------------------------------------
near: 0.01
far: 100

# --- losses --------------------------------------------------------------------------------------
weight: 1.0
lambda_mask: 1.0
lambda_depth: 1.2
lambda_normal: 1.0
lambda_color: 0.9
lambda_ssim: 0.2
lambda_lpips: 0.2

# --- optimization (shared) --------------------------------------------------------------------
num_iters: {num_iters}
frames_per_iter: 8
grad_clip: 0.0
display_interval: 25
eval_interval: 50
eval_views: 8

# --- SLAT refinement ------------------------------------------------------------------------------
lr: 3e-2                               # normalized latent units
optimize_shape: true
optimize_texture: true
latent_l2_weight: 0.0
loss_scaling: true

# --- o-voxel refinement ---------------------------------------------------------------------------
lr_ovoxel: 1e-2
lr_texture: 1e-2
ovoxel_optimize: [dual_vertices, split_weight, texture]
ovoxel_l2_weight: 0.0

# {{mesh_source}} is substituted, so every source keeps its own folder
out_dir: "{out_dir}"
"""


def main():
    parser = argparse.ArgumentParser(description="Generate TRELLIS.2 SLAT optimization configs")
    parser.add_argument("--prompts", type=Path, default=PROMPTS)
    parser.add_argument("--objects", nargs="*", default=None)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "slat_opt_configs")
    parser.add_argument("--output-root", type=Path, default=ROOT / "optimization_output" / "slat_opt")
    parser.add_argument("--encode-res", type=int, default=512)
    parser.add_argument("--num-iters", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    prompts = yaml.safe_load(args.prompts.read_text())
    tag = str(prompts["tag"])
    objects = args.objects or list(prompts["prompts"])
    args.config_dir.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    for obj in objects:
        path = args.config_dir / f"{obj}.yaml"
        if path.exists() and not args.overwrite:
            print(f"skip {obj}: {path} exists (use --overwrite)")
            skipped += 1
            continue
        meshes = {m: RUNS / tag / m / obj / f for m, f in MESH_FILES.items()}
        for m, p in meshes.items():
            if not p.exists():
                print(f"note  {obj}: {m} mesh not there yet ({p})")
        path.write_text(TEMPLATE.format(
            category=obj, config_name=f"slat_opt_configs/{obj}.yaml", data_dir=DATASET / obj,
            gt_mesh=DATASET / obj / "model.glb", tag=tag, prompt=prompts["prompts"][obj],
            encode_res=args.encode_res, num_iters=args.num_iters,
            out_dir=args.output_root / tag / "{mesh_source}" / obj, **{m: str(p) for m, p in meshes.items()}))
        print(f"write {path}")
        written += 1
    print(f"created {written} config(s); skipped {skipped}")


if __name__ == "__main__":
    main()
