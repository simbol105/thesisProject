"""Write one SLAT-optimization YAML per DTC object"""

import argparse
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECTS_ROOT = REPO_ROOT.parents[1]

DEFAULT_DATASET_ROOT = PROJECTS_ROOT / "dataset" / "DTC"
DEFAULT_PROMPTS = PROJECTS_ROOT / "benchmark" / "prompts.yaml"
DEFAULT_RUNS_ROOT = PROJECTS_ROOT / "generative_runs"
MESH_FILES = {"recgen": "mesh.obj", "sam3d": "mesh.glb", "trellis": "mesh.glb", "trellis2": "mesh.glb"}
DEFAULT_CONFIG_DIR = REPO_ROOT / "slat_opt_configs"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "optimization_output" / "slat_opt"

THESIS_OBJECTS = [
    "Birdhouse_Lighting001",
    "Cup_B08TWHJ33Q_Tan_Lighting001",
    "Figurine_B08FYFNYP4_LionKing_Lighting001",
    "Gargoyle_B08SQMBDXY_HandsOnKnees_Lighting001",
    "Mallard_B082D168CK_MintGreen_Lighting001",
    "Mouse_B0CHNVBBLF_Honeycomb_1_Lighting001",
]

TEMPLATE = """\
# SLAT optimization for {category}

category: "{category}"
data_dir: "{data_dir}"
gt_mesh: "{gt_mesh}"

# --- what is optimized ------------------------------------------------------------------------
optimize: slat                          # slat: the 8-channel SLAT feats (normalized)

# --- the reconstruction the SLAT is encoded from -------------------------------------------
mesh_source: recgen                     # recgen | sam3d | trellis | trellis2   (CLI --mesh-source wins)
recgen_mesh_path: "{recgen_mesh}"
sam3d_mesh_path: "{sam3d_mesh}"
trellis_mesh_path: "{trellis_mesh}"
trellis2_mesh_path: "{trellis2_mesh}"
prompt_image: "{prompt_image}"
prompt_idx: {prompt_idx}
normalize_mesh: true                    # re-fit the mesh to [-0.5, 0.5]^3 before voxelizing
fill_interior: false                    # true = flood-fill the interior

# --- mesh -> SLAT --------------------------------------------------------------------------
feature_cameras: trellis                # trellis: TRELLIS' own training rig
feature_source: render                  # render: DINOv2 on renders of the source mesh
feature_views: 150
dino_batch: 8
encoder_ckpt: "microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16"
model_name: "microsoft/TRELLIS-image-large"

# --- mesh frame -> dataset world frame ------------------------------------------------------
alignment: search                       # search: 24 axis rotations + ICP with scale
align_iou_views: 16
align_top_k: 4
alignment_path: ""                      # a saved alignment.npy skips the estimation

# --- rendering -------------------------------------------------------------------------------
near: 0.01
far: 1000
ssaa: 1

# --- optimization ------------------------------------------------------------------------------
optimizer: adam                         # adam | sgd
lr: {lr}                                # slat: in normalized units (feats std ~ 1)
num_iters: {num_iters}
frames_per_iter: {frames_per_iter}
seed: 1
grad_clip: 0.0                          # 0 disables
loss_scaling: true                      # GradScaler; the mesh decoder runs in fp16

weight: 1.5                             # image losses
sdf_weight: 0.5                         # flexicubes reg loss
tsdf_weight: 0.5                        # TSDF regression against the GT depth
latent_l2_weight: 0.0                   # pull the variable back towards its initial value
lambda_mask: 1.0
lambda_depth: 1.2
lambda_normal: 1.0
lambda_color: 0.9
lambda_ssim: 0.2
lambda_lpips: 0.2

# --- logging -------------------------------------------------------------------------------------
display_interval: {display_interval}                  # train-view panel every N iters
eval_interval: {eval_interval}                     # test-view renders + metrics.jsonl every N iters
eval_views: 8

# {{optimize}} / {{mesh_source}} are substituted, so every combination keeps its own folder
out_dir: "{out_dir}"
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SLAT optimization configs")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS,
                        help="prompts file of benchmark/run_all.py (its tag selects generative_runs/<tag>/)")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--objects", nargs="*", default=None, help="object directory names")
    parser.add_argument("--all", action="store_true", help="every object under the dataset root")
    parser.add_argument("--lr", type=float, default=6e-2)
    parser.add_argument("--num-iters", type=int, default=500)
    parser.add_argument("--frames-per-iter", type=int, default=10)
    parser.add_argument("--display-interval", type=int, default=25)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    if args.all:
        objects = sorted(p.name for p in dataset_root.iterdir() if p.is_dir())
    else:
        objects = args.objects or THESIS_OBJECTS

    config_dir = args.config_dir.expanduser().resolve()
    prompts = yaml.safe_load(args.prompts.read_text())
    tag = str(prompts["tag"])
    runs = args.runs_root.expanduser().resolve() / tag
    output_root = args.output_root.expanduser().resolve() / tag
    written = skipped = 0

    for category in objects:
        data_dir = dataset_root / category
        if not data_dir.is_dir():
            print(f"skip {category}: no dataset directory at {data_dir}")
            skipped += 1
            continue

        meshes = {m: runs / m / category / f for m, f in MESH_FILES.items()}
        for label, path in meshes.items():
            if not path.exists():
                print(f"note  {category}: {label} mesh not there yet ({path})")
        if category not in prompts["prompts"]:
            print(f"skip {category}: no prompt image in {args.prompts}")
            skipped += 1
            continue
        prompt_name = str(prompts["prompts"][category])
        frames = json.load(open(data_dir / "transforms_train.json"))["frames"]
        prompt_idx = next(i for i, f in enumerate(frames) if Path(f["file_path"]).name == Path(prompt_name).stem)

        config_path = config_dir / f"{category}.yaml"
        if config_path.exists() and not args.overwrite:
            print(f"skip {category}: config already exists (use --overwrite)")
            skipped += 1
            continue

        text = TEMPLATE.format(
            category=category,
            config_name=f"slat_opt_configs/{category}.yaml",
            data_dir=data_dir,
            gt_mesh=data_dir / "model.glb",
            recgen_mesh=meshes["recgen"],
            sam3d_mesh=meshes["sam3d"],
            trellis_mesh=meshes["trellis"],
            trellis2_mesh=meshes["trellis2"],
            tag=tag,
            prompt_image=frames[prompt_idx]["file_path"],
            prompt_idx=prompt_idx,
            out_dir=output_root / "{optimize}" / "{mesh_source}" / category,
            lr=args.lr,
            num_iters=args.num_iters,
            frames_per_iter=args.frames_per_iter,
            display_interval=args.display_interval,
            eval_interval=args.eval_interval,
        )
        if args.dry_run:
            print(f"would write {config_path}")
        else:
            config_dir.mkdir(parents=True, exist_ok=True)
            config_path.write_text(text, encoding="utf-8")
            print(f"write {config_path}")
        written += 1

    print(f"{'would create' if args.dry_run else 'created'} {written} config(s); skipped {skipped}")


if __name__ == "__main__":
    main()
