"""Write one FlexiCubes-optimization YAML per DTC object (recgen + SAM3D mesh paths in each).

    python flexicubes/generate_flexicubes_configs.py              # the six thesis objects
    python flexicubes/generate_flexicubes_configs.py --all        # every object under dataset/DTC
    python flexicubes/generate_flexicubes_configs.py --objects Cup_B08TWHJ33Q_Tan_Lighting001
"""

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]                 # .../trellis1/TRELLIS
PROJECTS_ROOT = REPO_ROOT.parents[1]                            # .../projects

DEFAULT_DATASET_ROOT = PROJECTS_ROOT / "dataset" / "DTC"
DEFAULT_RECGEN_ROOT = PROJECTS_ROOT / "recgen" / "thesis_output"
DEFAULT_SAM3D_ROOT = PROJECTS_ROOT / "sam3d" / "sam-3d-objects" / "sam3d_output"
DEFAULT_CONFIG_DIR = REPO_ROOT / "flexicubes_configs"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "optimization_output" / "flexicubes"

# the objects of the thesis comparison (zz_notes.txt), resolved to their dataset directory names
THESIS_OBJECTS = [
    "Birdhouse_Lighting001",
    "Cup_B08TWHJ33Q_Tan_Lighting001",
    "Figurine_B08FYFNYP4_LionKing_Lighting001",
    "Gargoyle_B08SQMBDXY_HandsOnKnees_Lighting001",
    "Mallard_B082D168CK_MintGreen_Lighting001",
    "Mouse_B0CHNVBBLF_Honeycomb_1_Lighting001",
]

TEMPLATE = """\
# FlexiCubes optimization for {category}
#
#   mesh (recgen / SAM3D) -> SDF on a {grid_res}^3 cube grid -> free sdf/deform/weight/colour
#   -> FlexiCubes -> renders vs. the DTC GT mesh -> backprop. No TRELLIS model is loaded.
#
#   This is the ablation partner of cubefeats_configs/{category}.yaml: same object, alignment,
#   cameras, losses and metrics -- only the parameterization differs.
#
#   sanity check (minutes):  python flexicubes/hypothesis_mesh_flexicubes.py --yaml-config {config_name} --check
#   run:                     sbatch flexicubes/run_flexicubes.sh {config_name}
#   run on the SAM3D mesh:   sbatch flexicubes/run_flexicubes.sh {config_name} --mesh-source sam3d

category: "{category}"
data_dir: "{data_dir}"
gt_mesh: "{gt_mesh}"

# --- the reconstruction the SDF is seeded from ---------------------------------------------
mesh_source: recgen                     # recgen | sam3d   (CLI --mesh-source wins)
recgen_mesh_path: "{recgen_mesh}"
sam3d_mesh_path: "{sam3d_mesh}"
normalize_mesh: true                    # re-fit the mesh to [-0.5, 0.5]^3 before sampling the SDF

# --- mesh -> SDF ------------------------------------------------------------------------------
grid_res: {grid_res}                            # FlexiCubes grid; params scale as res^3
sdf_init: floodfill                     # floodfill: unsigned distance + exterior flood fill
                                        #   (the Sparc3D algorithm). The recgen/SAM3D meshes are
                                        #   NOT watertight, so this is the only sign test that
                                        #   recovers a solid: measured on Birdhouse it puts
                                        #   18.9% of the grid inside vs 6.8% for raycast
                                        #   (the DTC GT mesh is 24.1%).
                                        # raycast:  open3d closest-triangle-normal signed distance
                                        # sparc3d:  the exact backend the trellis2 DMTet run uses
                                        #   (needs sparc3d_path and kaolin on a GPU)
sdf_tau_scale: 1.0                      # flood fill keeps cells further than
                                        #   tau_scale * sqrt(3)/(2*res) from the surface
# sparc3d_path: "/home/fr/fr_fr/fr_ba122/projects/trellis2/TRELLIS.2/third_parties/Sparc3Dsdf/src"

optimize_weights: true                  # FlexiCubes' 21 dual weights per cube
optimize_color: true                    # per-grid-vertex colour (sigmoid inside FlexiCubes)

# --- mesh frame -> dataset world frame ------------------------------------------------------
alignment: ransac                       # ransac | similarity | bbox | identity
                                        # folded into the cameras, never applied to the grid

# --- rendering -------------------------------------------------------------------------------
near: 0.01
far: 1000
ssaa: 1

# --- optimization ------------------------------------------------------------------------------
optimizer: adam                         # adam | sgd
lr: 1e-2                                # an SDF in world units, not a latent: much smaller than
                                        #   the cubefeats lr
num_iters: {num_iters}
frames_per_iter: {frames_per_iter}
seed: 1
grad_clip: 0.0                          # 0 disables

weight: 1.5                             # image losses
sdf_weight: 0.5                         # FlexiCubes regularizer (sdf sign BCE + L_dev + weights)
tsdf_weight: 0.5                        # TSDF regression against the GT depth
tsdf_band: 3.0                          # only grid verts within this many voxels of the surface
sdf_l2_weight: 0.0                      # pull the SDF back towards its initialization
lambda_depth: 1.2
lambda_ssim: 0.2
lambda_lpips: 0.2
lambda_color: 0.9

# --- logging -------------------------------------------------------------------------------------
display_interval: {display_interval}                  # train-view triptych (pred | gt | normal) every N iters
eval_interval: {eval_interval}                     # test-view renders + metrics.jsonl every N iters
eval_views: 8

# {{mesh_source}} is substituted with recgen / sam3d, so both runs keep their own folder
out_dir: "{out_dir}"
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FlexiCubes optimization configs")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--recgen-root", type=Path, default=DEFAULT_RECGEN_ROOT)
    parser.add_argument("--sam3d-root", type=Path, default=DEFAULT_SAM3D_ROOT)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--objects", nargs="*", default=None, help="object directory names")
    parser.add_argument("--all", action="store_true", help="every object under the dataset root")
    parser.add_argument("--num-iters", type=int, default=500)
    parser.add_argument("--frames-per-iter", type=int, default=10)
    parser.add_argument("--display-interval", type=int, default=25)
    parser.add_argument("--grid-res", type=int, default=128)
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
    output_root = args.output_root.expanduser().resolve()
    written = skipped = 0

    for category in objects:
        data_dir = dataset_root / category
        if not data_dir.is_dir():
            print(f"skip {category}: no dataset directory at {data_dir}")
            skipped += 1
            continue

        recgen_mesh = args.recgen_root.expanduser().resolve() / category / "mesh.obj"
        sam3d_mesh = args.sam3d_root.expanduser().resolve() / category / "mesh.glb"
        for label, path in (("recgen", recgen_mesh), ("sam3d", sam3d_mesh)):
            if not path.exists():
                print(f"note  {category}: {label} mesh not there yet ({path})")

        config_path = config_dir / f"{category}.yaml"
        if config_path.exists() and not args.overwrite:
            print(f"skip {category}: config already exists (use --overwrite)")
            skipped += 1
            continue

        text = TEMPLATE.format(
            category=category,
            config_name=f"cubefeats_configs/{category}.yaml",
            data_dir=data_dir,
            gt_mesh=data_dir / "model.glb",
            recgen_mesh=recgen_mesh,
            sam3d_mesh=sam3d_mesh,
            out_dir=output_root / "{mesh_source}" / category,
            num_iters=args.num_iters,
            frames_per_iter=args.frames_per_iter,
            display_interval=args.display_interval,
            grid_res=args.grid_res,
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
