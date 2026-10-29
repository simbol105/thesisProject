"""Paths, object names and method definitions shared by the CLI commands."""
import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.environ.get("THESIS3D_PROJECT", APP_ROOT.parent)).resolve()

BENCHMARK_DIR = PROJECT / "benchmark"
DATASET = PROJECT / "dataset" / "DTC"
APP_PROMPTS = APP_ROOT / "prompts.yaml"          # the app's own prompt list (never edits the project's)
LIVE_RESULTS = PROJECT / "benchmark_output"
LIVE_FIGURES = PROJECT / "benchmark_images"
SNAPSHOT_RESULTS = APP_ROOT / "results" / "scores"
SNAPSHOT_FIGURES = APP_ROOT / "results" / "figures"
TAG = "v2"

# the six objects of the thesis; the dataset holds more (see dataset_objects())
THESIS_OBJECTS = {
    "birdhouse": "Birdhouse_Lighting001",
    "cup": "Cup_B08TWHJ33Q_Tan_Lighting001",
    "figurine": "Figurine_B08FYFNYP4_LionKing_Lighting001",
    "gargoyle": "Gargoyle_B08SQMBDXY_HandsOnKnees_Lighting001",
    "mallard": "Mallard_B082D168CK_MintGreen_Lighting001",
    "mouse": "Mouse_B0CHNVBBLF_Honeycomb_1_Lighting001",
}

SOURCES = ["recgen", "sam3d", "trellis", "trellis2"]

# optimized variable -> (label, model repo, runner, config generator, config dir, mesh pattern)
VARIANTS = {
    "slat-t1": {
        "label": "SLAT (T1)", "env": "trellis", "cwd": PROJECT / "trellis1" / "TRELLIS",
        "configs": "slat_opt/generate_slat_opt_configs.py", "config_dir": "slat_opt_configs",
        "runner": "slat_opt/run_all_slat_opt.py", "multi_source": False,
        "mesh": "optimization_output/slat_opt/{tag}/slat/{src}/{obj}/final_mesh/slat_opt_world.glb",
    },
    "cubefeats-t1": {
        "label": "cubefeats (T1)", "env": "trellis", "cwd": PROJECT / "trellis1" / "TRELLIS",
        "configs": "cubefeats_opt/generate_cubefeats_opt_configs.py", "config_dir": "cubefeats_opt_configs",
        "runner": "cubefeats_opt/run_all_cubefeats_opt.py", "multi_source": True,
        "mesh": "optimization_output/cubefeats_opt/{tag}/{src}/{obj}/final_mesh/cubefeats_opt_world.glb",
    },
    "slat-t2": {
        "label": "SLAT (T2)", "env": "trellis2", "cwd": PROJECT / "trellis2" / "TRELLIS.2",
        "configs": "slat_opt/generate_slat_opt_configs.py", "config_dir": "slat_opt_configs",
        "runner": "slat_opt/run_all_slat_opt.py", "multi_source": True, "only": "slat",
        "mesh": "optimization_output/slat_opt/{tag}/{src}/{obj}/slat/mesh_final_world.ply",
    },
    "ovoxel-t2": {
        "label": "o-voxel (T2)", "env": "trellis2", "cwd": PROJECT / "trellis2" / "TRELLIS.2",
        "configs": "slat_opt/generate_slat_opt_configs.py", "config_dir": "slat_opt_configs",
        "runner": "slat_opt/run_all_slat_opt.py", "multi_source": True, "only": "ovoxel",
        "mesh": "optimization_output/slat_opt/{tag}/{src}/{obj}/ovoxel/mesh_final_world.ply",
    },
}

# benchmark method names as they appear in the results folder
BASELINES = {
    "nvdiffrec": "nvdiffrec (multi-view optimization)",
    "recgen": "RecGen (v1 prompts)", "recgen_v2": "RecGen",
    "sam3d": "SAM3D (v1 prompts)", "sam3d_v2": "SAM3D",
    "trellis": "TRELLIS (v1 prompts)", "trellis_v2": "TRELLIS",
    "trellis2": "TRELLIS.2 (v1 prompts)", "trellis2_v2": "TRELLIS.2",
}

# refined method names: <experiment>_<source>_<tag>
EXPERIMENT_KEY = {"slat-t1": "trellis1_slat", "cubefeats-t1": "trellis1_cubefeats",
                  "slat-t2": "trellis2_slat", "ovoxel-t2": "trellis2_ovoxel"}

METRICS = {
    "chamfer": ("chamfer (x2e-3)", "shape", "bidir_chamfer", 2e3, True, 4),
    "normal": ("normal (1-cos)", "geometry", "normal_angle", 1, True, 4),
    "depth": ("depth SI-MSE", "geometry", "depth_mse_scene", 1, True, 4),
    "view-psnr": ("view PSNR-L", "view", "psnr_ldr", 1, False, 2),
    "view-ssim": ("view SSIM", "view", "ssim", 1, False, 4),
    "view-lpips": ("view LPIPS", "view", "lpips", 1, True, 4),
    "light-psnr": ("light PSNR-L", "light", "psnr_ldr", 1, False, 2),
    "albedo": ("albedo PSNR", "material", "psnr_ldr", 1, False, 2),
}
DEFAULT_METRICS = ["chamfer", "normal", "depth", "albedo", "view-psnr", "view-lpips"]


def dataset_objects() -> list:
    """Every DTC object present locally (a directory with training cameras)."""
    if not DATASET.is_dir():
        return []
    return sorted(p.name for p in DATASET.iterdir() if (p / "transforms_train.json").exists())


def resolve_objects(names) -> list:
    """CLI object names -> DTC directory names."""
    if not names:
        return list(THESIS_OBJECTS.values())
    available = dataset_objects()
    out = []
    for n in names:
        key = n.lower()
        if key == "all":
            out += list(THESIS_OBJECTS.values())
        elif key in ("all-dataset", "dataset", "everything"):
            out += available
        elif key in THESIS_OBJECTS:
            out.append(THESIS_OBJECTS[key])
        elif n in available:
            out.append(n)
        else:
            hits = [o for o in available if o.lower().startswith(key)] or \
                   [o for o in available if key in o.lower()]
            if len(hits) == 1:
                out.append(hits[0])
            elif not hits:
                raise ValueError(f"unknown object {n!r}; try `thesis3d objects --dataset`")
            else:
                raise ValueError(f"{n!r} matches {len(hits)} objects: {', '.join(hits[:6])}"
                                 + (" ..." if len(hits) > 6 else ""))
    seen, unique = set(), []
    for o in out:
        if o not in seen:
            seen.add(o)
            unique.append(o)
    return unique


def short_name(obj: str) -> str:
    return obj.split("_")[0]


def refined_method(variant: str, source: str) -> str:
    return f"{EXPERIMENT_KEY[variant]}_{source}_{TAG}"


def results_dir(live: bool = False) -> Path:
    """Where scores.json files are read from: the app snapshot, or the live project."""
    if live or not SNAPSHOT_RESULTS.is_dir():
        return LIVE_RESULTS
    return SNAPSHOT_RESULTS
