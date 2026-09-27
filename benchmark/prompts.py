"""Prompt image per DTC object, shared by the generators (no heavy imports)."""
import json
from pathlib import Path

import yaml

PROJECTS = Path(__file__).resolve().parents[1]
DATASET = PROJECTS / "dataset" / "DTC"

OBJECTS = [
    "Birdhouse_Lighting001",
    "Cup_B08TWHJ33Q_Tan_Lighting001",
    "Figurine_B08FYFNYP4_LionKing_Lighting001",
    "Gargoyle_B08SQMBDXY_HandsOnKnees_Lighting001",
    "Mallard_B082D168CK_MintGreen_Lighting001",
    "Mouse_B0CHNVBBLF_Honeycomb_1_Lighting001",
]


def _with_ext(base: Path) -> Path:
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        if base.with_suffix(ext).exists():
            return base.with_suffix(ext)
    raise FileNotFoundError(base)


def train_frame_index(obj: str, image_name: str) -> int:
    """Index in transforms_train.json of a training image given by name (im_0012 / train/im_0012.png)."""
    stem = Path(image_name).stem
    frames = json.load(open(DATASET / obj / "transforms_train.json"))["frames"]
    for i, f in enumerate(frames):
        if Path(f["file_path"]).name == stem:
            return i
    raise KeyError(f"{image_name} is not a training image of {obj}")


def prompt_image(obj: str, source: str = "recgen") -> Path:
    if str(source).endswith((".yaml", ".yml")):
        name = yaml.safe_load(open(source))["prompts"][obj]
        frames = json.load(open(DATASET / obj / "transforms_train.json"))["frames"]
        return _with_ext(DATASET / obj / frames[train_frame_index(obj, name)]["file_path"])
    if source == "recgen":
        cfg = yaml.safe_load(open(PROJECTS / "recgen" / "thesis_configs" / f"{obj}.yaml"))
        frames = json.load(open(DATASET / obj / "transforms_train.json"))["frames"]
        return _with_ext(DATASET / obj / frames[int(cfg["prompt_id"])]["file_path"])
    if source == "sam3d":
        cfg = yaml.safe_load(open(PROJECTS / "sam3d" / "sam-3d-objects" / "sam3d_configs" / f"{obj}.yaml"))
        return Path(cfg["image_path"])
    raise ValueError(f"unknown prompt source {source}")
