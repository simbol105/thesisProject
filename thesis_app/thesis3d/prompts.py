"""The app's prompt list: which training image conditions each object's generation."""
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from . import config as C

HEADER = """\
# Prompt image per object, used by `thesis3d generate` for every method.
# Managed by the app (`thesis3d prompts ...`); the project's own benchmark/prompts.yaml is untouched.
# tag names the run: outputs go to generative_runs/<tag>/ and benchmark_output/<method>_<tag>/.
"""


def load() -> Dict:
    """The app's prompt file, seeded from the project's on first use."""
    if C.APP_PROMPTS.exists():
        return yaml.safe_load(C.APP_PROMPTS.read_text())
    project = C.BENCHMARK_DIR / "prompts.yaml"
    data = yaml.safe_load(project.read_text()) if project.exists() else {"tag": C.TAG, "prompts": {}}
    save(data)
    return data


def save(data: Dict) -> Path:
    lines = [HEADER, f"tag: {data.get('tag', C.TAG)}", "", "prompts:"]
    for obj, image in sorted(data.get("prompts", {}).items()):
        lines.append(f"  {obj}: {image}")
    C.APP_PROMPTS.write_text("\n".join(lines) + "\n")
    return C.APP_PROMPTS


def frames(obj: str) -> List[Dict]:
    return json.load(open(C.DATASET / obj / "transforms_train.json"))["frames"]


def object_center(obj: str):
    pose = json.load(open(C.DATASET / obj / "transforms_train.json"))["object_pose"]
    return [pose[0][3], pose[1][3], pose[2][3]]


def camera_angles(obj: str) -> List[Tuple[int, str, float, float]]:
    """(frame index, image stem, elevation deg, azimuth deg) for every training view."""
    cx, cy, cz = object_center(obj)
    out = []
    for i, f in enumerate(frames(obj)):
        E = f["transform_matrix"]                       # world -> camera
        R = [row[:3] for row in E[:3]]
        t = [row[3] for row in E[:3]]
        pos = [-(R[0][k] * t[0] + R[1][k] * t[1] + R[2][k] * t[2]) for k in range(3)]
        dx, dy, dz = pos[0] - cx, pos[1] - cy, pos[2] - cz
        r = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
        out.append((i, Path(f["file_path"]).name, math.degrees(math.asin(dz / r)),
                    math.degrees(math.atan2(dy, dx))))
    return out


def pick_view(obj: str, elevation: float = 30.0) -> Tuple[int, str, float, float]:
    """The training view whose elevation is closest to `elevation` (a raised three-quarter view)."""
    views = camera_angles(obj)
    if not views:
        raise ValueError(f"{obj} has no training cameras")
    return min(views, key=lambda v: abs(v[2] - elevation))


def resolve(obj: str, value: str) -> str:
    """Accept an image stem (im_0077), a file name or a frame index; return the stem."""
    fs = frames(obj)
    if value.isdigit():
        i = int(value)
        if not 0 <= i < len(fs):
            raise ValueError(f"frame {i} out of range (0..{len(fs) - 1})")
        return Path(fs[i]["file_path"]).name
    stem = Path(value).stem
    if any(Path(f["file_path"]).name == stem for f in fs):
        return stem
    raise ValueError(f"{value!r} is not a training image of {obj}")


def set_prompt(obj: str, image: str) -> str:
    data = load()
    data.setdefault("prompts", {})[obj] = image
    save(data)
    return image


def image_path(obj: str) -> Optional[Path]:
    """The resolved prompt image on disk, or None if the object has no prompt yet."""
    data = load()
    name = data.get("prompts", {}).get(obj)
    if name is None:
        return None
    for f in frames(obj):
        if Path(f["file_path"]).name == name:
            for ext in (".png", ".jpg", ".jpeg", ".webp"):
                p = (C.DATASET / obj / f["file_path"]).with_suffix(ext)
                if p.exists():
                    return p
    return None


def has_prompt(obj: str) -> bool:
    return obj in load().get("prompts", {})
