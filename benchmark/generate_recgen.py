"""Run RecGen on the DTC thesis objects from a chosen prompt image."""
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import DATASET, OBJECTS, PROJECTS, prompt_image, train_frame_index

RECGEN_ROOT = PROJECTS / "recgen"


def render_gt(obj: str, frame_idx: int):
    """pyrender render of the GT mesh from training camera `frame_idx` (recgen/generate.py's setup)."""
    import pyrender
    import trimesh
    transforms = json.load(open(DATASET / obj / "transforms_train.json"))
    intr = transforms["intrinsic"]
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    mesh = trimesh.load_mesh(DATASET / obj / "model.glb")
    mesh.apply_transform(np.array(transforms["object_pose"]))
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.001, zfar=100.0)
    cv_to_gl = np.diag([1, -1, -1, 1]).astype(np.float32)
    scene.add(camera, pose=np.linalg.inv(np.array(transforms["frames"][frame_idx]["transform_matrix"])) @ cv_to_gl)
    first = cv2.imread(str(DATASET / obj / (transforms["frames"][frame_idx]["file_path"] + ".png")), cv2.IMREAD_UNCHANGED)
    h, w = first.shape[:2]
    renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()
    return color, depth, (fx, fy, cx, cy)


def main():
    parser = argparse.ArgumentParser(description="RecGen on the DTC thesis objects")
    parser.add_argument("--objects", nargs="*", default=OBJECTS)
    parser.add_argument("--prompt-source", default="recgen", help="recgen | sam3d | path to a prompts yaml")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--rgb-source", choices=["photo", "render"], default="photo")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    script = RECGEN_ROOT / "scripts" / "run_inference.py"
    failed = []
    for obj in args.objects:
        out = (args.out_root / obj).resolve()
        if (out / "mesh.obj").exists() and not args.overwrite:
            print(f"[recgen] {obj}: mesh.obj exists, skipped")
            continue
        image_path = prompt_image(obj, args.prompt_source)
        idx = train_frame_index(obj, image_path.name)
        print(f"[recgen] {obj}: {image_path} (frame {idx}, rgb from {args.rgb_source})")
        color, depth, (fx, fy, cx, cy) = render_gt(obj, idx)
        if args.rgb_source == "photo":
            photo = np.array(Image.open(image_path).convert("RGBA"))
            mask = photo[..., 3] > 127
            rgb = (photo[..., :3] * mask[..., None]).astype(np.uint8)
        else:
            mask = color[..., 3] > 0
            rgb = color[..., :3]

        tmp = Path(tempfile.mkdtemp(prefix="recgen_inputs_"))
        try:
            Image.fromarray(rgb).save(tmp / "rgb.png")
            cv2.imwrite(str(tmp / "depth.png"), np.clip(depth.astype(np.float32) * 1000.0, 0, 65535).astype(np.uint16))
            cv2.imwrite(str(tmp / "mask.png"), mask.astype(np.uint8) * 255)
            with open(tmp / "intrinsics.yaml", "w") as f:
                yaml.safe_dump({"fu": float(fx), "fv": float(fy), "pu": float(cx), "pv": float(cy)}, f)
            out.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                [sys.executable, str(script), "--rgb", str(tmp / "rgb.png"), "--depth", str(tmp / "depth.png"),
                 "--mask", str(tmp / "mask.png"), "--intrinsics", str(tmp / "intrinsics.yaml"),
                 "--out", str(out), "--save-splat"],
                cwd=RECGEN_ROOT, check=False)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if result.returncode != 0:
            failed.append(obj)
            print(f"[recgen] {obj}: FAILED (exit {result.returncode})")
            continue
        (out / "prompt.json").write_text(json.dumps({
            "category": obj, "image_path": str(image_path), "train_frame_index": idx,
            "prompt_source": str(args.prompt_source), "rgb_source": args.rgb_source,
            "depth_source": "GT mesh rendered with pyrender at the prompt camera"}, indent=2))
        print(f"[recgen] {obj}: done -> {out}")
    if failed:
        raise SystemExit(f"failed: {failed}")


if __name__ == "__main__":
    main()
