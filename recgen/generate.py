import os
os.environ["ATTN_BACKEND"] = 'flash-attn'


from PIL import Image
import numpy as np

from pathlib import Path
import yaml
import cv2
import json
import trimesh
import subprocess
import sys
import tempfile
import shutil

# Use EGL for pyrender on headless compute nodes without an X display.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import pyrender
import argparse


SIZE = 1024

def load_image(path):
    image = Image.open(path)
    image = image.resize((SIZE, SIZE), Image.NEAREST)  # resize correctly
    image = np.array(image).astype(np.uint8)
    return image

def reconstruct(config_path: Path):
    config_yaml = yaml.safe_load(open(config_path, 'r'))
    
    category = config_yaml.get("category", "")
    out_dir = Path(config_yaml.get("output_path", f"./out/{category}")).expanduser().resolve()
    prompt_id = int(config_yaml.get("prompt_id", "5"))
    dataset_path = Path(config_yaml.get("data_dir", "/home/fr/fr_fr/fr_ba122/projects/dataset/DTC"))
    
    
    mesh_path = dataset_path / category / "model.glb"
    assert mesh_path.exists(), f"Mesh path {mesh_path} does not exist."
    transforms_json = dataset_path / category / "transforms_train.json"
    transforms = json.load(open(transforms_json, "r"))
    intrinsics = transforms["intrinsic"]
    fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    extrinsics = np.array(transforms["frames"][prompt_id]["transform_matrix"])
    object_pose = np.array(transforms["object_pose"])
    
    mesh = trimesh.load_mesh(mesh_path)
    cv_to_gl = np.diag([1, -1, -1, 1]).astype(np.float32)
    mesh.apply_transform(object_pose)
    
    pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
    scene.add(pyrender_mesh)
    
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.001,
        zfar=100.0)
    cv_to_gl = np.diag([1, -1, -1, 1]).astype(np.float32)
    extrinsics_t = np.linalg.inv(extrinsics) @ cv_to_gl
    scene.add(camera, pose=extrinsics_t)
    
    renderer = pyrender.OffscreenRenderer(viewport_width=2000, viewport_height=2000)
    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()
    
    mask = color[:, :, 3]
    rgb = color[:, :, :3]
    

    script_path = Path(__file__).resolve().parent / "scripts" / "run_inference.py"
    temp_dir = Path(tempfile.mkdtemp(prefix="recgen_inputs_"))
    try:
        rgb_path = temp_dir / "rgb.png"
        depth_path = temp_dir / "depth.png"
        mask_path = temp_dir / "mask.png"
        intrinsics_path = temp_dir / "intrinsics.yaml"

        Image.fromarray(rgb.astype(np.uint8)).save(rgb_path)
        depth_mm = np.clip(depth.astype(np.float32) * 1000.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(depth_path), depth_mm)
        cv2.imwrite(str(mask_path), (mask > 0).astype(np.uint8) * 255)
        with open(intrinsics_path, "w") as file:
            yaml.safe_dump(
                {"fu": float(fx), "fv": float(fy), "pu": float(cx), "pv": float(cy)},
                file,
            )

        subprocess.run(
            [
                "pixi",
                "run",
                "python",
                str(script_path),
                "--rgb", str(rgb_path),
                "--depth", str(depth_path),
                "--mask", str(mask_path),
                "--intrinsics", str(intrinsics_path),
                "--out", str(out_dir),
                "--save-splat",
            ],
            check=True,
            cwd=script_path.parent.parent,
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"Finished reconstruction for {category}, Results saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters to refine latent")
    parser.add_argument("--yaml-config", type=Path, required=True, help="Path to the YAML config file" )
    args = parser.parse_args()
    reconstruct(Path(args.yaml_config))