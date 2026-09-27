"""Run SAM3D reconstruction for one category YAML config."""

import argparse
import json
import sys
import time
from pathlib import Path

import imageio


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.append(str(REPO_ROOT / "notebook"))

from inference import (
    Inference,
    load_image,
    make_scene,
    ready_gaussian_for_video_rendering,
    render_video,
)


def load_config(config_path: Path) -> dict[str, object]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required to read SAM3D config files") from error

    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"config must contain a YAML mapping: {config_path}")
    return config


def reconstruct(config_path: Path) -> None:
    config = load_config(config_path)
    category = str(config["category"])
    image_path = Path(str(config["image_path"])).expanduser().resolve()
    output_path = Path(str(config["output_path"])).expanduser().resolve()
    checkpoint_path = Path(str(config["checkpoint_path"])).expanduser().resolve()
    seed = int(config.get("seed", 42))
    save_video = bool(config.get("save_video", True))
    with_mesh_postprocess = bool(config.get("with_mesh_postprocess", True))
    with_texture_baking = bool(config.get("with_texture_baking", True))

    if not image_path.is_file():
        raise FileNotFoundError(f"input image does not exist: {image_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint config does not exist: {checkpoint_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Loading SAM3D checkpoint for {category}", flush=True)
    inference = Inference(str(checkpoint_path), compile=False)
    image = load_image(str(image_path))
    mask = image[:, :, 3] > 0 
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f"SAM3D input must be RGBA, got shape {image.shape}")

    start_time = time.time()
    output = inference._pipeline.run(
        image,
        None,
        seed=seed,
        with_mesh_postprocess=with_mesh_postprocess,
        with_texture_baking=with_texture_baking,
    )
    runtime_seconds = time.time() - start_time

    output["glb"].export(output_path / "mesh.glb")
    gaussian = output["gaussian"][0]
    gaussian.save_ply(output_path / "gaussian.ply")

    if save_video:
        scene_gs = ready_gaussian_for_video_rendering(make_scene(output))
        video = render_video(
            scene_gs,
            r=1,
            fov=60,
            pitch_deg=15,
            yaw_start_deg=-45,
            resolution=512,
        )["color"]
        imageio.mimsave(
            output_path / "turntable.gif",
            video,
            format="GIF",
            duration=1000 / 30,
            loop=0,
        )

    metadata = {
        "category": category,
        "image_path": str(image_path),
        "checkpoint_path": str(checkpoint_path),
        "seed": seed,
        "with_mesh_postprocess": with_mesh_postprocess,
        "with_texture_baking": with_texture_baking,
        "runtime_seconds": runtime_seconds,
        "outputs": ["mesh.glb", "gaussian.ply"]
        + (["turntable.gif"] if save_video else []),
    }
    (output_path / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Finished {category}; outputs saved to {output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAM3D for one YAML config")
    parser.add_argument("--yaml-config", type=Path, required=True)
    args = parser.parse_args()
    reconstruct(args.yaml_config.expanduser().resolve())


if __name__ == "__main__":
    main()
