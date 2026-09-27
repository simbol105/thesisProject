"""Run TRELLIS image-to-3D reconstruction for one YAML config."""

import argparse
import json
import sys
import time
from pathlib import Path

import imageio
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils, render_utils


def load_config(config_path: Path) -> dict[str, object]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required to read TRELLIS config files") from error

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
    model_name = str(config.get("model_name", "microsoft/TRELLIS-image-large"))
    seed = int(config.get("seed", 1))
    save_videos = bool(config.get("save_videos", True))
    texture_size = int(config.get("texture_size", 1024))
    simplify = float(config.get("simplify", 0.95))

    if not image_path.is_file():
        raise FileNotFoundError(f"input image does not exist: {image_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Loading TRELLIS model for {category}", flush=True)
    pipeline = TrellisImageTo3DPipeline.from_pretrained(model_name)
    pipeline.cuda()

    print(f"Reconstructing {category} from {image_path.name}", flush=True)
    start_time = time.time()
    outputs = pipeline.run(Image.open(image_path), seed=seed)
    runtime_seconds = time.time() - start_time

    if save_videos:
        video = render_utils.render_video(outputs["gaussian"][0])["color"]
        imageio.mimsave(output_path / "gaussian.mp4", video, fps=30)
        video = render_utils.render_video(outputs["radiance_field"][0])["color"]
        imageio.mimsave(output_path / "radiance_field.mp4", video, fps=30)
        video = render_utils.render_video(outputs["mesh"][0])["normal"]
        imageio.mimsave(output_path / "mesh.mp4", video, fps=30)

    glb = postprocessing_utils.to_glb(
        outputs["gaussian"][0],
        outputs["mesh"][0],
        simplify=simplify,
        texture_size=texture_size,
    )
    glb.export(output_path / "mesh.glb")
    outputs["gaussian"][0].save_ply(output_path / "gaussian.ply")

    metadata = {
        "category": category,
        "image_path": str(image_path),
        "model_name": model_name,
        "seed": seed,
        "runtime_seconds": runtime_seconds,
        "outputs": ["mesh.glb", "gaussian.ply"]
        + (["gaussian.mp4", "radiance_field.mp4", "mesh.mp4"] if save_videos else []),
    }
    (output_path / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Finished {category}; outputs saved to {output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TRELLIS for one YAML config")
    parser.add_argument("--yaml-config", type=Path, required=True)
    args = parser.parse_args()
    reconstruct(args.yaml_config.expanduser().resolve())


if __name__ == "__main__":
    main()
