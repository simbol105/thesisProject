"""Generate one SAM3D YAML config for each DTC object directory."""

import argparse
import random
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path("/home/fr/fr_fr/fr_ba122/projects/dataset/DTC")
DEFAULT_CONFIG_DIR = REPO_ROOT / "sam3d_configs"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "sam3d_output"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints/hf/pipeline.yaml"


def render_config(
    category: str,
    image_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    seed: int,
    save_video: bool,
    with_mesh_postprocess: bool,
    with_texture_baking: bool,
) -> str:
    return "\n".join(
        [
            f'category: "{category}"',
            f'image_path: "{image_path}"',
            f'output_path: "{output_path}"',
            f'checkpoint_path: "{checkpoint_path}"',
            f"seed: {seed}",
            f"save_video: {str(save_video).lower()}",
            f"with_mesh_postprocess: {str(with_mesh_postprocess).lower()}",
            f"with_texture_baking: {str(with_texture_baking).lower()}",
            "",
        ]
    )


def select_image(category_path: Path, image_name: str) -> Path | None:
    """Use the requested frame, or a random existing frame from indices 0-10."""
    train_path = category_path / "train"
    requested_path = train_path / image_name
    if requested_path.is_file():
        return requested_path

    fallback_paths = [
        train_path / f"im_{index:04d}.png"
        for index in range(11)
        if (train_path / f"im_{index:04d}.png").is_file()
    ]
    if not fallback_paths:
        return None
    return random.choice(fallback_paths)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SAM3D configs for DTC object directories."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--image-name", default="im_0005.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--mesh-postprocess",
        action="store_true",
        help="enable mesh hole filling (requires a compatible nvdiffrast CUDA build)",
    )
    parser.add_argument("--no-texture-baking", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    config_dir = args.config_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not dataset_root.is_dir():
        parser.error(f"dataset directory does not exist: {dataset_root}")
    if not checkpoint_path.is_file():
        parser.error(f"checkpoint config does not exist: {checkpoint_path}")

    categories = sorted(path.name for path in dataset_root.iterdir() if path.is_dir())
    if not categories:
        parser.error(f"no object directories found in {dataset_root}")

    created = skipped = 0
    for category in categories:
        image_path = select_image(dataset_root / category, args.image_name)
        if image_path is None:
            print(
                f"skip {category}: no requested image or fallback image in indices 0-10"
            )
            skipped += 1
            continue
        if image_path.name != args.image_name:
            print(f"fallback {category}: using {image_path.name}")

        config_path = config_dir / f"{category}.yaml"
        if config_path.exists() and not args.overwrite:
            print(f"skip {category}: config already exists")
            skipped += 1
            continue

        config_text = render_config(
            category,
            image_path,
            output_root / category,
            checkpoint_path,
            args.seed,
            not args.no_video,
            args.mesh_postprocess,
            not args.no_texture_baking,
        )
        if args.dry_run:
            print(f"would write {config_path}")
        else:
            config_dir.mkdir(parents=True, exist_ok=True)
            config_path.write_text(config_text, encoding="utf-8")
            print(f"write {config_path}")
        created += 1

    action = "would create" if args.dry_run else "created"
    print(f"{action} {created} config(s); skipped {skipped}")


if __name__ == "__main__":
    main()
