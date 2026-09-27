"""Generate one RecGen thesis YAML config for each dataset object."""

import argparse
import json
from pathlib import Path


DEFAULT_DATASET_ROOT = Path("/home/fr/fr_fr/fr_ba122/projects/dataset/DTC")
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[1] / "thesis_configs"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "thesis_output"


def parse_prompt_overrides(value: str) -> dict[str, int]:
    """Parse a JSON object mapping category names to prompt indices."""
    try:
        overrides = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"invalid JSON prompt map: {error}") from error

    if not isinstance(overrides, dict):
        raise argparse.ArgumentTypeError("prompt overrides must be a JSON object")

    parsed: dict[str, int] = {}
    for category, prompt_id in overrides.items():
        if not isinstance(category, str) or not isinstance(prompt_id, int):
            raise argparse.ArgumentTypeError(
                "prompt overrides must map string category names to integer prompt IDs"
            )
        parsed[category] = prompt_id
    return parsed


def build_config(
    category: str,
    dataset_root: Path,
    output_root: Path,
    prompt_id: int,
) -> dict[str, object]:
    return {
        "category": category,
        "output_path": str(output_root / category),
        "prompt_id": prompt_id,
        "dataset_path": str(dataset_root),
    }


def render_config(config: dict[str, object]) -> str:
    """Render the fixed config schema without requiring a YAML package."""
    return "\n".join(
        [
            f'category: "{config["category"]}"',
            f'output_path: "{config["output_path"]}"',
            f'prompt_id: {config["prompt_id"]}',
            f'dataset_path: "{config["dataset_path"]}"',
            "",
        ]
    )


def generate_configs(
    dataset_root: Path,
    config_dir: Path,
    output_root: Path,
    default_prompt_id: int,
    prompt_overrides: dict[str, int],
    overwrite: bool,
    dry_run: bool,
) -> tuple[int, int]:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {dataset_root}")

    categories = sorted(path.name for path in dataset_root.iterdir() if path.is_dir())
    if not categories:
        raise RuntimeError(f"no object directories found in {dataset_root}")

    created = skipped = 0
    for category in categories:
        config_path = config_dir / f"{category}.yaml"
        if config_path.exists() and not overwrite:
            print(f"skip  {config_path} (already exists)")
            skipped += 1
            continue

        config = build_config(
            category,
            dataset_root,
            output_root,
            prompt_overrides.get(category, default_prompt_id),
        )
        if dry_run:
            print(f"would write {config_path}")
        else:
            config_dir.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                render_config(config),
                encoding="utf-8",
            )
            print(f"write {config_path}")
        created += 1

    return created, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate RecGen thesis configs for dataset object directories."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--prompt-id",
        type=int,
        default=5,
        help="default training-frame index used for every object (default: 5)",
    )
    parser.add_argument(
        "--prompt-overrides",
        type=parse_prompt_overrides,
        default={},
        help='JSON map of category names to prompt IDs, e.g. \'{"Birdhouse_Lighting001": 37}\'',
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace configs that already exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list configs without writing them",
    )
    args = parser.parse_args()

    created, skipped = generate_configs(
        dataset_root=args.dataset_root.expanduser().resolve(),
        config_dir=args.config_dir.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        default_prompt_id=args.prompt_id,
        prompt_overrides=args.prompt_overrides,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    action = "would create" if args.dry_run else "created"
    print(f"{action} {created} config(s); skipped {skipped} existing config(s)")


if __name__ == "__main__":
    main()
