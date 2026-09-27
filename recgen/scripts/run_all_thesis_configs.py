"""Run RecGen reconstruction for every thesis YAML config in sequence."""

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG_DIR = REPO_ROOT / "thesis_configs"
DEFAULT_GENERATE_SCRIPT = REPO_ROOT / "generate.py"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "thesis_output"


def run_configs(
    config_dir: Path,
    generate_script: Path,
    output_root: Path,
    stop_on_error: bool,
) -> int:
    config_paths = sorted(config_dir.glob("*.yaml"))
    if not config_paths:
        raise FileNotFoundError(f"no YAML configs found in {config_dir}")

    failed: list[Path] = []
    succeeded = 0
    skipped = 0
    for index, config_path in enumerate(config_paths, start=1):
        completion_marker = output_root / config_path.stem / "metadata.json"
        if completion_marker.is_file() and completion_marker.stat().st_size > 0:
            skipped += 1
            print(
                f"\n[{index}/{len(config_paths)}] Skipping {config_path.stem} "
                f"(already complete)",
                flush=True,
            )
            continue

        print(f"\n[{index}/{len(config_paths)}] Running {config_path.stem}", flush=True)
        result = subprocess.run(
            [sys.executable, str(generate_script), "--yaml-config", str(config_path)],
            cwd=REPO_ROOT,
            check=False,
        )
        if result.returncode == 0:
            succeeded += 1
            print(f"Completed {config_path.stem}", flush=True)
            continue

        failed.append(config_path)
        print(
            f"Failed {config_path.stem} with exit code {result.returncode}",
            flush=True,
        )
        if stop_on_error:
            break

    print(
        f"\nFinished: {succeeded} succeeded, {skipped} skipped, "
        f"{len(failed)} failed",
        flush=True,
    )
    if failed:
        print("Failed configs:", flush=True)
        for config_path in failed:
            print(f"  {config_path}", flush=True)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run generate.py for every thesis YAML config in sequence."
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help="directory containing category YAML configs",
    )
    parser.add_argument(
        "--generate-script",
        type=Path,
        default=DEFAULT_GENERATE_SCRIPT,
        help="path to the single-config RecGen entry point",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="root containing one category output directory per config",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="stop after the first failed category instead of continuing",
    )
    args = parser.parse_args()

    config_dir = args.config_dir.expanduser().resolve()
    generate_script = args.generate_script.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not config_dir.is_dir():
        parser.error(f"config directory does not exist: {config_dir}")
    if not generate_script.is_file():
        parser.error(f"generate script does not exist: {generate_script}")

    return run_configs(config_dir, generate_script, output_root, args.stop_on_error)


if __name__ == "__main__":
    raise SystemExit(main())
