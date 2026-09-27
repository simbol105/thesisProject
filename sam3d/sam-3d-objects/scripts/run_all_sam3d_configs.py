"""Run selected SAM3D category configs sequentially."""

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = REPO_ROOT / "sam3d_configs"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "sam3d_output"
DEFAULT_RUNNER = Path(__file__).resolve().parent / "run_sam3d_config.py"
CATEGORIES = [
    "Birdhouse_Lighting001",
    "Cup_B08TWHJ33Q_Tan_Lighting001",
    "Figurine_B08FYFNYP4_LionKing_Lighting001",
    "Gargoyle_B08SQMBDXY_HandsOnKnees_Lighting001",
    "Mallard_B082D168CK_MintGreen_Lighting001",
    "Mouse_B0CHNVBBLF_Honeycomb_1_Lighting001",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run selected SAM3D YAML configs")
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    config_dir = args.config_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    runner = args.runner.expanduser().resolve()
    configs = [config_dir / f"{category}.yaml" for category in CATEGORIES]
    missing = [config for config in configs if not config.is_file()]
    if missing:
        parser.error(f"missing YAML configs: {', '.join(map(str, missing))}")
    if not runner.is_file():
        parser.error(f"runner does not exist: {runner}")

    succeeded = skipped = 0
    failed: list[Path] = []
    for index, config_path in enumerate(configs, start=1):
        marker = output_root / config_path.stem / "metadata.json"
        if marker.is_file() and marker.stat().st_size > 0:
            skipped += 1
            print(f"[{index}/{len(configs)}] skip {config_path.stem}: complete", flush=True)
            continue

        print(f"[{index}/{len(configs)}] run {config_path.stem}", flush=True)
        result = subprocess.run(
            [sys.executable, str(runner), "--yaml-config", str(config_path)],
            cwd=REPO_ROOT,
            check=False,
        )
        if result.returncode == 0:
            succeeded += 1
        else:
            failed.append(config_path)
            print(f"failed with exit code {result.returncode}", flush=True)
            if args.stop_on_error:
                break

    print(
        f"Finished: {succeeded} succeeded, {skipped} skipped, {len(failed)} failed",
        flush=True,
    )
    if failed:
        print("Failed configs:")
        for config_path in failed:
            print(f"  {config_path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
