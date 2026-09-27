"""Run every SLAT-optimization config sequentially (one mesh source per invocation)."""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = REPO_ROOT / "slat_opt_configs"
RUNNER = Path(__file__).resolve().parent / "hypothesis_mesh_slat.py"

MESH_KEY = {"recgen": "recgen_mesh_path", "sam3d": "sam3d_mesh_path", "trellis": "trellis_mesh_path",
             "trellis2": "trellis2_mesh_path"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all SLAT optimization configs")
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--mesh-source", choices=list(MESH_KEY), default="recgen")
    parser.add_argument("--objects", nargs="*", default=None, help="only these config stems")
    parser.add_argument("--check", action="store_true", help="forward --check to every run")
    parser.add_argument("--force", action="store_true", help="rerun even if the output exists")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    configs = sorted(args.config_dir.expanduser().resolve().glob("*.yaml"))
    if args.objects:
        configs = [c for c in configs if c.stem in set(args.objects)]
    if not configs:
        parser.error(f"no YAML configs found in {args.config_dir}")

    succeeded = skipped = 0
    failed = []
    for index, config_path in enumerate(configs, start=1):
        cfg = yaml.safe_load(config_path.read_text())
        mesh_path = Path(cfg[MESH_KEY[args.mesh_source]])
        if not mesh_path.exists():
            print(f"[{index}/{len(configs)}] skip {config_path.stem}: "
                  f"no {args.mesh_source} mesh at {mesh_path}", flush=True)
            skipped += 1
            continue

        target = str(cfg.get("optimize", "slat"))
        out_dir = Path(cfg["out_dir"].format(mesh_source=args.mesh_source, optimize=target))
        done = out_dir / "final_mesh" / f"{target}_opt_world.glb"
        if done.is_file() and not args.check and not args.force:
            print(f"[{index}/{len(configs)}] skip {config_path.stem}: already done", flush=True)
            skipped += 1
            continue

        print(f"[{index}/{len(configs)}] run {config_path.stem} ({args.mesh_source}, optimize={target})", flush=True)
        cmd = [sys.executable, str(RUNNER), "--yaml-config", str(config_path),
               "--mesh-source", args.mesh_source]
        if args.check:
            cmd.append("--check")
        result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
        if result.returncode == 0:
            succeeded += 1
        else:
            failed.append(config_path)
            print(f"failed with exit code {result.returncode}", flush=True)
            if args.stop_on_error:
                break

    print(f"Finished: {succeeded} succeeded, {skipped} skipped, {len(failed)} failed", flush=True)
    for config_path in failed:
        print(f"  failed: {config_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
