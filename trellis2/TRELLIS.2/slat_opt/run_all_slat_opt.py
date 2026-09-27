"""Run every TRELLIS.2 SLAT-optimization config sequentially, for one or more mesh sources."""
import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNNER = Path(__file__).resolve().parent / "hypothesis_mesh_slat.py"
MESH_KEY = {"recgen": "recgen_mesh_path", "sam3d": "sam3d_mesh_path", "trellis": "trellis_mesh_path",
            "trellis2": "trellis2_mesh_path"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all TRELLIS.2 SLAT optimization configs")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "slat_opt_configs")
    parser.add_argument("--mesh-source", nargs="+", choices=list(MESH_KEY), default=["recgen"])
    parser.add_argument("--objects", nargs="*", default=None, help="only these config stems")
    parser.add_argument("--only", choices=["slat", "ovoxel"], default=None)
    parser.add_argument("--check", action="store_true", help="encode + round-trip check only")
    parser.add_argument("--force", action="store_true", help="rerun finished refinements")
    args = parser.parse_args()

    configs = sorted(args.config_dir.glob("*.yaml"))
    if args.objects:
        configs = [c for c in configs if c.stem in set(args.objects)]
    if not configs:
        parser.error(f"no configs in {args.config_dir} (run slat_opt/generate_slat_opt_configs.py)")

    failed, done, skipped = [], 0, 0
    for source in args.mesh_source:
        for i, config in enumerate(configs, 1):
            cfg = yaml.safe_load(config.read_text())
            if not Path(cfg[MESH_KEY[source]]).exists():
                print(f"[{source} {i}/{len(configs)}] skip {config.stem}: no {source} mesh", flush=True)
                skipped += 1
                continue
            print(f"[{source} {i}/{len(configs)}] run {config.stem}", flush=True)
            cmd = [sys.executable, str(RUNNER), "--yaml-config", str(config), "--mesh-source", source]
            cmd += (["--only", args.only] if args.only else []) + (["--check"] if args.check else [])
            cmd += ["--force"] if args.force else []
            if subprocess.run(cmd, cwd=ROOT, check=False).returncode == 0:
                done += 1
            else:
                failed.append(f"{source}/{config.stem}")
    print(f"Finished: {done} ok, {skipped} skipped, {len(failed)} failed")
    for f in failed:
        print(f"  failed: {f}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
