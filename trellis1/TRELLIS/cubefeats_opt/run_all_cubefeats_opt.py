"""Run every cubefeats-optimization config sequentially, for one or more mesh sources."""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
RUNNER = HERE / "hypothesis_mesh_cubefeats.py"
MESH_KEY = {"recgen": "recgen_mesh_path", "sam3d": "sam3d_mesh_path", "trellis": "trellis_mesh_path",
            "trellis2": "trellis2_mesh_path"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all TRELLIS1 cubefeats optimization configs")
    parser.add_argument("--config-dir", type=Path, default=REPO_ROOT / "cubefeats_opt_configs")
    parser.add_argument("--mesh-source", nargs="+", choices=list(MESH_KEY), default=["recgen"])
    parser.add_argument("--objects", nargs="*", default=None, help="only these config stems")
    parser.add_argument("--check", action="store_true", help="forward --check to every run")
    parser.add_argument("--force", action="store_true", help="rerun even if the output exists")
    args = parser.parse_args()

    configs = sorted(args.config_dir.expanduser().resolve().glob("*.yaml"))
    if args.objects:
        configs = [c for c in configs if c.stem in set(args.objects)]
    if not configs:
        parser.error(f"no configs in {args.config_dir} (run cubefeats_opt/generate_cubefeats_opt_configs.py)")

    done = skipped = 0
    failed = []
    for source in args.mesh_source:
        for i, config in enumerate(configs, 1):
            tag = f"[{source} {i}/{len(configs)}]"
            cfg = yaml.safe_load(config.read_text())
            if not Path(cfg[MESH_KEY[source]]).exists():
                print(f"{tag} skip {config.stem}: no {source} mesh", flush=True)
                skipped += 1
                continue
            out_dir = Path(cfg["out_dir"].format(mesh_source=source, optimize="cubefeats"))
            if (out_dir / "final_mesh" / "cubefeats_opt_world.glb").is_file() and not (args.check or args.force):
                print(f"{tag} skip {config.stem}: already done", flush=True)
                skipped += 1
                continue
            print(f"{tag} run {config.stem}", flush=True)
            cmd = [sys.executable, str(RUNNER), "--yaml-config", str(config), "--mesh-source", source]
            cmd += ["--check"] if args.check else []
            if subprocess.run(cmd, cwd=REPO_ROOT, check=False).returncode == 0:
                done += 1
            else:
                failed.append(f"{source}/{config.stem}")
    print(f"Finished: {done} ok, {skipped} skipped, {len(failed)} failed", flush=True)
    for f in failed:
        print(f"  failed: {f}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
