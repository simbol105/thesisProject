"""TRELLIS1 cubefeats optimization starting from an existing reconstruction."""

import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "slat_opt"))
from hypothesis_mesh_slat import MESH_KEYS, run


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize the TRELLIS1 cubefeats decoded from an encoded mesh")
    parser.add_argument("--yaml-config", type=Path, required=True)
    parser.add_argument("--mesh-source", choices=list(MESH_KEYS), default=None,
                        help="overrides mesh_source in the config")
    parser.add_argument("--out-dir", default=None, help="overrides out_dir in the config")
    parser.add_argument("--check", action="store_true",
                        help="stop after the initial mesh (alignment / encoding sanity check)")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.yaml_config.read_text())
    if str(cfg.get("optimize")) != "cubefeats":
        raise SystemExit(f"{args.yaml_config} has optimize: {cfg.get('optimize')}; "
                         f"use cubefeats_opt/generate_cubefeats_opt_configs.py")
    run(args.yaml_config, args.mesh_source, args.out_dir, args.check)


if __name__ == "__main__":
    main()
