"""TRELLIS.2 SLAT (and o-voxel) optimization starting from an existing reconstruction"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import argparse
import subprocess

import torch
import yaml

MESH_KEYS = {"recgen": "recgen_mesh_path", "sam3d": "sam3d_mesh_path", "trellis": "trellis_mesh_path",
             "trellis2": "trellis2_mesh_path"}
REFINERS = {"slat": ROOT / "topology_study" / "slat_refinement.py",
            "ovoxel": ROOT / "topology_study" / "ovoxel_refinement.py"}


def main():
    parser = argparse.ArgumentParser(description="TRELLIS.2 SLAT / o-voxel optimization from an existing mesh")
    parser.add_argument("--yaml-config", type=Path, required=True)
    parser.add_argument("--mesh-source", choices=list(MESH_KEYS), default=None, help="overrides mesh_source")
    parser.add_argument("--only", choices=list(REFINERS), default=None, help="run only one refinement")
    parser.add_argument("--check", action="store_true", help="encode + round-trip check only (no optimization)")
    parser.add_argument("--force", action="store_true", help="rerun refinements that already finished")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.yaml_config.read_text())
    source = args.mesh_source or str(cfg.get("mesh_source", "recgen"))
    mesh_path = Path(cfg[MESH_KEYS[source]])
    assert mesh_path.exists(), (f"{source} mesh does not exist: {mesh_path} -- generate it with "
                                f"benchmark/run_all.py --methods {source}")
    out_root = Path(cfg["out_dir"].format(mesh_source=source))
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[slat_opt2] {cfg['category']} from {source}: {mesh_path}\n[slat_opt2] out: {out_root}")

    state_path = out_root / "init_state.pt"
    if not state_path.exists() or args.check:
        import encode_mesh as E
        state, tm = E.encode_mesh(mesh_path, int(cfg.get("encode_res", 512)))
        if not state_path.exists():
            torch.save(state, state_path)
            print(f"[slat_opt2] wrote {state_path}")
        if args.check:
            E.roundtrip_check(state, tm, out_root / "encode_check", cfg)
            return
        del state, tm
        torch.cuda.empty_cache()
    else:
        print(f"[slat_opt2] reusing {state_path}")

    run_cfg = {**cfg, "out_root": str(out_root), "mesh_source": source}
    run_cfg_path = out_root / "run_config.yaml"
    run_cfg_path.write_text(yaml.safe_dump(run_cfg, sort_keys=False))
    targets = [args.only] if args.only else list(cfg.get("optimize", ["slat", "ovoxel"]))
    failed = []
    for target in targets:
        if (out_root / target / "summary.json").exists() and not args.force:
            print(f"[slat_opt2] {target}: finished already ({out_root / target / 'summary.json'}), skipped")
            continue
        print(f"[slat_opt2] === {target} refinement")
        if subprocess.run([sys.executable, str(REFINERS[target]), "--yaml-config", str(run_cfg_path)],
                          cwd=ROOT, check=False).returncode != 0:
            failed.append(target)
    if failed:
        raise SystemExit(f"failed: {failed}")


if __name__ == "__main__":
    main()
