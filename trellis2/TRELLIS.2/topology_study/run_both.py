"""Run the SLAT refinement, then the o-voxel refinement, on the same config."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SCRIPTS = { "ovoxel": HERE / "ovoxel_refinement.py"  }


def main() -> int:
    parser = argparse.ArgumentParser(description="TRELLIS.2 SLAT + o-voxel topology check")
    parser.add_argument("--yaml-config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--only", choices=list(SCRIPTS), default=None)
    parser.add_argument("--regenerate", action="store_true", help="resample the initial latents and alignment")
    parser.add_argument("--keep-going", action="store_true", help="run the second script even if the first fails")
    args = parser.parse_args()

    names = [args.only] if args.only else list(SCRIPTS)
    failed = []
    for i, name in enumerate(names):
        cmd = [sys.executable, str(SCRIPTS[name]), "--yaml-config", str(args.yaml_config.resolve())]
        if args.regenerate and i == 0:
            cmd.append("--regenerate")
        print(f"=== [{i + 1}/{len(names)}] {name}: {' '.join(cmd)}", flush=True)
        if subprocess.run(cmd, cwd=HERE.parent).returncode != 0:
            failed.append(name)
            print(f"=== {name} failed", flush=True)
            if not args.keep_going:
                break

    out_root = Path(yaml.safe_load(args.yaml_config.read_text())["out_root"])
    print("\n=== summary")
    for name in names:
        path = out_root / name / "summary.json"
        if not path.exists():
            print(f"{name:>7s}: no summary.json")
            continue
        s = json.loads(path.read_text())
        t0, t1, g = s["topology_init"], s["topology_final"], s["gate_changes"]
        print(f"{name:>7s}: chamfer {s['chamfer_mm'][0]:.2f} -> {s['chamfer_mm'][1]:.2f} mm | "
              f"euler {t0['euler']} -> {t1['euler']} | components {t0['components']} -> {t1['components']} | "
              f"boundary edges {t0['boundary_edges']} -> {t1['boundary_edges']} | "
              f"voxels +{g['voxels_added']}/-{g['voxels_removed']} | quad flags flipped {g['intersected_flips']}")
        if "gate_noise_floor" in s:
            n = s["gate_noise_floor"]
            print(f"{'':>7s}  noise floor (same latent decoded twice): voxels +{n['voxels_added']}/-{n['voxels_removed']} | "
                  f"quad flags flipped {n['intersected_flips']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
