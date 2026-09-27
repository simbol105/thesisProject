"""Run SAM3D on the DTC thesis objects from a chosen prompt image."""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import OBJECTS, PROJECTS, prompt_image

SAM3D_ROOT = PROJECTS / "sam3d" / "sam-3d-objects"


def main():
    parser = argparse.ArgumentParser(description="SAM3D on the DTC thesis objects")
    parser.add_argument("--objects", nargs="*", default=OBJECTS)
    parser.add_argument("--prompt-source", default="sam3d", help="recgen | sam3d | path to a prompts yaml")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    failed = []
    for obj in args.objects:
        out = (args.out_root / obj).resolve()
        if (out / "mesh.glb").exists() and not args.overwrite:
            print(f"[sam3d] {obj}: mesh.glb exists, skipped")
            continue
        base = yaml.safe_load(open(SAM3D_ROOT / "sam3d_configs" / f"{obj}.yaml"))
        image_path = prompt_image(obj, args.prompt_source)
        cfg = {**base, "category": obj, "image_path": str(image_path), "output_path": str(out)}
        print(f"[sam3d] {obj}: {image_path}")
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(cfg, f)
        result = subprocess.run([sys.executable, str(SAM3D_ROOT / "scripts" / "run_sam3d_config.py"),
                                 "--yaml-config", f.name], cwd=SAM3D_ROOT, check=False)
        Path(f.name).unlink(missing_ok=True)
        if result.returncode != 0:
            failed.append(obj)
            print(f"[sam3d] {obj}: FAILED (exit {result.returncode})")
    if failed:
        raise SystemExit(f"failed: {failed}")


if __name__ == "__main__":
    main()
