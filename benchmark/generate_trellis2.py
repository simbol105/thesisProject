"""Run TRELLIS.2 on the DTC thesis objects and save its official GLB export."""
import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import OBJECTS, PROJECTS, prompt_image

TRELLIS2_ROOT = PROJECTS / "trellis2" / "TRELLIS.2"
sys.path.insert(0, str(TRELLIS2_ROOT))

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="TRELLIS.2 on the DTC thesis objects")
    parser.add_argument("--objects", nargs="*", default=OBJECTS)
    parser.add_argument("--prompt-source", default="recgen",
                        help="recgen | sam3d | path to a prompts yaml (benchmark/prompts.yaml)")
    parser.add_argument("--out-root", type=Path, default=TRELLIS2_ROOT / "trellis2_output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pipeline-type", default=None,
                        help="512 | 1024 | 1024_cascade | 1536_cascade (default: the model's default)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    todo = [o for o in args.objects if args.overwrite or not (args.out_root / o / "mesh.glb").exists()]
    for o in set(args.objects) - set(todo):
        print(f"[trellis2] {o}: mesh.glb exists, skipped")
    if not todo:
        return

    import o_voxel
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()
    pipeline_type = args.pipeline_type or pipeline.default_pipeline_type

    failed = []
    for obj in todo:
        out = args.out_root / obj
        out.mkdir(parents=True, exist_ok=True)
        image_path = prompt_image(obj, args.prompt_source)
        print(f"[trellis2] {obj}: {image_path} (pipeline {pipeline_type})")
        try:
            start = time.time()
            mesh = pipeline.run(Image.open(image_path), seed=args.seed, pipeline_type=pipeline_type)[0]
            runtime = time.time() - start
            mesh.simplify(16777216)
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
                attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=1000000, texture_size=4096, remesh=True, remesh_band=1, remesh_project=0,
                verbose=True,
            )
            glb.export(str(out / "mesh.glb"), extension_webp=False)
            (out / "metadata.json").write_text(json.dumps({
                "category": obj, "method": "TRELLIS.2 (microsoft/TRELLIS.2-4B)",
                "image_path": str(image_path), "prompt_source": args.prompt_source, "seed": args.seed,
                "pipeline_type": pipeline_type, "runtime_seconds": runtime,
                "export": "o_voxel.postprocess.to_glb(decimation_target=1e6, texture_size=4096, remesh=True)",
                "outputs": ["mesh.glb"],
            }, indent=2))
            print(f"[trellis2] {obj}: done in {runtime:.1f}s -> {out / 'mesh.glb'}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            failed.append(obj)
            print(f"[trellis2] {obj}: FAILED ({e})")
        finally:
            import torch
            torch.cuda.empty_cache()
    if failed:
        raise SystemExit(f"failed: {failed}")


if __name__ == "__main__":
    main()
