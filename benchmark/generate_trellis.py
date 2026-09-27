"""Run TRELLIS (image-large) on the DTC thesis objects and save its official GLB export."""
import os

os.environ.setdefault("SPCONV_ALGO", "native")

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import OBJECTS, PROJECTS, prompt_image

TRELLIS_ROOT = PROJECTS / "trellis1" / "TRELLIS"
sys.path.insert(0, str(TRELLIS_ROOT))

import imageio
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="TRELLIS on the DTC thesis objects")
    parser.add_argument("--objects", nargs="*", default=OBJECTS)
    parser.add_argument("--prompt-source", default="recgen",
                        help="recgen | sam3d | path to a prompts yaml (benchmark/prompts.yaml)")
    parser.add_argument("--out-root", type=Path, default=TRELLIS_ROOT / "trellis_output")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    todo = [o for o in args.objects if args.overwrite or not (args.out_root / o / "mesh.glb").exists()]
    for o in set(args.objects) - set(todo):
        print(f"[trellis] {o}: mesh.glb exists, skipped")
    if not todo:
        return

    import torch
    cap = torch.cuda.get_device_capability()
    if cap < (9, 0):
        raise SystemExit(
            f"GPU {torch.cuda.get_device_name()} (sm_{cap[0]}{cap[1]}) cannot run this env's Gaussian "
            "rasterizer (built for sm_90 only). Run on an H100 node (gpu_h100), or rebuild it for this GPU:\n"
            "  TORCH_CUDA_ARCH_LIST='8.0;9.0' pip install --no-build-isolation --force-reinstall "
            "/tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/")

    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import render_utils, postprocessing_utils
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()

    failed = []
    for obj in todo:
        out = args.out_root / obj
        out.mkdir(parents=True, exist_ok=True)
        image_path = prompt_image(obj, args.prompt_source)
        print(f"[trellis] {obj}: {image_path}")
        try:
            start = time.time()
            outputs = pipeline.run(Image.open(image_path), seed=args.seed)
            runtime = time.time() - start
            glb = postprocessing_utils.to_glb(outputs['gaussian'][0], outputs['mesh'][0],
                                              simplify=0.95, texture_size=1024)
            glb.export(str(out / "mesh.glb"))
            outputs['gaussian'][0].save_ply(str(out / "gaussian.ply"))
            if not args.no_video:
                video = render_utils.render_video(outputs['gaussian'][0])['color']
                imageio.mimsave(str(out / "preview.mp4"), video, fps=30)
            (out / "metadata.json").write_text(json.dumps({
                "category": obj, "method": "TRELLIS (microsoft/TRELLIS-image-large)",
                "image_path": str(image_path), "prompt_source": args.prompt_source, "seed": args.seed,
                "runtime_seconds": runtime, "export": "postprocessing_utils.to_glb(simplify=0.95, texture_size=1024)",
                "outputs": ["mesh.glb", "gaussian.ply"] + ([] if args.no_video else ["preview.mp4"]),
            }, indent=2))
            print(f"[trellis] {obj}: done in {runtime:.1f}s -> {out / 'mesh.glb'}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            failed.append(obj)
            print(f"[trellis] {obj}: FAILED ({e})")
    if failed:
        raise SystemExit(f"failed: {failed}")


if __name__ == "__main__":
    main()
