import os
os.environ['ATTN_BACKEND'] = 'xformers'   # or flash-attn if you want
os.environ['SPCONV_ALGO'] = 'native'

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import yaml
import json
import imageio

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.renderers.mesh_renderer import MeshRenderer
from trellis.models.structured_latent_vae.decoder_mesh import SLatMeshDecoder
from trellis.utils import render_utils
from trellis.modules.sparse import SparseTensor
from trellis.representations.mesh.cube2mesh import SparseFeatures2Mesh
from trellis.utils.loss_utils import l1_loss, smooth_l1_loss, ssim, lpips
from opt_utils.utility import save_image
from typing import Dict, List, Optional, Union
import trimesh

try:
    import utils3d.torch
except Exception:  # pragma: no cover
    utils3d = None

RES = 512
DECODER_RES = 64

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def _try_image_path(base_path_no_ext: Path) -> Path:
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        cand = base_path_no_ext.with_suffix(ext)
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Could not find image for: {base_path_no_ext}")

def resize_rgba(path, res, device):
    img = Image.open(path).convert("RGBA")
    if img.size != (res, res):
        img = img.resize((res, res), Image.Resampling.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).to(device)
    return t.permute(2,0,1).contiguous()


def _load_json(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)





def scale_intrinsics(intr, orig_wh, res):
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    ow, oh = orig_wh
    return torch.tensor([
        [fx/ow, 0, cx/ow],
        [0, fy/oh, cy/oh],
        [0, 0, 1]
    ], dtype=torch.float32)

def refine_latent_minimal(config_path: Path):
    config_yaml = yaml.safe_load(open(config_path, 'r'))
    
    lr = float(config_yaml['lr'])
    num_iters = int(config_yaml['num_iters'])
    transforms_json_path = Path(config_yaml['transforms_json'])
    frames_per_iter = int(config_yaml['frames_per_iter'])
    near = float(config_yaml.get('near', 0.01))
    far = float(config_yaml.get('far', 100.0))
    ssaa = int(config_yaml.get('ssaa', 1))
    weight = float(config_yaml.get('weight', 5.0))
    sdf_weight = float(config_yaml.get('sdf_weight', 10.0))
    display_interval = int(config_yaml.get('display_interval', 10))
    seed = int(config_yaml.get("seed", 1))
    gt_mesh = Path(config_yaml.get("gt_mesh", ""))
    out_dir = config_yaml.get("out_dir", "optimization_output")
    Path(out_dir).mkdir(parents=True, exist_ok=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transforms = _load_json(transforms_json_path)
    frames = transforms["frames"]
    base_object_pose = torch.tensor(transforms["object_pose"], dtype=torch.float32, device=device)
    base_dir = transforms_json_path.parent  
    image_basenames = [f["file_path"] for f in frames]
    image_paths: List[Path] = []
    for base_name in image_basenames:
        base_no_ext = base_dir / base_name
        image_paths.append(_try_image_path(base_no_ext))

    intr = transforms["intrinsic"]
    prompt_idx = int(config_yaml.get("prompt_idx", np.random.randint(0, len(image_paths))))

    print("Loading Trellis2 ImageTo3DPipeline...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()

    prompt_img = Image.open(image_paths[prompt_idx])

    # Run pipeline
    outputs = pipeline.run(prompt_img, seed=1, return_latent=True)
    slat = outputs["latent"]
    coords = slat.coords
    mesh0 = outputs["mesh"][0]

    # Camera intrinsics/extrinsics
    intr = transforms["intrinsic"]
    intr_scaled = scale_intrinsics(intr, prompt_img.size, RES).to(device)

    extrinsics = torch.stack([torch.tensor(f["transform_matrix"], dtype=torch.float32) for f in frames]).to(device)

    # Renderer
    renderer = MeshRenderer(
        rendering_options={"resolution": RES, "near": 0.01, "far": 100.0, "ssaa": 1},
        device=device.type,
    )

    # Decoder
    decoder_args = {
        "resolution": DECODER_RES,
        "model_channels": 768,
        "latent_channels": 8,
        "num_blocks": 12,
        "num_heads": 12,
        "mlp_ratio": 4,
        "attn_mode": "swin",
        "window_size": 8,
        "use_fp16": True,
        "representation_config": {"use_color": True},
    }
    decoder = SLatMeshDecoder(**decoder_args).to(device)
    decoder.load_state_dict(pipeline.models["slat_decoder_mesh"].state_dict())
    decoder.eval()

    # Output directory
    out_dir = config_yaml.get("out_dir", "jitter_output")
    Path(out_dir).mkdir(exist_ok=True)

    # SLAT features to modify
    slat_feats = slat.feats.clone().to(device)

    print("Starting minimal jitter loop...")
    for it in range(10):
        # Add small random noise
        noise = 0.01* torch.randn_like(slat_feats)
        slat_jitter = SparseTensor(feats=slat_feats + noise, coords=coords)
        # slat.replace(feats=slat_feats + noise)

        # Decode to mesh
        mesh = decoder(slat_jitter)[0]

        # Extract vertices & faces (Trellis mesh stores them as torch tensors)
        verts = mesh.vertices.detach().cpu().numpy()
        faces = mesh.faces.detach().cpu().numpy()

        # Build trimesh object
        tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        # Save mesh
        mesh_path = f"{out_dir}/jitter_{it:03d}.obj"
        tm.export(mesh_path)

        print(f"Saved mesh: {mesh_path}")


    print("Done.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml-config", type=Path, required=True)
    args = parser.parse_args()
    refine_latent_minimal(args.yaml_config)
