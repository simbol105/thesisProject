from json import decoder
import os
os.environ['ATTN_BACKEND'] =  'flash-attn'    # 'flash-attn'  # 'xformers' # 
os.environ['SPCONV_ALGO'] = 'native' 
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))


from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
from trellis.renderers.mesh_renderer import MeshRenderer
from opt_utils.utility import display_image, save_image, align_meshes_similarity
from trellis.representations.mesh.cube2mesh import SparseFeatures2Mesh, MeshExtractResult
from trellis.models.structured_latent_vae.decoder_mesh import SLatMeshDecoder, ElasticSLatMeshDecoder
from trellis.modules import sparse as sp
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import imageio
from PIL import Image

from trellis.utils.loss_utils import l1_loss, smooth_l1_loss, ssim, lpips, l2_loss

try:
    import utils3d.torch
except Exception:  # pragma: no cover
    utils3d = None


import argparse
import yaml
import json
from typing import List, Dict, Tuple, Optional, Union
import trimesh

# Global settings 
RES = 512 # The render/grid resolution
DECODER_RES = 64
TEST_INDEX = 4
TEST_INTERVALS = [0, 1, 50, 100, 200,  500, 799, 999]


def _to_nchw_3ch(x):
    if x is None:
        return None
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)

    if x.dim() == 4:
        if x.shape[1] == 3:
            return x
        if x.shape[-1] == 3:
            return x.permute(0, 3, 1, 2)
    elif x.dim() == 3:
        if x.shape[0] == 3:
            return x.unsqueeze(0)
        if x.shape[-1] == 3:
            return x.permute(2, 0, 1).unsqueeze(0)

    raise ValueError(f"Expected a 3-channel image/normal tensor, got shape={tuple(x.shape)}")


def _perceptual_loss(gt, pred, name: str, lambda_ssim: float = 0.2, lambda_lpips: float = 0.2):
    gt = _to_nchw_3ch(gt)
    pred = _to_nchw_3ch(pred)
    terms = {
        f"{name}_loss": l1_loss(gt, pred),
        f"{name}_loss_ssim": 1.0 - ssim(gt, pred),
        f"{name}_loss_lpips": lpips(gt, pred),
    }
    terms[f"{name}_loss_perceptual"] = (
        terms[f"{name}_loss"]
        + terms[f"{name}_loss_ssim"] * lambda_ssim
        + terms[f"{name}_loss_lpips"] * lambda_lpips
    )
    return terms


def _tsdf_reg_loss_from_depth(rep, depth_map: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    if utils3d is None:
        raise ImportError("utils3d.torch is required to compute the TSDF term.")
    with torch.no_grad():
        projected_pts, pts_depth = utils3d.torch.project_cv(
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            points=rep.tsdf_v,
        )
        projected_pts = (projected_pts - 0.5) * 2.0
        depth_map_res = depth_map.shape[-1]
        gt_depth = torch.nn.functional.grid_sample(
            depth_map.reshape(1, 1, depth_map_res, depth_map_res),
            projected_pts.reshape(1, 1, -1, 2),
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )
        pseudo_sdf = gt_depth.flatten() - pts_depth.flatten()
        delta = 1.0 / rep.res * 3.0
        trunc_mask = pseudo_sdf > -delta

    gt_tsdf = pseudo_sdf[trunc_mask]
    tsdf = rep.tsdf_s.flatten()[trunc_mask]
    gt_tsdf = torch.clamp(gt_tsdf, -delta, delta)
    return torch.mean((tsdf - gt_tsdf) ** 2)


def tsdf_reg_loss(
    gt_mesh: Union[MeshExtractResult, List[MeshExtractResult]],
    rep: Union[MeshExtractResult, List[MeshExtractResult]],
    extrinsics: Union[torch.Tensor, List[torch.Tensor]],
    intrinsics: Union[torch.Tensor, List[torch.Tensor]],
    renderer: Optional[MeshRenderer] = None,
) -> torch.Tensor:
    """
    Project the current optimization mesh's TSDF samples into the GT camera view,
    render the GT mesh depth, and regress the current TSDF values to the pseudo-TSDF
    derived from that GT depth. This merges the trainer's `_tsdf_reg_loss` and
    `_calc_tsdf_loss` behavior into a single callable helper.
    """
    if isinstance(rep, (list, tuple)):
        if not isinstance(gt_mesh, (list, tuple)):
            gt_mesh = [gt_mesh] * len(rep)
        if not isinstance(extrinsics, (list, tuple)):
            extrinsics = [extrinsics] * len(rep)
        if not isinstance(intrinsics, (list, tuple)):
            intrinsics = [intrinsics] * len(rep)
        losses = [
            tsdf_reg_loss(g, r, e, i, renderer=renderer)
            for g, r, e, i in zip(gt_mesh, rep, extrinsics, intrinsics)
        ]
        return torch.stack(losses).mean()

    if renderer is None:
        raise ValueError("A renderer is required to render the GT depth map from gt_mesh.")
    if getattr(rep, "tsdf_v", None) is None or getattr(rep, "tsdf_s", None) is None:
        raise ValueError("Current mesh must expose tsdf_v and tsdf_s to compute TSDF loss.")

    gt_buffers = renderer.render(
        mesh=gt_mesh,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        return_types=["depth"],
    )
    depth_map = gt_buffers["depth"]
    return _tsdf_reg_loss_from_depth(rep, depth_map, extrinsics, intrinsics)


def compute_slat_mesh_decoder_losses(
    gt_rgb: torch.Tensor,
    gt_mask: torch.Tensor,
    gt_normal: torch.Tensor,
    gt_depth: torch.Tensor,
    pred_rgb: torch.Tensor,
    pred_mask: torch.Tensor,
    pred_normal: torch.Tensor,
    pred_depth: torch.Tensor,
    *,
    depth_loss_type: str = 'l1',
    lambda_depth: float = 1.0,
    lambda_ssim: float = 0.2,
    lambda_lpips: float = 0.2,
    lambda_color: float = 0.1,
    use_color: bool = True,
):
    """
    Same logic as your original version, but now takes explicit tensors
    and does NOT multiply anything by gt_mask.
    """
    ref_rgb = gt_rgb
    ref_normal_map = None
    pred_normal_map = None

    if ref_rgb is None or pred_rgb is None:
        raise ValueError("Both GT and predicted RGB buffers must be supplied.")

    device = ref_rgb.device
    terms = {"loss": torch.tensor(0.0, device=device, dtype=ref_rgb.dtype)}

    # mask loss (unchanged)
    if gt_mask is not None and pred_mask is not None:
        terms["mask_loss"] = l1_loss(pred_mask, gt_mask)
        terms["loss"] = terms["loss"] + terms["mask_loss"]

    # depth loss (NO mask multiplication)
    if gt_depth is not None and pred_depth is not None:
        if depth_loss_type == 'l1':
            terms["depth_loss"] = l1_loss(pred_depth, gt_depth)
        elif depth_loss_type == 'smooth_l1':
            terms["depth_loss"] = smooth_l1_loss(pred_depth, gt_depth, beta=1.0 / 128.0)
        else:
            raise ValueError(f"Unsupported depth loss type: {depth_loss_type}")

        terms["loss"] = terms["loss"] + terms["depth_loss"] * lambda_depth

    # normal loss (NO mask multiplication)
    if gt_normal is not None and pred_normal is not None:
        normal_terms = _perceptual_loss(pred_normal, gt_normal,
                                        "normal", lambda_ssim, lambda_lpips)
        terms.update(normal_terms)
        terms["loss"] = terms["loss"] + normal_terms["normal_loss_perceptual"]

    # normal_map loss (unchanged)
    if use_color and ref_normal_map is not None and pred_normal_map is not None:
        normal_map_terms = _perceptual_loss(ref_normal_map, pred_normal_map,
                                            "normal_map", lambda_ssim, lambda_lpips)
        terms.update(normal_map_terms)
        terms["loss"] = terms["loss"] + normal_map_terms["normal_map_loss_perceptual"] * lambda_color

    # color loss (NO mask multiplication)
    if use_color:
        color_terms = _perceptual_loss(ref_rgb, pred_rgb,
                                       "color", lambda_ssim, lambda_lpips)
        terms.update(color_terms)
        terms["loss"] = terms["loss"] + color_terms["color_loss_perceptual"] * lambda_color

    return terms


def rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.eye(4, dtype=np.float32)
    R[1,1], R[1,2] =  c, -s
    R[2,1], R[2,2] =  s,  c
    return R

def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.eye(4, dtype=np.float32)
    R[0,0], R[0,1] =  c, -s
    R[1,0], R[1,1] =  s,  c
    return R

def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.eye(4, dtype=np.float32)
    R[0,0], R[0,2] =  c,  s
    R[2,0], R[2,2] = -s,  c
    return R

def load_image(path):
    image = Image.open(path)
    image = np.array(image)
    image = image.astype(np.uint8)
    return image

# rot_x(np.pi) @ 
AXIS_SWAP =   rot_z(np.pi / 2) 
# AXIS_SWAP = np.eye(4, dtype=np.float32)


def dilate_sparse_tensor(  
    x: sp.SparseTensor,  
    res: int,                        # MUST equal SparseFeatures2Mesh.res exactly (e.g. self.res, not res+1)  
    radius: int = 1,  
    kernel: Optional[Tuple[Tuple[int, int, int], ...]] = None,  
    sdf_channel: int = 0,  
    sdf_outside_value: float = 1.0,  
    fill_value: float = 0.0,  
) -> sp.SparseTensor:  
    if not isinstance(x, sp.SparseTensor):  
        raise TypeError(f"Expected a SparseTensor, got {type(x)}")  
  
    coords = x.coords  
    feats = x.feats  
    if coords.numel() == 0:  
        return x  
    if coords.dim() != 2 or coords.shape[1] != 4:  
        raise ValueError(f"Expected coords [N,4] (batch,x,y,z), got shape={tuple(coords.shape)}")  
  
    # Sanity: input coords must already be within [0, res-1]  
    max_in = coords[:, 1:].max().item()  
    if max_in > res - 1:  
        raise ValueError(  
            f"Input coords already exceed valid range: max={max_in}, allowed max={res - 1}. "  
            f"Check that `res` matches SparseFeatures2Mesh.res, not res+1."  
        )  
  
    if kernel is None:  
        offsets = [  
            (dx, dy, dz)  
            for dz in range(-radius, radius + 1)  
            for dy in range(-radius, radius + 1)  
            for dx in range(-radius, radius + 1)  
            if not (dx == 0 and dy == 0 and dz == 0)  
        ]  
        kernel = tuple(offsets)  
  
    kernel_t = torch.as_tensor(kernel, device=coords.device, dtype=coords.dtype)  
    RES_MIN = 0
    # Use the maximum spatial coordinate present in the input SLAT as the upper bound
    # This prevents dilation from creating coordinates outside the original SLAT grid
    # even if the caller passes a larger `res` (e.g., render resolution).
    RES_MAX = int(coords[:, 1:].max().item())
  
    new_coords_list = []  
    for b in coords[:, 0].unique().tolist():  
        batch_mask = coords[:, 0] == b  
        batch_coords = coords[batch_mask][:, 1:]  # [M, 3]  
  
        shifted = (batch_coords.unsqueeze(1) + kernel_t.unsqueeze(0)).reshape(-1, 3)  
  
        # --- Clip strictly to [0, res-1] on ALL three axes independently ---  
        valid = (shifted >= RES_MIN).all(dim=1) & (shifted <= RES_MAX).all(dim=1)  
        shifted = shifted[valid]  
        if shifted.numel() == 0:  
            continue  
  
        batch_candidates = torch.cat(  
            [torch.full((shifted.shape[0], 1), b, device=coords.device, dtype=coords.dtype), shifted],  
            dim=1,  
        )  
        batch_candidates = torch.unique(batch_candidates, dim=0)  
  
        # Vectorized removal of coords already present in this batch  
        existing_batch = coords[batch_mask]  
        combined = torch.cat([existing_batch, batch_candidates], dim=0)  
        uniq, inverse, counts = torch.unique(combined, dim=0, return_inverse=True, return_counts=True)  
        # rows that came ONLY from batch_candidates (count==1) are genuinely new  
        is_new_uniq = counts == 1  
        cand_inverse = inverse[existing_batch.shape[0]:]  
        new_only = batch_candidates[is_new_uniq[cand_inverse]]  
  
        if new_only.numel() > 0:  
            new_coords_list.append(new_only)  
  
    if not new_coords_list:  
        return x  
  
    new_coords = torch.cat(new_coords_list, dim=0)
    # Ensure dilated coordinates do not step outside the original SLAT spatial range [0, RES_MAX].
    new_coords[:, 1:] = new_coords[:, 1:].clamp(0, RES_MAX)
  
    # --- Hard assertion: catch any bug before it reaches SparseTensor/get_dense_attrs ---  
    assert new_coords[:, 1:].min().item() >= 0, "Negative coordinate after dilation"
    assert new_coords[:, 1:].max().item() <= RES_MAX, (
        f"Out-of-range coordinate after dilation: max={new_coords[:, 1:].max().item()}, "
        f"allowed max={RES_MAX}"
    )
  
    new_feats = torch.full(  
        (new_coords.shape[0],) + tuple(feats.shape[1:]),  
        fill_value,  
        device=feats.device,  
        dtype=feats.dtype,  
    )  
    new_feats[:, sdf_channel] = sdf_outside_value  
  
    merged_coords = torch.cat([coords, new_coords], dim=0).to(dtype=torch.int32)  
    merged_feats = torch.cat([feats, new_feats], dim=0)  
  
    # Final, redundant guard (kept intentionally paranoid)  
    bad_mask = ~(((merged_coords[:, 1:] >= 0) & (merged_coords[:, 1:] <= RES_MAX)).all(dim=1))
    if bad_mask.any():
        raise IndexError(f"Found out-of-range coordinates after dilation (RES_MAX={RES_MAX}): {merged_coords[bad_mask][:5]}")
  
    # Sort for batch-contiguity, as required by SparseTensor (see DEBUG assert in basic.py)  
    key = (  
        merged_coords[:, 0].to(torch.int64) * (res ** 3)  
        + merged_coords[:, 1].to(torch.int64) * (res ** 2)  
        + merged_coords[:, 2].to(torch.int64) * res  
        + merged_coords[:, 3].to(torch.int64)  
    )  
    order = torch.argsort(key)  
    merged_coords = merged_coords[order].contiguous()  
    merged_feats = merged_feats[order].contiguous()  
    
    # Mask for the new tensors
    is_new_mask = torch.zeros(coords.shape[0] + new_coords.shape[0], dtype=torch.bool, device=feats.device)
    is_new_mask[coords.shape[0]:] = True # Mark the appended tokens as true
    is_new_mask = is_new_mask[order] 
  
    if merged_feats.shape[0] != merged_coords.shape[0]:  
        raise RuntimeError("Merged feats and coords have mismatched lengths")  
  
    return sp.SparseTensor(feats=merged_feats, coords=merged_coords), is_new_mask

def _scale_opencv_intrinsics(intr: Dict, orig_wh: Tuple[int, int], render_resolution: int) -> torch.Tensor:
    fx, fy, cx, cy = float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"])
    orig_w, orig_h = orig_wh

    fx_n = fx / float(orig_w)
    fy_n = fy / float(orig_h)
    cx_n = cx / float(orig_w)
    cy_n = cy / float(orig_h)

    intr_normalized = torch.tensor(
        [[fx_n, 0.0, cx_n], [0.0, fy_n, cy_n], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    return intr_normalized

def _load_json(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)

def _try_image_path(base_path_no_ext: Path) -> Path:
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        cand = base_path_no_ext.with_suffix(ext)
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Could not find image for: {base_path_no_ext}")

def _resize_rgb_to_tensor(img_path: Path, render_resolution: int, device: torch.device) -> torch.Tensor:
    img = Image.open(img_path).convert("RGBA")
    if img.size != (render_resolution, render_resolution):
        img = img.resize((render_resolution, render_resolution), Image.Resampling.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0  
    t = torch.from_numpy(arr).to(device=device, dtype=torch.float32)  
    t = t.permute(2, 0, 1).contiguous()  
    return t   

def ensure_nchw(x):
    """Normalize tensor to NCHW format.

    Accepts tensors in shapes:
    - (C, H, W)
    - (H, W, C)
    - (B, C, H, W)
    - (B, H, W, C)
    - (H, W)
    - (C, H, W) with no batch dim
    Returns tensor with shape (B, C, H, W).
    """
    if x is None:
        return None
    # ensure tensor
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)

    if x.dim() == 4:
        # could be (B, C, H, W) or (B, H, W, C)
        B, a, b, c = x.shape
        # If last dim is small (<=4) and second dim is large, assume NHWC
        if c <= 4 and a > c:
            return x.permute(0, 3, 1, 2)
        # If second dim is small (<=4) and looks like channels, assume NCHW
        if a <= 4 and a <= 16:
            return x
        # Fallback: if last dim is small, permute
        if c <= 16:
            return x.permute(0, 3, 1, 2)
        return x
    elif x.dim() == 3:
        # could be (C, H, W), (H, W, C), or (B, H, W)
        a, b, c = x.shape
        # HWC: last dim small
        if c <= 4 and a > c:
            return x.permute(2, 0, 1).unsqueeze(0)
        # CHW: first dim small
        if a <= 4 and b > 16 and c > 16:
            return x.unsqueeze(0)
        # B,H,W -> add channel
        return x.unsqueeze(1)
    elif x.dim() == 2:
        # single-channel image (H, W)
        return x.unsqueeze(0).unsqueeze(0)

    return x

def refine_latent(config_path: Path):
    config_yaml = yaml.safe_load(open(config_path, 'r'))

    lr = float(config_yaml['lr'])
    num_iters = int(config_yaml['num_iters'])
    transforms_json_path = Path(config_yaml['transforms_json'])
    frames_per_iter = int(config_yaml['frames_per_iter'])
    near = float(config_yaml.get('near', 0.01))
    far = float(config_yaml.get('far', 100.0))
    ssaa = int(config_yaml.get('ssaa', 1))
    weight = float(config_yaml.get('weight', 5.0))
    sdf_weight = float(config_yaml.get('sdf_weight', 1.0))
    tsdf_weight = float(config_yaml.get('tsdf_weight', 1.0))
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
    multi_images = [
        prompt_img,
        Image.open(image_paths[1]),
        Image.open(image_paths[96]),
        Image.open(image_paths[20]),
        Image.open(image_paths[36]),
        Image.open(image_paths[36]),
    ]
    
    # images = [
    #     Image.open("assets/example_multi_image/character_1.png"),
    #     Image.open("assets/example_multi_image/character_2.png"),
    #     Image.open("assets/example_multi_image/character_3.png"),
    # ]
    # run_multi_image
    outputs = pipeline.run_multi_image(
        prompt_img,
        # multi_images,
        seed=seed,
        # Optional parameters
        sparse_structure_sampler_params={
            "steps": 12,
            "cfg_strength": 7.5,
        },
        slat_sampler_params={
            "steps": 12,
            "cfg_strength": 5,
        },
        return_latent=True,
    )
    
    orig_w, orig_h = prompt_img.size
    intr_scaled = _scale_opencv_intrinsics(intr, (orig_w, orig_h), RES).to(device=device)

    # camera extrinsics
    extrinsics = []
    for f in frames:
        extrinsics.append(torch.tensor(f["transform_matrix"], dtype=torch.float32))
    
    # setup renderer
    renderer = MeshRenderer(
        rendering_options={
                    "resolution": RES,
                    "near": near,
                    "far": far,
                    "ssaa": ssaa,
                },
                device=device.type,
    )
    

    slat = outputs['latent']
    mesh = outputs['mesh'][0]
    
    # print("Slat_ max, min: ", slat_.coords.min(), slat_.coords.max())
    
    
    # print(f"Lantent SLAT shape: {slat.shape}")
    # print(f"Lantent SLAT features: {slat.feats.shape}")
    # print("Latent SLAT coords: ", slat.coords.shape)
    # print("Latent SLAT layout: ", slat.layout)
    # print("Latent SLAT data: ", slat.data)
    
    
    
    '''
    DILATION of Input SLAT
    '''
    # slat, slat_mask  = dilate_sparse_tensor(  
    #     x = slat_,  
    #     res = DECODER_RES * 4,                      # must match SparseFeatures2Mesh's res  
    #     radius = 1,  
    #     kernel = None,  
    #     sdf_channel = 0,          # index of the sdf channel in feats  
    #     sdf_outside_value  = 1.0,  # matches get_dense_attrs' "outside" default  
    #     fill_value =  1.0,  
    # )
    
    
    
    
    
    
    
    # cond = pipeline.get_cond([prompt_img])
    # slat = pipeline.sample_slat(cond, slat_dilated.coords, {})


    
 
 
    # print("Latent SLAT data: ", slat.data)
    slat_opt = nn.Parameter(slat.feats.clone().detach())
    slat_opt.requires_grad_(True)

    
    extrinsics_t = torch.stack(extrinsics, dim=0).to(device=device)
    ref_images = torch.stack(
        [_resize_rgb_to_tensor(p, RES, device) for p in image_paths], dim=0
    )
    N = len(image_paths)
    rng = np.random.default_rng(seed=seed)

    if gt_mesh.exists():
        dtc_mesh = trimesh.load_mesh(gt_mesh)
        with torch.no_grad():
            dtc_mesh.apply_transform(base_object_pose.detach().cpu().numpy())
            alignment, _ = align_meshes_similarity(mesh, dtc_mesh) 
            alignment = torch.from_numpy(alignment   @ AXIS_SWAP).to(device=device, dtype=torch.float32) # @ AXIS_SWAP
        
        dtc_mesh.visual = dtc_mesh.visual.to_color()
        colors = np.array(dtc_mesh.visual.vertex_colors)[:, :3] / 255.0
        attrs = torch.from_numpy(colors).float().to(device)
        dtc_mesh = MeshExtractResult(
            vertices=torch.from_numpy(dtc_mesh.vertices).to(device=device, dtype=torch.float32),
            faces=torch.from_numpy(dtc_mesh.faces).to(device=device, dtype=torch.long),
            vertex_attrs=attrs,
        )


        
    else:
        print("No ground truth mesh provided, skipping alignment step.")
        alignment = torch.eye(4, device=device) 
    object_pose = alignment #base_object_pose @ alignment
    print(" Starting latent optimization...")

    l1_criterion = nn.L1Loss(reduction="mean")
    l2_criterion = nn.MSELoss(reduction="mean")
    # gt_mask = None
    
    # decoder = pipeline.models['slat_decoder_mesh']
    # decoder.train()
    # print(pipeline.models.keys())
    # set up the decoder
    # decoder = SparseFeatures2Mesh(
    #     device=device.type, res=RES, use_color=True
    # )
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
        "representation_config": { "use_color": True }  
    }
    # decoder = pipeline.models['slat_decoder_mesh']
    decoder = SLatMeshDecoder(**decoder_args)
    decoder.load_state_dict(pipeline.models['slat_decoder_mesh'].state_dict())

    decoder.to(device=device)

    decoder.train()


    # decoder = MLPUpsampler(
    #     resolution=DECODER_RES,
    #     latent_channels=8,
    #     hidden=64,
    #     representation_config={"use_color": True}
    #     ).to(device)

    
    
    # setup optimizer 
    if config_yaml.get("optimizer", "") == "sgd":
        optimizer = torch.optim.SGD([slat_opt] , lr=lr) # + list(decoder.parameters())
        # decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=0.001*lr)
    else: 
        optimizer = torch.optim.Adam([slat_opt] , lr=lr)
        # decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001*lr)
        
    print("Decoder in training mode: ", decoder.training)
    for i in range(num_iters + 1):
        indices = (
            np.arange(N)
            if frames_per_iter is None or frames_per_iter == N
            else np.atleast_1d(rng.choice(N, size=frames_per_iter, replace=False))
        )
        optimizer.zero_grad()
       
        # mean_slat = slat_opt.min()
        # std_slat = slat_opt.std()
        # slat_opt = slat_opt *std_slat + mean_slat
        
        x = sp.SparseTensor(feats=slat_opt, coords=slat.coords) 
        
        
        
   
        
        meshes = decoder.forward(x)
        mesh = meshes[0]
        # print("Mesh sucess: \n", mesh.success)
        
        reg_loss = torch.zeros((), device=device)
        reg_loss += sum([rep.reg_loss for rep in meshes]) / len(meshes)
        print(f"[Iter {i:03d}] Reg loss: {float(reg_loss.detach().cpu()):.4f}")

       
       
        # if hasattr(mesh, "sdf") and mesh.sdf is not None:
        #     mesh.sdf  = torch.clamp(mesh.sdf, -0.1, 0.1)

        # print("SDF reg loss: ", mesh.reg_loss.detach().cpu())

        total_loss = torch.zeros((), device=device)
        tsdf_loss = torch.zeros((), device=device)
        img_losses = torch.zeros((), device=device)
        # Add the loss from the mesh
        
        for idx in indices: 
            
            
            extr = extrinsics_t[idx ] @ object_pose
            
            if gt_mesh.exists():
                gt_buffers = renderer.render(
                    mesh = dtc_mesh, 
                    extrinsics=extrinsics_t[idx], #@  base_object_pose, #torch.from_numpy(AXIS_SWAP).to(device),
                    intrinsics=intr_scaled,
                    return_types=["color", "mask", "normal", "depth"]
                )
                
                gt_mask = gt_buffers["mask"]
                ref_rgb = gt_buffers["color"] * gt_mask
                
                
                gt_normal = gt_buffers["normal"] * gt_mask
                gt_depth = gt_buffers["depth"] * gt_mask
                            
            buffers = renderer.render(
                mesh=mesh, 
                extrinsics=extr,
                intrinsics=intr_scaled,
                # return_types=["color", "mask", "normal"]
                return_types=[ "color", "mask", "normal", "depth"]
            )
            pred_mask = buffers["mask"]
            pred_rgb = buffers["color"] 
            pred_normal = buffers["normal"] 
            pred_depth = buffers["depth"] 
            
            
            if gt_mesh.exists():
                # pred_rgb = pred_rgb * gt_mask
                # pred_normal = pred_normal * gt_mask
                # pred_depth = pred_depth * gt_mask

                tsdf_loss += tsdf_reg_loss(
                    gt_mesh=dtc_mesh,
                    rep=mesh,
                    extrinsics=extr,
                    intrinsics=intr_scaled,
                    renderer=renderer,
                )
                # print("tsdf loss: \n", tsdf_loss.detach().cpu())
                # loss = l2_loss(ref_rgb , pred_rgb * gt_mask)
                # loss += l2_loss(pred_normal * gt_mask , gt_normal)
                # loss += l1_loss(pred_mask , gt_mask)
                # loss += l1_loss(pred_depth * gt_mask , gt_depth)
                

                loss_terms = compute_slat_mesh_decoder_losses(
                    gt_rgb=ref_rgb,
                    gt_mask=gt_mask,
                    gt_normal=gt_normal,
                    gt_depth=gt_depth,
                    pred_depth=pred_depth * gt_mask,
                    pred_rgb=pred_rgb * gt_mask,
                    pred_mask=pred_mask,
                    pred_normal=pred_normal * gt_mask,
                    lambda_depth=1.2,
                    lambda_ssim=0.2,
                    lambda_lpips=0.2,
                    lambda_color=0.9,
                    use_color=True,
                )
                
                

                img_losses  += loss_terms["loss"]
            else:
                ref_rgb = ref_images[idx][:3]#.unsqueeze(0)
                gt_mask = ref_images[idx][3:] if ref_images.shape[1] == 4 else None
            
                if pred_rgb.ndim == 4:
                    pred_rgb = pred_rgb.squeeze(0)
                    pred_rgb = pred_rgb.permute(2, 0, 1) #.contiguous()
                    
                loss = l2_criterion(pred_rgb, ref_rgb)

            # Prepare the other outputs for visualization. 
            if pred_mask.ndim == 2:
                pred_mask = pred_mask[None, ...]  
                # gt_mask = gt_mask[None, ...]      
            elif pred_mask.shape[0] != 1:
                pred_mask = pred_mask.permute(2, 0, 1)
                # gt_mask = gt_mask.permute(2, 0, 1)
                
            # if pred_depth.ndim == 2:
            #     pred_depth = pred_depth[None, ...]  
            #     # gt_depth = gt_depth[None, ...]      
            # elif pred_depth.shape[0] != 1:
            #     pred_depth = pred_depth.permute(2, 0, 1)
                # gt_depth = gt_depth.permute(2, 0, 1)

            if display_interval and (idx % display_interval == 0 or idx == frames_per_iter - 1):
    
                # --- Normalize shapes to NCHW ---
                pred_rgb_n   = ensure_nchw(pred_rgb)        # (1,3,H,W)
                ref_rgb_n    = ensure_nchw(ref_rgb[:3])     # (1,3,H,W)
                normal_n     = ensure_nchw(pred_normal)     # (1,3,H,W)

                # Normalize normals to [0,1]
                normal_vis = (normal_n - normal_n.min()) / (normal_n.max() - normal_n.min() + 1e-6)

                # Mask → NCHW (1,1,H,W)
                if pred_mask is None:
                    pred_mask_n = torch.ones(1,1,normal_vis.shape[2], normal_vis.shape[3], device=normal_vis.device)
                else:
                    pred_mask_n = ensure_nchw(pred_mask)     # handles HWC, HW, CHW, etc.

                # --- Build visualization ---
                # Concatenate horizontally: [pred | ref | normal]
                vis = torch.cat([pred_rgb_n, ref_rgb_n, normal_vis], dim=3)   # concat along width

                # Convert NCHW → HWC
                vis_hwc = vis[0].permute(1, 2, 0).clamp(0.0, 1.0)

                img_np = vis_hwc.detach().cpu().numpy()
                save_image(f"{out_dir}/opt_{i:04d}_{idx:02d}.png", img_np)




            # total_loss += weight * loss
            # total_loss += sdf_weight * reg_loss # if mesh.reg_loss is not None else 0.0
        img_losses = img_losses / len(indices)
        tsdf_loss = tsdf_loss / len(indices)  
        total_loss = (sdf_weight * reg_loss) + (tsdf_weight * tsdf_loss) + (weight * img_losses)
        print(f"[Iter {i:03d}] Loss: {float(total_loss.detach().cpu()):.4f}")


        # print("rgb_loss:", rgb_loss.item())
        # print("mesh.reg_loss:", mesh.reg_loss.item() if mesh.reg_loss is not None else None)
        # print("total_loss:", total_loss.item())

      

        total_loss.backward()
        # with torch.autograd.detect_anomaly():
        #     total_loss.backward()


        print("SLAT grad: ", slat_opt.grad.norm().item())
        

        # grad clip 
        # torch.nn.utils.clip_grad_norm_([slat_opt], max_norm=1.0)          
                            
        optimizer.step()
        # decoder_optimizer.step()

    print("Latent optimization completed!")

            
                        
    





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters to refine latent")
    parser.add_argument("--yaml-config", type=Path, required=True, help="Path to the YAML config file" )
    args = parser.parse_args()
    refine_latent(Path(args.yaml_config))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
 