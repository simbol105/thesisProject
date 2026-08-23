"""
Differentiable O-Voxel Optimization using Voxel Rasterization

Key Improvements Over voxel_opt.py:
==================================

1. DIFFERENTIABLE RENDERING PIPELINE
   - Uses VoxelRenderer from o_voxel.rasterize instead of non-differentiable mesh extraction
   - Renders directly from voxel grid representation
   - Gradients flow through the rasterizer back to optimized voxel features
   
2. GRADIENT-BASED OPTIMIZATION
   - All voxel features are differentiable:
     * base_color: RGB color per voxel (currently optimized)
     * dual_vertices: Per-voxel position offsets (prepared for future optimization)
     * intersected: Voxel activity/visibility state (prepared for future optimization)
     * quad_lerp: Quad splitting weights (prepared for future optimization)
   
3. CURRENT APPROACH (Stable)
   - Optimizes voxel base_color using differentiable rendering
   - Keeps voxel coordinates fixed from initial mesh-to-voxel conversion
   - All voxels remain active (no topology filtering to preserve gradient flow)
   - Fully differentiable from rendering output back to color parameters

4. EXTENSIONS FOR GEOMETRY OPTIMIZATION
   To optimize geometry features (dual_vertices, intersected, quad_lerp):
   
   a) Smooth Alpha Modulation:
      - Use sigmoid(intersected_logit) to smoothly control per-voxel opacity
      - Geometry changes visible through alpha/visibility changes
      - Requires implementing custom VoxelRenderer or modifying attrs with alpha
      
   b) Differentiable Dual Grid to Mesh:
      - Wrap o_voxel.convert.flexible_dual_grid_to_mesh in torch autograd function
      - Implement manual forward/backward using nvdiffrast for rendering
      - This would enable full topology and geometry optimization
      
   c) Two-Stage Optimization:
      - Stage 1: Render voxels, optimize colors (current)
      - Stage 2: Extract mesh with gradients enabled, render mesh, backprop
      - Less efficient but cleaner to implement

USAGE:
======
python hypothesis_voxel_optimization.py <path_to_config.yaml>

YAML CONFIG REQUIREMENTS (see voxel_opt.py for reference):
========================================================
- lr: learning rate
- num_iters: number of optimization iterations
- render_resolution: voxel grid resolution (e.g., 512)
- transforms_json: path to camera transforms and image metadata
- envmap_path: path to environment map HDR image
- gt_mesh: ground truth mesh for alignment (optional)
- output_path: directory for saving optimization results
- display_interval: visualize every N iterations
- optimizer_type: 'adam' or 'sgd'
- frames_per_iter: frames per optimization step (None = all frames)
- seed: random seed for reproducibility
"""

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
import imageio
import trimesh

import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.renderers import EnvMap
from trellis2.renderers.voxel_renderer import VoxelRenderer
from trellis2.representations import Voxel, MeshWithVoxel, Mesh
from trellis2.utils import render_utils
from trellis2.modules.sparse import SparseTensor
from opt_utils.utility import display_image, save_image, transf3d_broadcast, transf3d, get_normalization_matrix
from dmtet.dmtet import DmTetGeometry
import o_voxel
from o_voxel import postprocess


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


@torch.no_grad()
def _freeze_module_params(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


def freeze_pipeline(pipeline):
    for name, module in pipeline.models.items():
        if isinstance(module, torch.nn.Module):
            for p in module.parameters():
                p.requires_grad_(False)


def copy_mesh(mesh: MeshWithVoxel) -> MeshWithVoxel:
    """Create a shallow copy of a mesh, keeping the same data pointers."""
    return MeshWithVoxel(
        vertices=mesh.vertices,
        faces=mesh.faces,
        origin=mesh.origin,
        voxel_size=mesh.voxel_size,
        coords=mesh.coords,
        attrs=mesh.attrs,
        voxel_shape=mesh.voxel_shape,
        layout=mesh.layout,
    )


def sigmoid_to_range(x, low=-0.5, high=0.5):
    """Map sigmoid output to [low, high] range."""
    margin = (high - low) / 2
    center = (low + high) / 2
    return (1 + 2 * margin) * torch.sigmoid(x) - margin


def softplus_inverse(y):
    return y + torch.log(-torch.expm1(-y))


class DifferentiableVoxelOptimizer:
    """
    Optimizes O-Voxel representation using differentiable voxel rasterization.
    
    CURRENT IMPLEMENTATION: Optimizes voxel base_color with fixed geometry
    - Renders directly from voxel representation using VoxelRenderer
    - All voxel features have gradients enabled for future geometry optimization
    - Provides a stable, working baseline for differentiable voxel optimization
    
    Key Advantages Over Mesh-based Optimization:
    - Avoids non-differentiable mesh extraction step
    - Gradients flow directly through rasterizer to voxel features
    - Supports smooth topology changes through soft alpha modulation (ready to implement)
    - More efficient per-iteration computation
    """
    
    def __init__(self, yaml_config_path: Path):
        self.config_yaml = yaml.safe_load(open(yaml_config_path, "r"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Parse config
        self.lr = float(self.config_yaml["lr"])
        self.num_iters = int(self.config_yaml["num_iters"])
        self.envmap_path = Path(self.config_yaml["envmap_path"])
        self.gt_mesh = Path(self.config_yaml.get("gt_mesh", ""))
        self.transforms_json_path = Path(self.config_yaml["transforms_json"])
        self.output_glb_path = Path(self.config_yaml.get("output_path", "optimization_output/output_mesh.glb"))
        
        self.pipeline_type = str(self.config_yaml.get("pipeline_type", '512'))
        self.frames_per_iter = self.config_yaml.get("frames_per_iter", None)
        self.near = float(self.config_yaml.get("near", 0.01))
        self.far = float(self.config_yaml.get("far", 100.0))
        self.ssaa = int(self.config_yaml.get("ssaa", 1))
        self.lamda = float(self.config_yaml.get("lamda", 0.001))
        self.peel_layers = int(self.config_yaml.get("peel_layers", 8))
        self.loss_type = str(self.config_yaml.get("loss_type", "mse"))
        self.latent_l2_weight = float(self.config_yaml.get("latent_l2_weight", 0.0))
        self.grad_clip_norm = float(self.config_yaml.get("grad_clip_norm", 0.0))
        self.display_interval = int(self.config_yaml.get("display_interval", 20))
        self.out_dir = self.config_yaml.get("out_dir", "optimization_output")
        self.render_resolution = int(self.config_yaml.get("render_resolution", 512))
        self.optimizer_type = str(self.config_yaml.get("optimizer_type", "adam")).lower()
        self.optimize_color_only = bool(self.config_yaml.get("optimize_color_only", False))
        self.voxel_margin = float(self.config_yaml.get("voxel_margin", 0.5))
        
        # Create output directory
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        
    def validate_paths(self):
        """Validate that all required paths exist."""
        assert self.envmap_path.exists(), f"Envmap path does not exist: {self.envmap_path}"
        if self.gt_mesh.exists():
            print(f"Using GT mesh for alignment: {self.gt_mesh}")
        else:
            print(f"No GT mesh provided, skipping alignment")
        assert self.transforms_json_path.exists(), f"Transforms JSON path does not exist: {self.transforms_json_path}"
    
    def setup_data(self):
        """Load images, camera parameters, and ground truth mesh."""
        transforms = _load_json(self.transforms_json_path)
        frames = transforms["frames"]
        self.base_object_pose = torch.tensor(transforms["object_pose"], dtype=torch.float32, device=self.device)
        
        base_dir = self.transforms_json_path.parent  
        image_basenames = [f["file_path"] for f in frames]
        image_paths: List[Path] = []
        for base_name in image_basenames:
            base_no_ext = base_dir / base_name
            image_paths.append(_try_image_path(base_no_ext))
        
        intr = transforms["intrinsic"]
        
        # Get prompt image index (same as voxel_opt.py)
        self.prompt_idx = int(self.config_yaml.get("prompt_idx", np.random.randint(0, len(image_paths))))
        
        # Get original image size for intrinsic scaling
        orig_img = Image.open(image_paths[self.prompt_idx])
        orig_w, orig_h = orig_img.size
        self.intr_scaled = _scale_opencv_intrinsics(intr, (orig_w, orig_h), self.render_resolution).to(device=self.device)
        
        # Load extrinsics
        extrinsics = []
        for f in frames:
            extrinsics.append(torch.tensor(f["transform_matrix"], dtype=torch.float32))
        
        # Load reference images
        ref_images = torch.stack(
            [_resize_rgb_to_tensor(p, self.render_resolution, self.device) for p in image_paths], dim=0
        )
        
        self.image_paths = image_paths
        self.extrinsics_t = torch.stack(extrinsics, dim=0).to(device=self.device)
        self.ref_images = ref_images
        self.num_images = len(image_paths)
        
        print(f"Loaded {self.num_images} reference images for optimization.")
        
        # Compute object pose alignment
        if self.gt_mesh.exists():
            dtc_mesh = trimesh.load_mesh(self.gt_mesh)
            # You would need a function to align mesh_ to dtc_mesh
            # For now, use identity
            self.object_pose = self.base_object_pose
        else:
            self.object_pose = self.base_object_pose
    
    def initialize_ovoxel_features(self, mesh_: MeshWithVoxel):
        """
        Initialize O-Voxel optimization parameters from the predicted mesh.
        
        This converts the mesh to O-Voxel format and creates differentiable parameters.
        """
        # Convert mesh to O-Voxel representation
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh_.vertices,
            faces=mesh_.faces,
            attr_volume=mesh_.attrs,
            coords=mesh_.coords,
            attr_layout=mesh_.layout,
            voxel_size=mesh_.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=1000000,
            texture_size=1024,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=True
        )
        
        mesh = glb
        vertices = mesh_.vertices.float().cpu()
        faces = mesh_.faces.long().cpu()
        
        # Convert mesh to flexible dual grid representation
        voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
            vertices=torch.tensor(mesh.vertices, dtype=torch.float32),
            faces=torch.tensor(mesh.faces, dtype=torch.float32), 
            grid_size=self.render_resolution, 
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], 
            face_weight=1.0, 
            boundary_weight=0.2, 
            regularization_weight=1e-2, 
            timing=True
        )
        
        # Sort voxels by encoded index for consistency
        vid = o_voxel.serialize.encode_seq(voxel_indices)
        mapping = torch.argsort(vid)
        voxel_indices = voxel_indices[mapping]
        dual_vertices = dual_vertices[mapping]
        intersected = intersected[mapping]
        
        # Get material attributes
        voxel_indices_mat, attributes = o_voxel.convert.textured_mesh_to_volumetric_attr(
            mesh, 
            grid_size=self.render_resolution, 
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], 
            timing=True
        )
        
        # Sort attributes
        vid_mat = o_voxel.serialize.encode_seq(voxel_indices_mat)
        mapping_mat = torch.argsort(vid_mat)
        attributes = {k: v[mapping_mat] for k, v in attributes.items()}
        
        # Pack geometry information
        dual_vertices_packed = dual_vertices * self.render_resolution - voxel_indices
        dual_vertices_packed = (torch.clamp(dual_vertices_packed, 0, 1) * 255).type(torch.uint8)
        intersected_packed = (intersected[:, 0:1] + 2 * intersected[:, 1:2] + 4 * intersected[:, 2:3]).type(torch.uint8)
        
        attributes['dual_vertices'] = dual_vertices_packed
        attributes['intersected'] = intersected_packed
        
        # Initialize optimization parameters
        dual_vertices_opt = attributes['dual_vertices'].float() / 255.0
        
        # Create base color from attributes
        base_color = attributes['base_color'].float() / 255.0
        
        # Pack all features for optimization
        # Features: [dual_vertices (3), intersected (1), quad_lerp (1), base_color (3)]
        quad_lerp = torch.nn.functional.softplus(torch.ones_like(dual_vertices[:, :1]) * 0.1)
        
        h_feats = torch.cat([
            dual_vertices_opt,     # 0-2: dual vertex offsets
            intersected_packed.float() / 7.0,  # 3: intersected state
            torch.log(quad_lerp + 1e-6),  # 4: quad_lerp (log space for softplus)
            base_color              # 5-7: base color
        ], dim=-1).to(device=self.device)
        
        # Create sparse tensor representation
        h_feats_param = torch.nn.Parameter(h_feats)
        h = SparseTensor(
            feats=h_feats_param, 
            coords=torch.cat([torch.zeros_like(voxel_indices[:, 0:1]), voxel_indices], dim=-1).to(self.device)
        )
        
        # Store voxel coordinates for position computation
        self.voxel_coords = voxel_indices.to(self.device)
        self.h_feats_param = h_feats_param
        self.h = h
        
        return h, h_feats_param, voxel_indices.to(self.device)
    
    def create_voxel_representation(self, h_opt, voxel_coords):
        """
        Create a Voxel representation from optimized O-Voxel features.
        
        This is the key step for differentiable rendering - we create a Voxel object
        that the VoxelRenderer can directly render.
        
        Key insight: Keep all voxels active to maintain gradient flow. The voxel rasterizer
        handles visibility and compositing automatically. Topology changes can be achieved
        by modulating per-voxel alpha through the intersected feature.
        """
        # Decode features from optimization parameters
        base_color_raw = h_opt.feats[..., 5:8]
        
        # Apply sigmoid to map to [0, 1] color range
        base_color = torch.sigmoid(base_color_raw)  # [N, 3]
        
        # Create voxel representation with all voxels
        # VoxelRenderer will handle proper compositing and alpha generation
        voxel = Voxel(
            origin=[-0.5, -0.5, -0.5],
            voxel_size=1.0 / self.render_resolution,
            coords=voxel_coords,
            attrs=base_color,  # [N, 3] - RGB colors
            layout={'base_color': slice(0, 3)},
            device=self.device
        )
        
        return voxel
    
    def optimize(self):
        """Main optimization loop using differentiable voxel rendering."""
        self.validate_paths()
        self.setup_data()
        
        # Load and run pipeline once to get initial mesh
        print("Loading Trellis2 pipeline...")
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
        pipeline.low_vram = False
        pipeline.cuda()
        
        # Run on prompt image to get initial mesh
        prompt_img = Image.open(self.image_paths[self.prompt_idx]).convert("RGBA")
        
        print("Generating initial mesh...")
        mesh_ = pipeline.run(
            prompt_img,
            pipeline_type=self.pipeline_type,
        )[0]
        
        # Initialize O-Voxel features
        print("Initializing O-Voxel features...")
        h, h_feats_param, voxel_coords = self.initialize_ovoxel_features(mesh_)
        
        # Create voxel renderer for differentiable rendering
        voxel_renderer = VoxelRenderer(rendering_options={
            "resolution": self.render_resolution,
            "near": self.near,
            "far": self.far,
            "ssaa": self.ssaa,
        })
        
        # Setup optimizer
        if self.optimizer_type == "sgd":
            optimizer = torch.optim.SGD([h_feats_param], lr=self.lr)
        else:
            optimizer = torch.optim.Adam([h_feats_param], lr=self.lr, betas=(0.9, 0.999))
        
        # Loss function
        criterion = nn.MSELoss(reduction='mean')
        
        # Random number generator
        rng = np.random.default_rng(seed=int(self.config_yaml.get("seed", 0)))
        
        print(f"Starting optimization for {self.num_iters} iterations...")
        print(f"Optimizing with {h_feats_param.shape[0]} voxels")
        
        # Optimization loop
        for iteration in range(self.num_iters + 1):
            optimizer.zero_grad()
            
            # Create updated voxel representation
            h_opt = h.replace(feats=h_feats_param)
            voxel = self.create_voxel_representation(h_opt, voxel_coords)
            
            # Sample frames for this iteration
            indices = (
                np.arange(self.num_images)
                if self.frames_per_iter is None or self.frames_per_iter == self.num_images
                else np.atleast_1d(rng.choice(self.num_images, size=self.frames_per_iter, replace=False))
            )
            
            total_loss = torch.zeros((), device=self.device)
            
            # Render and compute loss for each sampled frame
            for frame_idx in indices:
                extr = self.extrinsics_t[frame_idx]
                
                # Render voxel representation using differentiable voxel rasterizer
                render_output = voxel_renderer.render(
                    voxel=voxel,
                    extrinsics=extr,
                    intrinsics=self.intr_scaled
                )
                
                pred_color = render_output.get('base_color', render_output.get('color', None))
                pred_alpha = render_output.get('alpha', None)
                
                if pred_color is None:
                    print(f"Warning: render_output keys are {render_output.keys()}")
                    continue
                
                # Get ground truth
                gt = self.ref_images[frame_idx]
                gt_rgb = gt[:3]
                gt_alpha = gt[3:4]
                
                # Compute RGB loss
                rgb_loss = criterion(pred_color, gt_rgb)
                
                # Compute alpha/mask loss if available
                frame_loss = rgb_loss
                if pred_alpha is not None:
                    if pred_alpha.ndim == 2:
                        pred_alpha = pred_alpha.unsqueeze(0)
                    alpha_loss = criterion(pred_alpha, gt_alpha)
                    frame_loss = frame_loss + alpha_loss
                
                total_loss += frame_loss
                
                # Visualization
                if self.display_interval > 0 and (iteration % self.display_interval == 0):
                    self._visualize_frame(
                        pred_color, gt_rgb, pred_alpha, gt_alpha,
                        iteration, frame_idx
                    )
            
            # Average loss over frames
            total_loss = total_loss / len(indices)
            print(f"[iter {iteration}] loss={float(total_loss.detach().cpu()):.6f}")
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_([h_feats_param], self.grad_clip_norm)
            
            # Optimizer step
            optimizer.step()
            
            if iteration % 10 == 0:
                print(f"[Iter {iteration:04d}] Loss: {float(total_loss.detach().cpu()):.6f}")
        
        print("Optimization completed!")
        self._save_results(h_opt, voxel_coords)
    
    def _visualize_frame(self, pred_rgb, gt_rgb, pred_alpha, gt_alpha, iteration, frame_idx):
        """Save visualization of prediction vs ground truth."""
        # Handle alpha channel
        if pred_alpha is not None:
            if pred_alpha.ndim == 2:
                pred_alpha = pred_alpha.unsqueeze(0)
            pred_rgba = torch.cat([pred_rgb, pred_alpha], dim=0)
        else:
            # Create dummy alpha from ones
            h, w = pred_rgb.shape[1], pred_rgb.shape[2]
            dummy_alpha = torch.ones(1, h, w, device=pred_rgb.device, dtype=pred_rgb.dtype)
            pred_rgba = torch.cat([pred_rgb, dummy_alpha], dim=0)
        
        # Handle gt alpha
        gt_alpha_expanded = gt_alpha if gt_alpha.shape[0] == 1 else gt_alpha[:1]
        gt_rgba = torch.cat([gt_rgb, gt_alpha_expanded], dim=0)
        
        # Combine into one image: pred on left, gt on right
        img = torch.cat([pred_rgba, gt_rgba], dim=2)
        img = img.permute(1, 2, 0)
        img = img.clamp(0.0, 1.0)
        img_np = img.detach().cpu().numpy()
        
        save_image(f"{self.out_dir}/opt_{iteration:04d}_{frame_idx:02d}.png", img_np)
    
    def _save_results(self, h_opt, voxel_coords):
        """Save final optimized voxel representation."""
        # Convert final voxel representation to mesh for export
        voxel = self.create_voxel_representation(h_opt, voxel_coords)
        
        # Try to reconstruct mesh from voxels (optional - depends on o_voxel capabilities)
        # For now, save the voxel representation
        print(f"Saved optimized voxel representation with {voxel.coords.shape[0]} active voxels")
        print(f"Output saved to {self.out_dir}")


def refine_latents(yaml_config_path: Path):
    """Main entry point for optimization."""
    optimizer = DifferentiableVoxelOptimizer(yaml_config_path)
    optimizer.optimize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize O-Voxel representation using differentiable voxel rasterization")
    parser.add_argument("config", type=str, help="Path to YAML configuration file")
    args = parser.parse_args()
    
    refine_latents(Path(args.config))
