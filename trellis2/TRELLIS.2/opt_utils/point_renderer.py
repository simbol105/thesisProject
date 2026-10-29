import torch
import nvdiffrast.torch as dr
from easydict import EasyDict as edict
from typing import Dict, Tuple

def intrinsics_to_projection(
        intrinsics: torch.Tensor,
        near: float,
        far: float,
    ) -> torch.Tensor:
    """OpenCV intrinsics to OpenGL perspective matrix"""
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = - 2 * cy + 1
    ret[2, 2] = (far + near) / (far - near)
    ret[2, 3] = 2 * near * far / (near - far)
    ret[3, 2] = 1.
    return ret
    

def ovoxel_to_pointcloud(vertices, coords, attrs, resolution=512):  
    """Convert O-Voxel to point cloud format."""
    world_positions = (coords + vertices) / resolution - 0.5  
    return world_positions, attrs


class PointRenderer:
    """Differentiable point renderer with Trellis2-compatible API."""

    def __init__(self, rendering_options={}, device="cuda"):
        self.rendering_options = edict({
            "resolution": None,
            "near": 0.1,             
            "far": 100.0,
            "ssaa": 1,
            "chunk_size": None,
            "antialias": True,
            "clamp_barycentric_coords": False,
            "point_radius": 0.01,     
        })
        self.rendering_options.update(rendering_options)
        self.device = device
        self.glctx = dr.RasterizeGLContext(device=device)
    
    
    def render(
        self,
        world_positions: torch.Tensor,   # (N,3)
        attrs: torch.Tensor,             # (N,C) or None
        extrinsics: torch.Tensor,        # (4,4)
        intrinsics: torch.Tensor,        # (3,3)
        return_types=["mask", "normal", "depth", "attr"],
        transformation=None
    ) -> edict:

        opt = self.rendering_options
        if opt.resolution is None:
            raise ValueError("rendering_options['resolution'] must be set")
        H = W = opt.resolution
        near = opt.near if opt.near is not None else 0.1
        far = opt.far if opt.far is not None else 100.0

        # 1. Build Projection Matrix
        proj = intrinsics_to_projection(intrinsics.to(self.device), near, far)

        # 2. Transform points to Clip Space
        N = world_positions.shape[0]
        pts = world_positions.to(self.device)
        ones = torch.ones((N, 1), device=self.device)
        pts_h = torch.cat([pts, ones], dim=-1)
        clip_centers = pts_h @ extrinsics.T @ proj.T  # (N,4)

        # 3. MEMORY FIX: Use Broadcasting instead of repeat_interleave
        w_coords = clip_centers[:, 3:4] # (N, 1)
        r = opt.point_radius
        
        # Base localized offsets for a point square
        base_offsets = torch.tensor([
            [-r, -r], [r, -r], [-r, r], [r, r]
        ], device=self.device, dtype=torch.float32) # (4, 2)

        # Broadcast offset shifts natively across the N dimension -> Shape: (N, 4, 2)
        offsets = base_offsets.unsqueeze(0) * w_coords.unsqueeze(-1)
        
        # Expand clip centers to (N, 4, 4) using views to keep memory footprint flat
        v_clip = clip_centers.unsqueeze(1).repeat(1, 4, 1)
        v_clip[:, :, :2] += offsets
        v_clip = v_clip.view(-1, 4) # Reshape to final sequential array: (4*N, 4)

        # 4. MEMORY FIX: Generate faces with low-overhead integer sequence
        base_idx = torch.arange(0, N, device=self.device, dtype=torch.int32).unsqueeze(-1) * 4
        quad_offsets = torch.tensor([0, 1, 2, 1, 3, 2], device=self.device, dtype=torch.int32).unsqueeze(0)
        faces = (base_idx + quad_offsets).view(-1, 3) # Shape: (2*N, 3)

        # 5. Local UV configurations to calculate round point shapes
        base_uvs = torch.tensor([
            [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]
        ], device=self.device, dtype=torch.float32)
        local_uvs = base_uvs.unsqueeze(0).repeat(N, 1, 1).view(1, -1, 2) # (1, 4*N, 2)

        # 6. Rasterize Triangles
        rast, _ = dr.rasterize(
            self.glctx,
            v_clip.unsqueeze(0),
            faces,
            resolution=[H, W],
        )

        out = edict()
        rast_mask = (rast[0, :, :, 3] > 0).float()
        
        # 7. Compute Soft Differentiable Alpha Profile (Fixes Gradients)
        uv_img = dr.interpolate(local_uvs, rast, faces)[0].squeeze(0)  # (H, W, 2)
        dist_from_center = torch.sum(uv_img ** 2, dim=-1)  # (H, W)
        
        # Smoothly fades visibility down to 0 at point boundaries
        soft_alpha = torch.clamp(1.0 - dist_from_center, min=0.0, max=1.0)
        final_mask = rast_mask * soft_alpha

        if "mask" in return_types:
            out.mask = final_mask

     
        # 8. MEMORY FIX: Process attributes cleanly using indexed views
        if "attr" in return_types and attrs is not None:
            # 1. Expand point features to match the expanded quad vertices
            v_attrs = attrs.to(self.device).unsqueeze(1).repeat(1, 4, 1).view(1, -1, attrs.shape[-1])
            attr_img = dr.interpolate(v_attrs, rast, faces)[0] # (1, H, W, C)
            
            if opt.antialias:
                # FIX: Extract only the first 3 channels
                proxy_channels = min(3, attr_img.shape[-1])
                proxy_img = attr_img[..., :proxy_channels]
                
                # Calculate antialiasing on the proxy tensor
                aa_proxy = dr.antialias(proxy_img, rast, v_clip.unsqueeze(0), faces)
                
                # Extract the edge blend factor: where aa_proxy deviates from proxy_img
                eps = 1e-5
                edge_filter = (aa_proxy + eps) / (proxy_img + eps)
                edge_filter = torch.clamp(edge_filter[..., :1], min=0.0, max=1.0) # Isolate a single spatial multiplier mask
                
                # Apply the calculated edge transparency cleanly to your high-dimensional attributes
                attr_img = attr_img * edge_filter
            
            attr_img_3d = attr_img.squeeze(0)
            attr_img_3d = attr_img_3d * final_mask.unsqueeze(-1)
            out.attr = attr_img_3d.permute(2, 0, 1)  # [C, H, W]


        # 9. Depth Layout Setup
        if "depth" in return_types or "normal" in return_types:
            clip_depth = rast[0, :, :, 2]
            linear_depth = (2.0 * near * far) / (far + near - clip_depth * (far - near))
            linear_depth = linear_depth * final_mask
            
            if "depth" in return_types:
                out.depth = linear_depth

        # 10. Normals Layout Setup
        if "normal" in return_types:
            dx = linear_depth[:, 1:] - linear_depth[:, :-1]
            dy = linear_depth[1:, :] - linear_depth[:-1, :]

            dx = dx.unsqueeze(0).unsqueeze(0)
            dx = torch.nn.functional.pad(dx, (0, 1, 0, 0), mode='replicate').squeeze(0).squeeze(0)
            dy = dy.unsqueeze(0).unsqueeze(0)
            dy = torch.nn.functional.pad(dy, (0, 0, 1, 0), mode='replicate').squeeze(0).squeeze(0)

            normal = torch.stack([-dx, -dy, torch.ones_like(linear_depth)], dim=0)
            normal = torch.nn.functional.normalize(normal, dim=0)
            
            normal = torch.where(final_mask.unsqueeze(0) > 0, normal, torch.tensor([[[0.0]], [[0.0]], [[1.0]]], device=self.device))
            out.normal = normal

        return out
    

# vertices: (N, 3) dual vertex positions within voxels
      
      