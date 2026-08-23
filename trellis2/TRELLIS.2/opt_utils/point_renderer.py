

import torch
import nvdiffrast.torch as dr
from easydict import EasyDict as edict
from typing import Dict, Tuple

def intrinsics_to_projection(
        intrinsics: torch.Tensor,
        near: float,
        far: float,
    ) -> torch.Tensor:
    """
    OpenCV intrinsics to OpenGL perspective matrix
    """
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
    """  
    Convert O-Voxel to point cloud format.  
    """  
    world_positions = (coords + vertices) / resolution - 0.5  
    return world_positions, attrs


class PointRenderer:
    """
    Differentiable point renderer with Trellis2-compatible API.
    Uses nvdiffrast triangle rasterization by expanding points into quads.
    """

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
        # Replaces the massive .repeat(N) loop with a sequential tensor broadcast layout
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

     
        # --------------------------------------------------------------
        # 8. MEMORY FIX: Process attributes cleanly using indexed views
        # --------------------------------------------------------------
        if "attr" in return_types and attrs is not None:
            # 1. Expand point features to match the expanded quad vertices
            v_attrs = attrs.to(self.device).unsqueeze(1).repeat(1, 4, 1).view(1, -1, attrs.shape[-1])
            attr_img = dr.interpolate(v_attrs, rast, faces)[0] # (1, H, W, C)
            
            if opt.antialias:
                # FIX: Extract only the first 3 channels (or a single channel mask) 
                # to safely pass to the antialiasing function without exceeding the 8-channel limit
                proxy_channels = min(3, attr_img.shape[-1])
                proxy_img = attr_img[..., :proxy_channels]
                
                # Calculate antialiasing on the proxy tensor
                aa_proxy = dr.antialias(proxy_img, rast, v_clip.unsqueeze(0), faces)
                
                # Extract the edge blend factor: where aa_proxy deviates from proxy_img
                # Avoid dividing by zero with a small epsilon
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
    





    # def render(
    #     self,
    #     world_positions: torch.Tensor,   # (N,3)
    #     attrs: torch.Tensor,             # (N,C) or None
    #     extrinsics: torch.Tensor,        # (4,4)
    #     intrinsics: torch.Tensor,        # (3,3)
    #     return_types=["mask", "normal", "depth", "attr"],
    #     transformation=None
    # ) -> edict:

    #     opt = self.rendering_options
    #     if opt.resolution is None:
    #         raise ValueError("rendering_options['resolution'] must be set")
    #     H = W = opt.resolution
    #     near = opt.near if opt.near is not None else 0.1
    #     far = opt.far if opt.far is not None else 100.0

    #     # 1. Build Projection Matrix
    #     proj = intrinsics_to_projection(intrinsics.to(self.device), near, far)

    #     # 2. Transform points to Clip Space
    #     N = world_positions.shape[0]
    #     pts = world_positions.to(self.device)
    #     ones = torch.ones((N, 1), device=self.device)
    #     pts_h = torch.cat([pts, ones], dim=-1)
    #     clip_centers = pts_h @ extrinsics.T @ proj.T  # (N,4)

    #     # 3. MEMORY FIX: Use Broadcasting instead of repeat_interleave
    #     w_coords = clip_centers[:, 3:4] # (N, 1)
    #     r = opt.point_radius
        
    #     # Base localized offsets for a point square
    #     base_offsets = torch.tensor([
    #         [-r, -r], [r, -r], [-r, r], [r, r]
    #     ], device=self.device, dtype=torch.float32) # (4, 2)

    #     # Broadcast offset shifts natively across the N dimension -> Shape: (N, 4, 2)
    #     offsets = base_offsets.unsqueeze(0) * w_coords.unsqueeze(-1)
        
    #     # Expand clip centers to (N, 4, 4) using views to keep memory footprint flat
    #     v_clip = clip_centers.unsqueeze(1).repeat(1, 4, 1)
    #     v_clip[:, :, :2] += offsets
    #     v_clip = v_clip.view(-1, 4) # Reshape to final sequential array: (4*N, 4)

    #     # 4. MEMORY FIX: Generate faces with low-overhead integer sequence
    #     base_idx = torch.arange(0, N, device=self.device, dtype=torch.int32).unsqueeze(-1) * 4
    #     quad_offsets = torch.tensor([0, 1, 2, 1, 3, 2], device=self.device, dtype=torch.int32).unsqueeze(0)
    #     faces = (base_idx + quad_offsets).view(-1, 3) # Shape: (2*N, 3)

    #     # 5. Local UV configurations to calculate round point shapes
    #     base_uvs = torch.tensor([
    #         [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]
    #     ], device=self.device, dtype=torch.float32)
    #     local_uvs = base_uvs.unsqueeze(0).repeat(N, 1, 1).view(1, -1, 2) # (1, 4*N, 2)

    #     # 6. Rasterize Triangles
    #     rast, _ = dr.rasterize(
    #         self.glctx,
    #         v_clip.unsqueeze(0),
    #         faces,
    #         resolution=[H, W],
    #     )

    #     out = edict()
    #     rast_mask = (rast[0, :, :, 3] > 0).float()
        
    #     # 7. Compute Soft Differentiable Alpha Profile (Fixes Gradients)
    #     uv_img = dr.interpolate(local_uvs, rast, faces)[0].squeeze(0)  # (H, W, 2)
    #     dist_from_center = torch.sum(uv_img ** 2, dim=-1)  # (H, W)
        
    #     # Smoothly fades visibility down to 0 at point boundaries
    #     soft_alpha = torch.clamp(1.0 - dist_from_center, min=0.0, max=1.0)
    #     final_mask = rast_mask * soft_alpha

    #     if "mask" in return_types:
    #         out.mask = final_mask

    #     # --------------------------------------------------------------
    #     # 8. MEMORY FIX & ATTRIBUTE REFINEMENT
    #     # --------------------------------------------------------------
    #     if "attr" in return_types and attrs is not None:
    #         # Expand point features to match the expanded quad vertices
    #         v_attrs = attrs.to(self.device).unsqueeze(1).repeat(1, 4, 1).view(1, -1, attrs.shape[-1])
    #         attr_img = dr.interpolate(v_attrs, rast, faces)[0] # (1, H, W, C)
            
    #         if opt.antialias:
    #             # Use standard, low-overhead blending over the low-dimensional proxy 
    #             # channels instead of dividing tensors
    #             proxy_channels = min(3, attr_img.shape[-1])
    #             proxy_img = attr_img[..., :proxy_channels]
                
    #             # Apply the native dr.antialias call
    #             aa_proxy = dr.antialias(proxy_img, rast, v_clip.unsqueeze(0), faces)
                
    #             # Extract clean silhouette multipliers natively from the alpha channel
    #             # instead of calculating pixel divisions
    #             attr_img = attr_img * final_mask.unsqueeze(0).unsqueeze(-1)
    #         else:
    #             attr_img = attr_img * final_mask.unsqueeze(0).unsqueeze(-1)
            
    #         attr_img_3d = attr_img.squeeze(0)
            
    #         # FIX: Explicitly clamp the background to pure black (0.0) 
    #         # to overwrite any persistent checkerboard noise frames
    #         attr_img_3d = torch.where(final_mask.unsqueeze(-1) > 0, attr_img_3d, torch.zeros_like(attr_img_3d))
    #         out.attr = attr_img_3d.permute(2, 0, 1)  # [C, H, W]

    #     # 9. Depth Layout Setup
    #     if "depth" in return_types or "normal" in return_types:
    #         clip_depth = rast[0, :, :, 2]
    #         linear_depth = (2.0 * near * far) / (far + near - clip_depth * (far - near))
    #         linear_depth = linear_depth * final_mask
            
    #         if "depth" in return_types:
    #             out.depth = linear_depth

    #     # 10. Normals Layout Setup
    #     if "normal" in return_types:
    #         dx = linear_depth[:, 1:] - linear_depth[:, :-1]
    #         dy = linear_depth[1:, :] - linear_depth[:-1, :]

    #         dx = dx.unsqueeze(0).unsqueeze(0)
    #         dx = torch.nn.functional.pad(dx, (0, 1, 0, 0), mode='replicate').squeeze(0).squeeze(0)
    #         dy = dy.unsqueeze(0).unsqueeze(0)
    #         dy = torch.nn.functional.pad(dy, (0, 0, 1, 0), mode='replicate').squeeze(0).squeeze(0)

    #         normal = torch.stack([-dx, -dy, torch.ones_like(linear_depth)], dim=0)
    #         normal = torch.nn.functional.normalize(normal, dim=0)
            
    #         # FIX: Clear background space to a pure, flat, solid sky-blue layout
    #         bg_normal = torch.tensor([0.0, 0.0, 1.0], device=self.device).view(3, 1, 1)
    #         normal = torch.where(final_mask.unsqueeze(0) > 0, normal, bg_normal)
    #         out.normal = normal

    #     return out



    # def render(
    #     self,
    #     world_positions: torch.Tensor,   # (N,3)
    #     attrs: torch.Tensor,             # (N,C) or None
    #     extrinsics: torch.Tensor,        # (4,4)
    #     intrinsics: torch.Tensor,        # (3,3)
    #     return_types=["mask", "normal", "depth", "attr"],
    #     transformation=None
    # ) -> edict:

    #     opt = self.rendering_options
    #     if opt.resolution is None:
    #         raise ValueError("rendering_options['resolution'] must be set")
    #     H = W = opt.resolution
    #     near = opt.near if opt.near is not None else 0.1
    #     far = opt.far if opt.far is not None else 100.0

    #     # 1. FIX: Use your accurate intrinsics_to_projection function
    #     proj = intrinsics_to_projection(intrinsics.to(self.device), near, far)

    #     # 2. Transform points to Clip Space
    #     N = world_positions.shape[0]
    #     pts = world_positions.to(self.device)
    #     ones = torch.ones((N, 1), device=self.device)
    #     pts_h = torch.cat([pts, ones], dim=-1)
        
    #     # Multiply matching standard layout conventions
    #     clip_centers = pts_h @ extrinsics.T @ proj.T  # (N,4)

    #     # 3. Generate Expanded Triangle Quad Vertices with local UVs
    #     w_coords = clip_centers[:, 3:4]
    #     r = opt.point_radius
        
    #     # Local 2D quad corner offsets scaled by W for perspective consistency
    #     offsets = torch.tensor([
    #         [-r, -r], [r, -r], [-r, r], [r, r]
    #     ], device=self.device, dtype=torch.float32).repeat(N, 1) * w_coords.repeat_interleave(4, dim=0)

    #     v_clip = clip_centers.repeat_interleave(4, dim=0)
    #     v_clip[:, :2] += offsets

    #     # UV coordinates to define circular shapes
    #     local_uvs = torch.tensor([
    #         [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]
    #     ], device=self.device, dtype=torch.float32).repeat(N, 1).unsqueeze(0)

    #     # Build stable face matrices matching sequential index layout
    #     base_idx = torch.arange(0, N, device=self.device).repeat_interleave(6) * 4
    #     quad_offsets = torch.tensor([0, 1, 2, 1, 3, 2], device=self.device, dtype=torch.int32).repeat(N)
    #     faces = (base_idx + quad_offsets).view(-1, 3).int()

    #     # 4. Standard Hardware Rasterization
    #     rast, _ = dr.rasterize(
    #         self.glctx,
    #         v_clip.unsqueeze(0),
    #         faces,
    #         resolution=[H, W],
    #     )

    #     out = edict()
        
    #     # 5. Extract UV attributes to determine point circular masking
    #     uv_img = dr.interpolate(local_uvs, rast, faces)[0].squeeze(0)  # (H, W, 2)
    #     dist_from_center = torch.sum(uv_img ** 2, dim=-1)  # (H, W)
        
    #     # Binary mask: 1.0 if inside circle radius, 0.0 if outside corner edge
    #     circle_mask = (dist_from_center <= 1.0).float()
    #     final_mask = (rast[0, :, :, 3] > 0).float() * circle_mask

    #     # 6. Mask Output Configuration
    #     if "mask" in return_types:
    #         out.mask = final_mask

    #     # 7. Attribute Interpolation with Clean Masking
    #     if "attr" in return_types and attrs is not None:
    #         v_attrs = attrs.to(self.device).repeat_interleave(4, dim=0).unsqueeze(0)
    #         attr_img = dr.interpolate(v_attrs, rast, faces)[0] # Shape: (1, H, W, C)
            
    #         # Apply antialiasing wrapper cleanly if selected options ask for it
    #         if opt.antialias:
    #             attr_img = dr.antialias(attr_img, rast, v_clip.unsqueeze(0), faces)

    #         attr_img_3d = attr_img.squeeze(0)
            
    #         # Mask out background canvas pixels and square quad edges
    #         # Set background to black (0.0) instead of undefined mud colors
    #         attr_img_3d = attr_img_3d * final_mask.unsqueeze(-1)
    #         out.attr = attr_img_3d.permute(2, 0, 1)  # [C, H, W]

    #     # 8. Metrics Setup (Depth and Normals)
    #     if "depth" in return_types or "normal" in return_types:
    #         clip_depth = rast[0, :, :, 2]
            
    #         # Unproject depth out of exponential clip-space back to real world linear metrics
    #         linear_depth = (2.0 * near * far) / (far + near - clip_depth * (far - near))
    #         linear_depth = linear_depth * final_mask
            
    #         if opt.antialias:
    #             linear_depth_4d = linear_depth.unsqueeze(0).unsqueeze(-1)
    #             linear_depth_4d = dr.antialias(linear_depth_4d, rast, v_clip.unsqueeze(0), faces)
    #             linear_depth = linear_depth_4d.squeeze(0).squeeze(-1)
            
    #         if "depth" in return_types:
    #             out.depth = linear_depth

    #     if "normal" in return_types:
    #         dx = linear_depth[:, 1:] - linear_depth[:, :-1]
    #         dy = linear_depth[1:, :] - linear_depth[:-1, :]

    #         dx = dx.unsqueeze(0).unsqueeze(0)
    #         dx = torch.nn.functional.pad(dx, (0, 1, 0, 0), mode='replicate').squeeze(0).squeeze(0)
    #         dy = dy.unsqueeze(0).unsqueeze(0)
    #         dy = torch.nn.functional.pad(dy, (0, 0, 1, 0), mode='replicate').squeeze(0).squeeze(0)

    #         normal = torch.stack([-dx, -dy, torch.ones_like(linear_depth)], dim=0)
    #         normal = torch.nn.functional.normalize(normal, dim=0)
            
    #         # Revert background elements to a clean, flat blue layout
    #         normal = torch.where(final_mask.unsqueeze(0) > 0, normal, torch.tensor([[[0.0]], [[0.0]], [[1.0]]], device=self.device))
    #         out.normal = normal

    #     return out

# import torch
# import nvdiffrast.torch as dr
# from easydict import EasyDict as edict







# def intrinsics_to_projection(
#         intrinsics: torch.Tensor,
#         near: float,
#         far: float,
#     ) -> torch.Tensor:
#     """
#     OpenCV intrinsics to OpenGL perspective matrix

#     Args:
#         intrinsics (torch.Tensor): [3, 3] OpenCV intrinsics matrix
#         near (float): near plane to clip
#         far (float): far plane to clip
#     Returns:
#         (torch.Tensor): [4, 4] OpenGL perspective matrix
#     """
#     fx, fy = intrinsics[0, 0], intrinsics[1, 1]
#     cx, cy = intrinsics[0, 2], intrinsics[1, 2]
#     ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
#     ret[0, 0] = 2 * fx
#     ret[1, 1] = 2 * fy
#     ret[0, 2] = 2 * cx - 1
#     ret[1, 2] = - 2 * cy + 1
#     ret[2, 2] = (far + near) / (far - near)
#     ret[2, 3] = 2 * near * far / (near - far)
#     ret[3, 2] = 1.
#     return ret
    


# def ovoxel_to_pointcloud(vertices, coords, attrs, resolution=512):  
#     """  
#     Convert O-Voxel to point cloud format.  
      
#     Args:  
#         vertices: (N, 3) dual vertex positions within voxels  
#         coords: (N, 3) voxel grid coordinates  
#         attrs: (N, C) voxel attributes  
#         resolution: int, grid resolution  
      
#     Returns:  
#         positions: (N, 3) world-space positions  
#         features: (N, C) features  
#     """  
#     # Convert voxel-local dual vertices to world space  
#     world_positions = (coords + vertices) / resolution - 0.5  
      
#     return world_positions, attrs






# class PointRenderer:
#     """
#     Differentiable point renderer with Trellis2-compatible API.
#     Uses nvdiffrast triangle rasterization by expanding points into quads.
#     """

#     def __init__(self, rendering_options={}, device="cuda"):
#         # Default Trellis2 render options
#         self.rendering_options = edict({
#             "resolution": None,
#             "near": None,
#             "far": None,
#             "ssaa": 1,
#             "chunk_size": None,
#             "antialias": True,
#             "clamp_barycentric_coords": False,
#             "point_radius": 0.01,     # Quad half-width in screen/world proportions
#         })
#         self.rendering_options.update(rendering_options)

#         self.device = device
#         # Note: dr.RasterizeGLContext can also be swapped with dr.RasterizeCudaContext
#         self.glctx = dr.RasterizeGLContext(device=device)

   
#     def render(
#         self,
#         world_positions: torch.Tensor,   # (N,3)
#         attrs: torch.Tensor,             # (N,C) or None
#         extrinsics: torch.Tensor,        # (4,4)
#         intrinsics: torch.Tensor,        # (3,3)
#         return_types=["mask", "normal", "depth", "attr"],
#         transformation=None
#     ) -> edict:

#         opt = self.rendering_options
#         if opt.resolution is None:
#             raise ValueError("rendering_options['resolution'] must be set")
#         H = W = opt.resolution

#         # 1. Build Projection Matrix
#         fx, fy = intrinsics[0, 0], intrinsics[1, 1]
#         cx, cy = intrinsics[0, 2], intrinsics[1, 2]
#         proj = torch.tensor([
#             [2 * fx / W,       0,          0, 0],
#             [0,           2 * fy / H,      0, 0],
#             [0,                0,         -1, -1],
#             [0,                0,         -1,  0],
#         ], device=self.device, dtype=torch.float32)

#         # 2. Transform points to Clip Space
#         N = world_positions.shape[0]
#         pts = world_positions.to(self.device)
#         ones = torch.ones((N, 1), device=self.device)
#         pts_h = torch.cat([pts, ones], dim=-1)
#         clip_centers = pts_h @ extrinsics.T @ proj.T  # (N,4)

#         # 3. Generate Expanded Triangle Quad Vertices with local UVs
#         w_coords = clip_centers[:, 3:4]
#         r = opt.point_radius
        
#         # Local 2D quad corner offsets
#         offsets = torch.tensor([
#             [-r, -r], [r, -r], [-r, r], [r, r]
#         ], device=self.device, dtype=torch.float32).repeat(N, 1) * w_coords.repeat_interleave(4, dim=0)

#         v_clip = clip_centers.repeat_interleave(4, dim=0)
#         v_clip[:, :2] += offsets

#         # UV coordinates to define point circular boundaries
#         local_uvs = torch.tensor([
#             [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]
#         ], device=self.device, dtype=torch.float32).repeat(N, 1).unsqueeze(0)

#         # Build stable face matrices matching sequential index layout
#         base_idx = torch.arange(0, N, device=self.device).repeat_interleave(6) * 4
#         quad_offsets = torch.tensor([0, 1, 2, 1, 3, 2], device=self.device, dtype=torch.int32).repeat(N)
#         faces = (base_idx + quad_offsets).view(-1, 3).int()

#         # 4. Standard Hardware Rasterization (Enables native Z-buffer sorting)
#         rast, _ = dr.rasterize(
#             self.glctx,
#             v_clip.unsqueeze(0),
#             faces,
#             resolution=[H, W],
#         )

#         out = edict()
        
#         # 5. Extract UV attributes to determine point circular masking
#         uv_img, _ = dr.interpolate(local_uvs, rast, faces)
#         uv_img = uv_img.squeeze(0)  # (H, W, 2)
#         dist_from_center = torch.sum(uv_img ** 2, dim=-1)  # (H, W)
        
#         # Binary mask: 1.0 if inside circle radius, 0.0 if outside corner edge
#         circle_mask = (dist_from_center <= 1.0).float()
#         final_mask = (rast[0, :, :, 3] > 0).float() * circle_mask

#         # 6. Mask Output Configuration
#         if "mask" in return_types:
#             out.mask = final_mask

#         # 7. Attribute Interpolation with Alpha Masking Cutouts
#         if "attr" in return_types and attrs is not None:
#             v_attrs = attrs.to(self.device).repeat_interleave(4, dim=0).unsqueeze(0)
#             attr_img = dr.interpolate(v_attrs, rast, faces)[0]   #.unsqueeze(0)  #[0].squeeze(0) # (H, W, C)
            
#             # # Mask out background canvas pixels and square quad edges
#             # attr_img = attr_img * final_mask.unsqueeze(-1)
#             # out.attr = attr_img.permute(2, 0, 1)  # [C, H, W]
          
#             final_attr_img = dr.antialias(attr_img, rast, v_clip.unsqueeze(0), faces)

#             out.attr = final_attr_img.squeeze(0).permute(2, 0, 1)


#         # 8. Metrics Setup (Depth and Normals)
#         if "depth" in return_types or "normal" in return_types:
#             clip_depth = rast[0, :, :, 2]
#             near, far = 0.1, 100.0
#             linear_depth = (2.0 * near * far) / (far + near - clip_depth * (far - near))
#             linear_depth = linear_depth * final_mask
#             linear_depth = linear_depth.unsqueeze(0).unsqueeze(-1)
#             linear_depth = dr.antialias(linear_depth, rast, v_clip.unsqueeze(0), faces )
#             linear_depth = linear_depth.squeeze(0).squeeze(-1)
            
#             if "depth" in return_types:
#                 out.depth = linear_depth

#         if "normal" in return_types:
#             dx = linear_depth[:, 1:] - linear_depth[:, :-1]
#             dy = linear_depth[1:, :] - linear_depth[:-1, :]

#             dx = dx.unsqueeze(0).unsqueeze(0)
#             dx = torch.nn.functional.pad(dx, (0, 1, 0, 0), mode='replicate').squeeze(0).squeeze(0)
#             dy = dy.unsqueeze(0).unsqueeze(0)
#             dy = torch.nn.functional.pad(dy, (0, 0, 1, 0), mode='replicate').squeeze(0).squeeze(0)

#             normal = torch.stack([-dx, -dy, torch.ones_like(linear_depth)], dim=0)
#             normal = torch.nn.functional.normalize(normal, dim=0)
            
#             # Revert background elements to a clean, flat blue layout
#             normal = torch.where(final_mask.unsqueeze(0) > 0, normal, torch.tensor([[[0.0]], [[0.0]], [[1.0]]], device=self.device))
#             out.normal = normal

#         return out