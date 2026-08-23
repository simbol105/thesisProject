"""
Diagnostic script to verify:
1. What mesh type is being created (MeshWithVoxel vs Mesh)
2. Whether mesh attributes actually affect rendered output
3. Whether gradients flow through attributes
"""
import torch
import torch.nn as nn
from pathlib import Path
import json
import numpy as np
from PIL import Image
import trimesh
import yaml

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.representations.mesh import MeshWithVoxel, Mesh
from trellis2.renderers.mesh_renderer import MeshRenderer
# from trellis2.utils.data_utils import get_intrinsics_from_calibration
from opt_utils.utility import get_normalization_matrix
from typing import Dict, Tuple
# from trellis2.data.sparse_tensor import SparseTensor



def get_intrinsics_from_calibration(intr: Dict, orig_wh: Tuple[int, int], render_resolution: int) -> torch.Tensor:
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




def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load config
    with open("yaml_config/config.yaml", "r") as f:
        config_yaml = yaml.safe_load(f)
    
    # Initialize pipeline
    print("📌 Initializing pipeline...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.low_vram = False
    pipeline.cuda()
    
    # Load image and process it
    image_path = "/home/fr/fr_fr/fr_ba122/projects/dataset/DTC/Birdhouse_Lighting001/train/im_0084.png"
    print(f"📌 Loading image from {image_path}")
    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512))
    
    # Run initial pipeline decode (this produces shape_slat, tex_slat)
    print("📌 Running pipeline.run()...")
    with torch.no_grad():
        _, (shape_slat, tex_slat, res) = pipeline.run(
        image,
        return_latent=True,
        pipeline_type='512',
    )
   
    render_resolution = res
    
    print(f"\n✓ shape_slat: coords {shape_slat.coords.shape}, feats {shape_slat.feats.shape}")
    print(f"✓ tex_slat: coords {tex_slat.coords.shape}, feats {tex_slat.feats.shape}")
    
    # Test 1: Decode and check mesh type
    print("\n" + "="*60)
    print("TEST 1: Check mesh types returned by decode")
    print("="*60)
    
    meshes, subs = pipeline.decode_shape_slat(shape_slat, render_resolution)
    print(f"\nShapes from decode_shape_slat:")
    for i, m in enumerate(meshes):
        print(f"  Mesh {i}: type={type(m).__name__}, vertices shape={m.vertices.shape}")
        if hasattr(m, 'attrs'):
            print(f"           has attrs: Yes, shape={m.attrs.shape}")
        else:
            print(f"           has attrs: No")
    
    tex_voxels = pipeline.decode_tex_slat(tex_slat, subs)
    print(f"\nTex voxels from decode_tex_slat:")
    for i, v in enumerate(tex_voxels):
        print(f"  Voxel {i}: coords {v.coords.shape}, feats {v.feats.shape}")
        if hasattr(v, 'shape'):
            print(f"           shape={v.shape}")
        if hasattr(v, 'spatial_shape'):
            print(f"           spatial_shape={v.spatial_shape}")
    
    # Test 2: Create MeshWithVoxel and verify attrs
    print("\n" + "="*60)
    print("TEST 2: Create MeshWithVoxel and check attrs")
    print("="*60)
    
    m = meshes[0]
    v = tex_voxels[0]
    
    mesh_with_voxel = MeshWithVoxel(
        vertices=m.vertices,
        faces=m.faces,
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1.0 / render_resolution,
        coords=v.coords[:, 1:],
        attrs=v.feats,
        voxel_shape=torch.Size([*v.shape, *v.spatial_shape]),
        layout=pipeline.pbr_attr_layout,
    )
    
    print(f"\nMeshWithVoxel created:")
    print(f"  Type: {type(mesh_with_voxel).__name__}")
    print(f"  Is MeshWithVoxel? {isinstance(mesh_with_voxel, MeshWithVoxel)}")
    print(f"  Vertices shape: {mesh_with_voxel.vertices.shape}")
    print(f"  Attrs shape: {mesh_with_voxel.attrs.shape}")
    print(f"  Attrs dtype: {mesh_with_voxel.attrs.dtype}")
    print(f"  Attrs min/max: {mesh_with_voxel.attrs.min():.4f} / {mesh_with_voxel.attrs.max():.4f}")
    print(f"  Attrs requires_grad: {mesh_with_voxel.attrs.requires_grad}")
    
    # Test 3: Try to optimize mesh.attrs and check if rendering changes
    print("\n" + "="*60)
    print("TEST 3: Can we modify attrs and get rendering differences?")
    print("="*60)
    
    # Setup renderer
    # renderer = MeshRenderer()
    renderer = MeshRenderer(
        rendering_options={
            "resolution": render_resolution,
            "near": 0.01,
            "far": 100,
           
        },
        device=device.type,
    )

    
    # Load camera parameters
    with open(config_yaml["transforms_json"], "r") as f:
        transforms_data = json.load(f)
    
    # Get first test view
    test_frame = transforms_data['frames'][0]
    C2W = np.array(test_frame['transform_matrix'])
    # W2C = np.linalg.inv(C2W)
    
    # Extract pose (3x4 extrinsics)
    extrinsics = torch.tensor(C2W, dtype=torch.float32, device=device)
    
    # Get camera intrinsics
    # calibration = {
    #     'cx': transforms_data['cx'],
    #     'cy': transforms_data['cy'],
    #     'fx': transforms_data['fx'],
    #     'fy': transforms_data['fy'],
    # }
    calibration= transforms_data["intrinsic"]
    # intrinsics = get_intrinsics_from_calibration(
    #     calibration, width=512, height=512
    # )
    
    orig_w, orig_h = 2000, 2000
    intrinsics_scaled = get_intrinsics_from_calibration(calibration, (orig_w, orig_h), 512).to(device=device)
    
    # intrinsics_scaled = intrinsics.clone()
    # intrinsics_scaled[:2] /= 512.0  # Normalize to [0, 1]
    # intrinsics_scaled = torch.tensor(intrinsics_scaled, dtype=torch.float32, device=device)
    
    # Load ground truth mesh for alignment
    gt_mesh = trimesh.load_mesh(config_yaml["gt_mesh"])
    
    # Compute fixed object transform
    object_pose = get_normalization_matrix(gt_mesh, mesh_with_voxel)
    
    # Test rendering with unmodified attrs
    print("\nTest A: Render with original attributes...")
    with torch.no_grad():
        buffers_orig = renderer.render(
            mesh=mesh_with_voxel,
            extrinsics=extrinsics,
            intrinsics=intrinsics_scaled,
            return_types=["base_color", "mask"],
            transformation=object_pose
        )
    pred_orig = buffers_orig["base_color"]
    mask_orig = buffers_orig["mask"]
    print(f"  pred shape: {pred_orig.shape}, min/max: {pred_orig.min():.4f} / {pred_orig.max():.4f}")
    
    # Modify attrs and re-render
    print("\nTest B: Modify attributes and re-render...")
    mesh_modified = MeshWithVoxel(
        vertices=mesh_with_voxel.vertices.clone(),
        faces=mesh_with_voxel.faces,
        origin=mesh_with_voxel.origin,
        voxel_size=mesh_with_voxel.voxel_size,
        coords=mesh_with_voxel.coords.clone(),
        attrs=(mesh_with_voxel.attrs * 0.5).clone(),  # Darken by 50%
        voxel_shape=mesh_with_voxel.voxel_shape,
        layout=mesh_with_voxel.layout,
    )
    
    with torch.no_grad():
        buffers_mod = renderer.render(
            mesh=mesh_modified,
            extrinsics=extrinsics,
            intrinsics=intrinsics_scaled,
            return_types=["base_color", "mask"],
            transformation=object_pose
        )
    pred_mod = buffers_mod["base_color"]
    
    # Compare
    diff = (pred_mod - pred_orig).abs()
    print(f"  Modified pred min/max: {pred_mod.min():.4f} / {pred_mod.max():.4f}")
    print(f"  Difference: min={diff.min():.6f}, max={diff.max():.6f}, mean={diff.mean():.6f}")
    
    if diff.mean() < 1e-6:
        print("  ❌ CRITICAL: Modifying attributes has NO effect on rendering!")
        print("     The renderer is not using mesh.attrs!")
    else:
        print("  ✓ Attributes DO affect rendering!")
        print(f"     (darkening attrs by 50% changes image by {diff.mean():.4f} on average)")
    
    # Test 4: Try to backprop through attrs
    print("\n" + "="*60)
    print("TEST 4: Can we backprop through mesh.attrs?")
    print("="*60)
    
    mesh_opt = MeshWithVoxel(
        vertices=mesh_with_voxel.vertices.detach().clone(),
        faces=mesh_with_voxel.faces,
        origin=mesh_with_voxel.origin,
        voxel_size=mesh_with_voxel.voxel_size,
        coords=mesh_with_voxel.coords.detach().clone(),
        attrs=nn.Parameter(mesh_with_voxel.attrs.detach().clone()),  # Make it a parameter
        voxel_shape=mesh_with_voxel.voxel_shape,
        layout=mesh_with_voxel.layout,
    )
    
    print(f"\nOptimizable mesh attrs requires_grad: {mesh_opt.attrs.requires_grad}")
    
    # Render and compute simple loss
    buffers_opt = renderer.render(
        mesh=mesh_opt,
        extrinsics=extrinsics,
        intrinsics=intrinsics_scaled,
        return_types=["base_color", "mask"],
        transformation=object_pose
    )
    pred_opt = buffers_opt["base_color"]
    mask_opt = buffers_opt["mask"]
    
    # Simple loss: mean color
    loss = (pred_opt * mask_opt).mean()
    print(f"\nLoss computed: {loss.item():.6f}")
    print(f"Loss requires_grad: {loss.requires_grad}")
    
    # Try to backprop
    try:
        loss.backward()
        grad = mesh_opt.attrs.grad
        if grad is not None:
            print(f"✓ Backprop successful!")
            print(f"  Attrs gradient: min={grad.min():.6e}, max={grad.max():.6e}, norm={grad.norm():.6e}")
        else:
            print(f"❌ Backprop failed: attrs.grad is None!")
    except Exception as e:
        print(f"❌ Backprop failed with error: {e}")
    
    print("\n" + "="*60)
    print("TESTS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
