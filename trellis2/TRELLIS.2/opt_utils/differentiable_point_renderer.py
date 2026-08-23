import torch
import torch.nn as nn
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import PointsRasterizer, PointsRasterizationSettings, FoVPerspectiveCameras
# Import the unified Pulsar wrapper class
from pytorch3d.renderer.points.pulsar import PulsarPointsRenderer

device = torch.device("cuda:0")

# 1. Standard structural components
cameras = FoVPerspectiveCameras(device=device, znear=0.01, zfar=10.0)
raster_settings = PointsRasterizationSettings(
    image_size=512, 
    radius=0.01, # Pulsar interprets this as the sphere radius
)

# 2. Instantiate PulsarPointsRenderer WITHOUT a compositor
# n_channels defines the output (e.g., 4 for RGBA)
renderer = PulsarPointsRenderer(
    rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
    n_channels=4 
).to(device)

# 3. Dummy O-Voxel point tracking tensors
point_positions = torch.randn((1, 100000, 3), device=device, requires_grad=True)
# Pulsar natively tracks gradients across RGB + Opacity channels when using RGBA
point_features = torch.ones((1, 100000, 4), device=device, requires_grad=True) 

optimizer = torch.optim.Adam([point_positions, point_features], lr=0.01)

# 4. Optimization Loop
for step in range(100):
    optimizer.zero_grad()
    
    point_cloud = Pointclouds(points=point_positions, features=point_features)
    
    # Forward Pass: Pulsar outputs the composed image array directly.
    # 'gamma' dictates the blending sharpness parameters
    rendered_rgba = renderer(point_cloud, gamma=(1e-4,)) 
    
    loss = nn.functional.mse_loss(rendered_rgba[0, ..., :3], target_rgb_image)
    
    loss.backward()
    optimizer.step()
