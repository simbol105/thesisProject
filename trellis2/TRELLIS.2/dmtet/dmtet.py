
# Get initial structure from DmTeT, 
# Update dmtet directly with the loss  
# If it works, use the logic to update the trellis latent optimization process. 
# 

###############################################################################
# marching tetrahedrons (differentiable)
#
# This function is adapted from 
# https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/ops/conversions/tetmesh.py
###############################################################################

import trimesh 
from trellis2.representations import MeshWithVoxel
import torch
import numpy as np
from pathlib import Path
import igl
from trellis2.representations import Mesh, MeshWithPbrMaterial
import math
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, "..", "third_parties", "Sparc3Dsdf", "src"))
# sys.path.append("third_parties/Sparc3Dsdf")
from sparc3d_sdf.sdf import compute_sdf_on_grid
from sparc3d_sdf.utils import load_obj


# helper functions 

def bbox(vertices: torch.Tensor) -> torch.Tensor:
    min_, max_ = vertices.min(dim=0).values, vertices.max(dim=0).values
    return torch.stack([min_, max_], dim=0)

def normalize(vertices: torch.Tensor, length: float = 1.0) -> torch.Tensor:
    """
    uniform scale and center to unit width cube centered at origin
    """
    min_, max_ = bbox(vertices)
    scale = (max_ - min_).max() / length
    center_ = (min_ + max_) / 2
    return (vertices - center_) / (scale + 1e-8)


def _combine_scene(mesh) -> tuple[torch.Tensor, torch.Tensor]:
    # Handle both Scene and Trimesh objects
    if isinstance(mesh, trimesh.Scene):
        geom = list(mesh.geometry.values())
    else:
        # Single Trimesh object
        geom = [mesh]
    
    vertex_counts = torch.cat([torch.tensor([len(m.vertices)]) for m in geom])
    offsets = [vertex_counts[:i].sum().item() for i in range(len(geom))]
    
    # Handle both numpy arrays and tensors
    def to_tensor(arr):
        if isinstance(arr, torch.Tensor):
            return arr.float()
        else:
            return torch.from_numpy(arr).float()
    
    def to_tensor_long(arr):
        if isinstance(arr, torch.Tensor):
            return arr.long()
        else:
            return torch.from_numpy(arr).long()
    
    vertices = torch.cat([to_tensor(m.vertices) for m in geom])
    faces = torch.cat(
        [to_tensor_long(m.faces) + offset for m, offset in zip(geom, offsets)]
    )

    return vertices, faces


def load_obj(mesh) -> tuple[torch.Tensor, torch.Tensor]:
    # scene = trimesh.load(path)

    vertices, faces = _combine_scene(mesh)
    vertices = normalize(vertices)

    # Convert to numpy arrays for trimesh (must be on CPU)
    vertices_np = vertices.cpu().numpy() if isinstance(vertices, torch.Tensor) else vertices
    faces_np = faces.cpu().numpy() if isinstance(faces, torch.Tensor) else faces
    
    mesh = trimesh.Trimesh(vertices_np, faces_np)
    mesh.process(validate=True)

    vertices = torch.tensor(mesh.vertices).float()
    faces = torch.tensor(mesh.faces).long()

    return vertices, faces


triangle_table = torch.tensor([
        [-1, -1, -1, -1, -1, -1],
        [ 1,  0,  2, -1, -1, -1],
        [ 4,  0,  3, -1, -1, -1],
        [ 1,  4,  2,  1,  3,  4],
        [ 3,  1,  5, -1, -1, -1],
        [ 2,  3,  0,  2,  5,  3],
        [ 1,  4,  0,  1,  5,  4],
        [ 4,  2,  5, -1, -1, -1],
        [ 4,  5,  2, -1, -1, -1],
        [ 4,  1,  0,  4,  5,  1],
        [ 3,  2,  0,  3,  5,  2],
        [ 1,  3,  5, -1, -1, -1],
        [ 4,  1,  2,  4,  3,  1],
        [ 3,  0,  4, -1, -1, -1],
        [ 2,  0,  1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1]
        ], dtype=torch.long, device="cuda")

num_triangles_table = torch.tensor([0,1,1,2,1,2,2,1,1,2,2,1,2,1,1,0], dtype=torch.long, device="cuda")
base_tet_edges = torch.tensor([0,1,0,2,0,3,1,2,1,3,2,3], dtype=torch.long, device="cuda")
v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device="cuda"))


def sort_edges(edges_ex2):
    with torch.no_grad():
        order = (edges_ex2[:,0] > edges_ex2[:,1]).long()
        order = order.unsqueeze(dim=1)

        a = torch.gather(input=edges_ex2, index=order, dim=1)      
        b = torch.gather(input=edges_ex2, index=1-order, dim=1)  

    return torch.stack([a, b],-1)

def map_uv(faces, face_gidx, max_idx):
    N = int(np.ceil(np.sqrt((max_idx+1)//2)))
    tex_y, tex_x = torch.meshgrid(
        torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda"),
        torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda")
    )

    pad = 0.9 / N

    uvs = torch.stack([
        tex_x      , tex_y,
        tex_x + pad, tex_y,
        tex_x + pad, tex_y + pad,
        tex_x      , tex_y + pad
    ], dim=-1).view(-1, 2)

    def _idx(tet_idx, N):
        x = tet_idx % N
        y = torch.div(tet_idx, N, rounding_mode='trunc')
        return y * N + x

    tet_idx = _idx(torch.div(face_gidx, 2, rounding_mode='trunc'), N)
    tri_idx = face_gidx % 2

    uv_idx = torch.stack((
        tet_idx * 4, tet_idx * 4 + tri_idx + 1, tet_idx * 4 + tri_idx + 2
    ), dim = -1). view(-1, 3)

    return uvs, uv_idx

def marching_tets(pos_nx3, sdf_n, tet_fx4):
    with torch.no_grad():
        # Ensure sdf_n is on the same device as tet_fx4 (GPU)
        sdf_n = sdf_n.to(tet_fx4.device)
        occ_n = sdf_n > 0
        occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1,4)
        occ_sum = torch.sum(occ_fx4, -1)
        valid_tets = (occ_sum>0) & (occ_sum<4)
        occ_sum = occ_sum[valid_tets]

        # find all vertices
        all_edges = tet_fx4[valid_tets][:,base_tet_edges].reshape(-1,2)
        all_edges = sort_edges(all_edges)
        unique_edges, idx_map = torch.unique(all_edges,dim=0, return_inverse=True)  
        
        unique_edges = unique_edges.long()
        mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1,2).sum(-1) == 1
        mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device="cuda") * -1
        mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long,device="cuda")
        idx_map = mapping[idx_map] # map edges to verts

        interp_v = unique_edges[mask_edges]
    edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1,2,3)
    edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1,2,1)
    edges_to_interp_sdf[:,-1] *= -1

    denominator = edges_to_interp_sdf.sum(1,keepdim = True)

    edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1])/denominator
    verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

    idx_map = idx_map.reshape(-1,6)

    tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
    num_triangles = num_triangles_table[tetindex]

    # Generate triangle indices
    faces = torch.cat((
        torch.gather(input=idx_map[num_triangles == 1], dim=1, index=triangle_table[tetindex[num_triangles == 1]][:, :3]).reshape(-1,3),
        torch.gather(input=idx_map[num_triangles == 2], dim=1, index=triangle_table[tetindex[num_triangles == 2]][:, :6]).reshape(-1,3),
    ), dim=0)

    # Get global face index (static, does not depend on topology)
    num_tets = tet_fx4.shape[0]
    tet_gidx = torch.arange(num_tets, dtype=torch.long, device="cuda")[valid_tets]
    face_gidx = torch.cat((
        tet_gidx[num_triangles == 1]*2,
        torch.stack((tet_gidx[num_triangles == 2]*2, tet_gidx[num_triangles == 2]*2 + 1), dim=-1).view(-1)
    ), dim=0)

    uvs, uv_idx = map_uv(faces, face_gidx, num_tets*2)

    return verts, faces, uvs, uv_idx

def sdf_reg_loss(sdf, all_edges):
    sdf_f1x6x2 = sdf[all_edges.reshape(-1)].reshape(-1,2)
    mask = torch.sign(sdf_f1x6x2[...,0]) != torch.sign(sdf_f1x6x2[...,1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,0], (sdf_f1x6x2[...,1] > 0).float()) + \
            torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,1], (sdf_f1x6x2[...,0] > 0).float())
    return sdf_diff

class DmTetGeometry: 
    def __init__(self, scale,res=128, init_mesh=None, device = 'cuda'):
        self.resolution = res 
        self.device = device
        tet_path = Path(f"dmtet/tets/{self.resolution}_tets.npz")
        tets = np.load(tet_path)
        self.verts = torch.tensor(tets['vertices']).float().to(self.device) * scale
        self.indices = torch.tensor(tets['indices']).long().to(self.device)
        self.generate_edges()
        if (init_mesh is not None):
            if hasattr(init_mesh, "simplify"): 
                init_mesh.simplify(target=100000)
            sdf = self.get_sdf(init_mesh)
        else:
            sdf = torch.rand_like(self.verts[:, 0]) - 0.1
        self.sdf = sdf.clone().detach().requires_grad_(True)
        self.deform = torch.zeros_like(self.verts).clone().requires_grad_(True)
        
    def generate_edges(self):
        with torch.no_grad():
            edges = torch.tensor([0,1,0,2,0,3,1,2,1,3,2,3], dtype = torch.long, device = "cuda")
            all_edges = self.indices[:, edges].reshape(-1, 2)
            all_edges_sorted =  torch.sort(all_edges, dim=1)[0]
            self.all_edges = torch.unique(all_edges_sorted, dim=0)
    
    def parameters(self):
        return [self.sdf, self.deform]
    
    def getOptimizer(self, lr_pos):
        return torch.optim.Adam(self.parameters(), lr=lr_pos)
    
    # def get_sdf(self, verts, faces): 
    #     S, I, C = igl.signed_distance(self.verts.cpu().numpy(), verts.cpu().numpy(), faces.cpu().numpy())
        
    #     # mesh = trimesh.Trimesh(verts.cpu().numpy(), faces.cpu().numpy(), process=False)
    #     # sdf = mesh.nearest.signed_distance(self.verts.cpu().numpy())
    #     # return torch.tensor(sdf, device = verts.device, dtype=torch.float)
    #     return torch.tensor(S, device = verts.device, dtype=torch.float)
    
    # def get_sdf(self, mesh, N=512):

    #     # query = self.verts.cpu().numpy()
    #     # V = verts.cpu().numpy()
    #     # F = faces.cpu().numpy()

    #     # N = query.shape[0]
    #     # k = max(1, int(N * sample_ratio))
    #     # idx = np.random.choice(N, k, replace=False)
    #     # query_sampled = query[idx]

    #     # S_sampled, I, C, _ = igl.signed_distance(query_sampled, V, F)
        

    #     # # Reconstruct full SDF by nearest-neighbor fill 
     
    #     # S_full = np.zeros(N)
    #     # S_full[idx] = S_sampled

    #     # from sklearn.neighbors import NearestNeighbors
    #     # nbrs = NearestNeighbors(n_neighbors=1).fit(query_sampled)
    #     # _, nn_idx = nbrs.kneighbors(query)
    #     # S_full = S_sampled[nn_idx[:, 0]]

    #     # return torch.tensor(S_full, device=verts.device, dtype=torch.float)
    #     verts, faces = load_obj(mesh)
    #     sdf, grid = compute_sdf_on_grid(
    #         verts.cpu().numpy(),
    #         faces,
    #         resolution=N,
    #         surface_threshold=math.sqrt(3) / N,
    #         intermediate_resolutions=[64, 256],
    #     )
    #     # Handle both numpy arrays and tensors, place on GPU device
    #     if isinstance(sdf, torch.Tensor):
    #         return sdf.detach().clone().to(dtype=torch.float, device=self.device)
    #     else:
    #         return torch.tensor(sdf, dtype=torch.float, device=self.device)
    # def get_sdf(self, mesh, N=256):
    # # 1. Load and normalize mesh
    #     verts, faces = load_obj(mesh)

    #     # 2. Compute dense SDF grid
    #     sdf_grid, _ = compute_sdf_on_grid(
    #         verts.cpu().numpy(),
    #         faces,
    #         resolution=N,
    #         surface_threshold=math.sqrt(3) / N,
    #         intermediate_resolutions=[64, 256],
    #     )

    #     # 3. Convert to torch
    #     sdf_grid = torch.tensor(sdf_grid, dtype=torch.float32, device=self.device)
    #     sdf_grid = sdf_grid[None, None]  # [1,1,D,H,W]

    #     # 4. Normalize DMTet vertices to [-1,1]
    #     v = self.verts / self.verts.abs().max()

    #     # 5. Convert to grid_sample coords: [0,1] → [-1,1]
    #     coords = (v + 1) / 2
    #     coords = coords[:, [2,1,0]]  # z,y,x ordering
    #     coords = coords[None, None]  # [1,1,N,3]

    #     # 6. Sample SDF at tet vertices
    #     sdf_samples = torch.nn.functional.grid_sample(
    #         sdf_grid,
    #         coords,
    #         mode='bilinear',
    #         padding_mode='border',
    #         align_corners=True
    #     ).view(-1)

    #     return sdf_samples
    
    def get_sdf(self, mesh, N=256):
        # 1. Load and normalize mesh to unit cube
        verts, faces = load_obj(mesh)

        # 2. Compute dense SDF grid
        sdf_grid, _ = compute_sdf_on_grid(
            verts.cpu().numpy(),
            faces,
            resolution=N,
            surface_threshold=math.sqrt(3) / N,
            intermediate_resolutions=[64, 256],
        )

        # 3. Convert to torch (avoid warning)
        if isinstance(sdf_grid, torch.Tensor):
            sdf_grid = sdf_grid.detach().clone().to(self.device, dtype=torch.float32)
        else:
            sdf_grid = torch.as_tensor(sdf_grid, dtype=torch.float32, device=self.device)

        # sdf_grid: [D, H, W] -> [1,1,D,H,W]
        sdf_grid = sdf_grid.unsqueeze(0).unsqueeze(0)

        # 4. Normalize DMTet vertices to [-1, 1] (assumes verts live in same cube)
        v = self.verts  # [num_vertices, 3]
        v = v / v.abs().max()  # crude but consistent normalization

        # 5. Build grid_sample coords: [1, D_out, H_out, W_out, 3]
        # We treat points as a 1D "volume": D_out = N_pts, H_out = 1, W_out = 1
        coords = (v + 1) / 2          # [0,1]
        coords = coords[:, [2, 1, 0]] # z, y, x
        coords = coords.view(1, -1, 1, 1, 3) * 2 - 1  # -> [-1,1], shape [1, N_pts, 1, 1, 3]

        # 6. Sample SDF at tet vertices
        sdf_samples = torch.nn.functional.grid_sample(
            sdf_grid,
            coords,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        ).view(-1)  # [num_vertices]

        return sdf_samples


        

    
    @torch.no_grad()
    def getAABB(self):
        return torch.min(self.verts, dim=0).values, torch.max(self.verts, dim=0).values
    
    # def get_mesh(self, material):
    #     v_deformed = self.verts + 2 / (self.resolution * 2) * torch.tanh(self.deform)
    #     verts, faces, uvs, uv_idx = marching_tets(v_deformed, self.sdf, self.indices)
    #     # imesh = mesh.Mesh(verts, faces, v_tex=uvs, t_tex_idx=uv_idx, material=material)
        
    #     # # with torch.no_grad():
    #     imesh = mesh.auto_normals(imesh)
    #     imesh = mesh.compute_tangents(imesh)
    #     return imesh
    
    # def get_mesh(self, resolution, coords, feats,shape, spatial_shape, pbr_att_layout):
    def get_mesh(self, vertex_attr):
        # 1. Extract geometry from DMTet
        v_deformed = self.verts + 2 / (self.resolution * 2) * torch.tanh(self.deform)
        verts, faces, uvs, uv_idx = marching_tets(v_deformed, self.sdf, self.indices)

        # imesh = MeshWithVoxel(
        #     vertices = verts,                      
        #     faces = faces,                      
        #     origin=[-0.5, -0.5, -0.5],  
        #     voxel_size=1.0 / resolution,
        #     coords=coords[:, 1:],     
        #     attrs=feats,              
        #     voxel_shape=torch.Size([*shape, *spatial_shape]),
        #     layout=pbr_att_layout,
        # )
        imesh = Mesh(vertices=verts, faces=faces, vertex_attrs=vertex_attr)
        # mesh = MeshWithPbrMaterial(vertices=verts, faces=faces, materials=material, uv_coords=uvs, material_ids=uv_idx)
        return imesh
        # return mesh 
    
    def get_sdf_loss(self, sdf_weight=0.05): 
        sdf_loss = sdf_reg_loss(self.sdf, self.all_edges).mean() * sdf_weight
        return sdf_loss

            
            
    def forward(self,render, target_img,render_options,  loss_fn, material, denoiser=None ):
        opt_mesh = self.get_mesh(material)
        buffers = render(opt_mesh, **render_options)
        img_loss = torch.nn.functional.mse_loss(buffers['base_color'], target_img[:3] ) # normal rgb loss
        img_loss += loss_fn(buffers['base_color'] * buffers['mask'], target_img) # rgb loss with mask
        
        # SDF regularizer 
        sdf_weight = 0.1
        sdf_loss = sdf_reg_loss(self.sdf, self.all_edges).mean() * sdf_weight
        
        return img_loss, sdf_loss


