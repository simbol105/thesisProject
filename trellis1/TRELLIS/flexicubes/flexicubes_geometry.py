"""
FlexiCubes geometry with an SDF initialized from a mesh -- the FlexiCubes counterpart of
trellis2/TRELLIS.2/dmtet/dmtet.py.

`DmTetGeometry` takes an initial mesh, converts it to an SDF on a tetrahedral grid, and then
optimizes (sdf, deform) directly through marching tetrahedra, with no TRELLIS decoder in the
loop. This does the same on a *cube* grid with FlexiCubes, which additionally optimizes the
per-cube dual weights:

    mesh  -> SDF on the (res+1)^3 grid vertices        (init only)
          -> nn.Parameter sdf [Nv], deform [Nv, 3], weight [Nc, 21], color [Nv, 3]
          -> FlexiCubes                                -> mesh, rendered, backpropagated

The grid is TRELLIS' own `construct_dense_grid` / `get_defomed_verts`, and the extractor is the
FlexiCubes copy vendored at trellis/representations/mesh/flexicubes -- the same one
`SparseFeatures2Mesh` drives. So this run and the cubefeats run extract meshes through
identical machinery on an identical grid; the only difference is where sdf/deform/weights come
from (free parameters here, the SLAT mesh decoder there). That makes the two directly
comparable.

Note: the vendored FlexiCubes differs from upstream (github.com/nv-tlabs/FlexiCubes) in two
ways that matter, which is why nothing needs to be cloned:
  * `construct_voxel_grid` was removed -- use `utils_cube.construct_dense_grid` instead;
  * `__call__` takes `beta=`/`alpha=` (upstream: `beta_fx12=`/`alpha_fx8=`) and returns
    4 values (vertices, faces, L_dev, vertices_color) rather than 3.
The bundled `flexicubes/examples/optimize.py` is upstream's and no longer matches the class.
"""

import math
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import trimesh
from scipy.ndimage import label

from trellis.representations.mesh.cube2mesh import MeshExtractResult
from trellis.representations.mesh.flexicubes.flexicubes import FlexiCubes
from trellis.representations.mesh.utils_cube import (
    construct_dense_grid,
    cube_edges,
    get_defomed_verts,
)


def sdf_reg_loss(sdf: torch.Tensor, all_edges: torch.Tensor) -> torch.Tensor:
    """Sign-consistency BCE on grid edges -- removes internal floaters that no view can see.

    Verbatim from flexicubes/examples/loss.py (and from dmtet/dmtet.py, which uses the same).
    """
    sdf_f1x6x2 = sdf[all_edges.reshape(-1)].reshape(-1, 2)
    mask = torch.sign(sdf_f1x6x2[..., 0]) != torch.sign(sdf_f1x6x2[..., 1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    if sdf_f1x6x2.numel() == 0:
        return sdf.new_zeros(())
    sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(
        sdf_f1x6x2[..., 0], (sdf_f1x6x2[..., 1] > 0).float()
    ) + torch.nn.functional.binary_cross_entropy_with_logits(
        sdf_f1x6x2[..., 1], (sdf_f1x6x2[..., 0] > 0).float()
    )
    return sdf_diff


# ==========================================================================================
# mesh -> SDF on the grid vertices
# ==========================================================================================

def _raycasting_scene(tri_mesh: trimesh.Trimesh) -> o3d.t.geometry.RaycastingScene:
    mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(tri_mesh.vertices, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(tri_mesh.faces, dtype=np.uint32)),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    return scene


def _grid_points(res: int) -> np.ndarray:
    """The (res+1)^3 grid vertices of `construct_dense_grid`, in [-0.5, 0.5], 'ij' order."""
    rv = res + 1
    idx = np.stack(np.meshgrid(*[np.arange(rv)] * 3, indexing="ij"), axis=-1)
    return idx.reshape(-1, 3).astype(np.float32) / res - 0.5


def sdf_from_mesh(
    tri_mesh: trimesh.Trimesh,
    res: int,
    method: str = "floodfill",
    tau_scale: float = 1.0,
    sparc3d_path: Optional[str] = None,
) -> np.ndarray:
    """Signed distance at the (res+1)^3 grid vertices. Negative inside (FlexiCubes' convention:
    `occ_n = scalar_field < 0`).

    method:
      floodfill -- unsigned distance, then label the *exterior* by flood-filling the cells that
                   are further than tau = tau_scale * sqrt(3)/(2*res) from the surface, starting
                   from the grid boundary; everything not reached is inside. This is the Sparc3D
                   algorithm (arXiv:2505.14521) and the only one of the three that is correct on
                   an open, double-sided mesh -- which is what recgen and SAM3D produce.
      raycast   -- open3d's closest-triangle-normal signed distance. Correct only for watertight
                   meshes; on the recgen/SAM3D reconstructions it marks only a thin crust as
                   inside (measured: 6.8% of the grid vs 18.9% for floodfill, GT is 24.1%).
      sparc3d   -- the exact backend the trellis2 DMTet run uses, via
                   `sparc3d_sdf.sdf.compute_sdf_on_grid`. Needs `sparc3d_path` on sys.path and a
                   GPU (kaolin). Same algorithm as floodfill, different implementation.
    """
    if method == "sparc3d":
        import sys
        if sparc3d_path is not None and sparc3d_path not in sys.path:
            sys.path.append(sparc3d_path)
        from sparc3d_sdf.sdf import compute_sdf_on_grid
        sdf, _ = compute_sdf_on_grid(
            torch.as_tensor(np.asarray(tri_mesh.vertices), dtype=torch.float32),
            torch.as_tensor(np.asarray(tri_mesh.faces), dtype=torch.long),
            resolution=res,
            surface_threshold=math.sqrt(3) / res,
            intermediate_resolutions=[64] if res > 64 else None,
        )
        sdf = sdf.detach().cpu().numpy() if isinstance(sdf, torch.Tensor) else np.asarray(sdf)
        return sdf.reshape(-1).astype(np.float32)

    scene = _raycasting_scene(tri_mesh)
    query = o3d.core.Tensor(_grid_points(res))

    if method == "raycast":
        return scene.compute_signed_distance(query).numpy().astype(np.float32)

    if method != "floodfill":
        raise ValueError(f"Unknown sdf_init: {method}")

    rv = res + 1
    udf = scene.compute_distance(query).numpy().reshape(rv, rv, rv)
    tau = tau_scale * math.sqrt(3) / (2 * res)
    components, _ = label(udf > tau)                       # 6-connected free space
    boundary = np.unique(np.concatenate([
        components[0].ravel(), components[-1].ravel(),
        components[:, 0].ravel(), components[:, -1].ravel(),
        components[:, :, 0].ravel(), components[:, :, -1].ravel(),
    ]))
    boundary = boundary[boundary != 0]
    exterior = np.isin(components, boundary)
    return np.where(exterior, udf, -udf).reshape(-1).astype(np.float32)


def colors_from_mesh(tri_mesh: trimesh.Trimesh, vertex_colors: np.ndarray, res: int) -> np.ndarray:
    """Colour of the closest surface point for every grid vertex, in [0, 1]."""
    scene = _raycasting_scene(tri_mesh)
    hit = scene.compute_closest_points(o3d.core.Tensor(_grid_points(res)))
    prim = hit["primitive_ids"].numpy().astype(np.int64)
    uv = hit["primitive_uvs"].numpy().astype(np.float64)
    tri = np.asarray(tri_mesh.faces)[prim]                 # [N, 3]
    bary = np.stack([1.0 - uv[:, 0] - uv[:, 1], uv[:, 0], uv[:, 1]], axis=-1)
    return np.clip((vertex_colors[tri] * bary[..., None]).sum(axis=1), 0.0, 1.0)


def _logit(x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


# ==========================================================================================
# the geometry
# ==========================================================================================

class FlexiCubesGeometry:
    """Free (sdf, deform, weight, colour) on a res^3 cube grid, extracted with FlexiCubes."""

    def __init__(
        self,
        res: int = 128,
        init_mesh: Optional[trimesh.Trimesh] = None,
        init_colors: Optional[np.ndarray] = None,
        device: str = "cuda",
        sdf_init: str = "floodfill",
        tau_scale: float = 1.0,
        sparc3d_path: Optional[str] = None,
        optimize_weights: bool = True,
        optimize_color: bool = True,
    ):
        self.res = res
        self.device = torch.device(device)
        self.fc = FlexiCubes(device=str(self.device))

        # TRELLIS' grid: verts are integer coords on [0, res], cubes index them
        verts_i, cubes = construct_dense_grid(res, device=str(self.device))
        self.verts_i = verts_i
        self.cubes = cubes
        self.generate_edges()

        n_verts = verts_i.shape[0]
        n_cubes = cubes.shape[0]

        if init_mesh is not None:
            sdf = sdf_from_mesh(init_mesh, res, sdf_init, tau_scale, sparc3d_path)
            sdf = torch.from_numpy(sdf).to(self.device)
            inside = float((sdf < 0).float().mean())
            print(f"[flexicubes] sdf_init={sdf_init}  range [{sdf.min():.4f}, {sdf.max():.4f}]  "
                  f"inside {inside * 100:.2f}% of the {res + 1}^3 grid vertices")
            if inside < 1e-4:
                print("[flexicubes] WARNING: almost nothing is inside -- the initial mesh will be "
                      "empty. Check the sign convention / try sdf_init: raycast.")
        else:
            sdf = torch.rand(n_verts, device=self.device) - 0.1

        self.sdf = nn.Parameter(sdf.clone().detach().float())
        self.deform = nn.Parameter(torch.zeros(n_verts, 3, device=self.device))

        # FlexiCubes' 21 dual weights per cube: beta (12), alpha (8), gamma_f (1).
        # 0 means "unweighted", which is the plain dual-contouring behaviour.
        self.weight = nn.Parameter(torch.zeros(n_cubes, 21, device=self.device))
        self.weight.requires_grad_(optimize_weights)

        if init_mesh is not None and init_colors is not None:
            rgb = colors_from_mesh(init_mesh, init_colors, res)
        else:
            rgb = np.full((n_verts, 3), 0.5, dtype=np.float32)
        # FlexiCubes applies sigmoid() to voxelgrid_colors, so store logits
        self.color = nn.Parameter(
            torch.from_numpy(_logit(rgb).astype(np.float32)).to(self.device)
        )
        self.color.requires_grad_(optimize_color)

    def generate_edges(self) -> None:
        """The unique edges of the cube grid, for `sdf_reg_loss`."""
        with torch.no_grad():
            edges = self.cubes[:, cube_edges.to(self.cubes.device)].reshape(-1, 2)
            edges, _ = torch.sort(edges, dim=1)
            # unique on a flat int64 key: far cheaper than torch.unique(dim=0) on ~25M rows
            n = self.cubes.max().item() + 1
            key = edges[:, 0] * n + edges[:, 1]
            key = torch.unique(key)
            self.all_edges = torch.stack([key // n, key % n], dim=1)

    def parameters(self):
        return [p for p in (self.sdf, self.deform, self.weight, self.color) if p.requires_grad]

    def getOptimizer(self, lr: float, optimizer: str = "adam"):
        if optimizer == "sgd":
            return torch.optim.SGD(self.parameters(), lr=lr)
        return torch.optim.Adam(self.parameters(), lr=lr)

    def grid_verts(self) -> torch.Tensor:
        """Deformed grid vertices in [-0.5, 0.5] (deformation capped at half a voxel)."""
        return get_defomed_verts(self.verts_i, self.deform, self.res)

    def get_mesh(self, training: bool = True, tsdf_band: float = 3.0) -> MeshExtractResult:
        grid_verts = self.grid_verts()
        vertices, faces, L_dev, colors = self.fc(
            grid_verts,
            self.sdf,
            self.cubes,
            self.res,
            beta=self.weight[:, :12],
            alpha=self.weight[:, 12:20],
            gamma_f=self.weight[:, 20],
            voxelgrid_colors=self.color,
            training=training,
        )
        mesh = MeshExtractResult(vertices=vertices, faces=faces, vertex_attrs=colors, res=self.res)
        if training:
            mesh.reg_loss = self.reg_loss(L_dev)
            # TSDF samples, as cube2mesh exposes them -- restricted to a band around the surface
            # so the projection in tsdf_reg_loss stays the same size as in the cubefeats run
            with torch.no_grad():
                band = self.sdf.abs() < (tsdf_band / self.res)
            mesh.tsdf_v = grid_verts[band]
            mesh.tsdf_s = self.sdf[band]
        return mesh

    def reg_loss(self, L_dev: torch.Tensor, sdf_weight: float = 0.2) -> torch.Tensor:
        """flexicubes/examples/optimize.py's regularizer."""
        loss = sdf_reg_loss(self.sdf, self.all_edges).mean() * sdf_weight
        loss = loss + L_dev.mean() * 0.5
        loss = loss + self.weight[:, :20].abs().mean() * 0.1
        return loss

    def state(self) -> dict:
        return {
            "res": self.res,
            "sdf": self.sdf.detach().cpu(),
            "deform": self.deform.detach().cpu(),
            "weight": self.weight.detach().cpu(),
            "color": self.color.detach().cpu(),
        }
