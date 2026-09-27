# FlexiCubes optimization from recgen / SAM3D meshes

The trellis1 counterpart of `trellis2/TRELLIS.2/hypothesis_trellis_init.py`: take the initial
mesh, convert it to an SDF, and optimize that **directly**, with no TRELLIS decoder in the loop.
DMTet + marching tetrahedra there, FlexiCubes + a cube grid here.

```
recgen  thesis_output/<obj>/mesh.obj      (vertex colors, already in [-0.5, 0.5]^3)
SAM3D   sam3d_output/<obj>/mesh.glb
   |
   |  unsigned distance on the (res+1)^3 grid  +  exterior flood fill   (sign)
   v
sdf [Nv]  deform [Nv,3]  weight [Nc,21]  colour [Nv,3]     <-- ALL nn.Parameter
   |  FlexiCubes
   v
mesh -> MeshRenderer -> losses vs. the DTC GT mesh -> backprop
```

**Nothing needs to be cloned.** TRELLIS already vendors the official NVIDIA FlexiCubes at
`trellis/representations/mesh/flexicubes/` — the same copy `SparseFeatures2Mesh` drives. It
differs from upstream in two ways that matter:

* `construct_voxel_grid` was removed; use `utils_cube.construct_dense_grid` (which this run
  does, so the grid is identical to the cubefeats run's).
* `__call__` takes `beta=` / `alpha=` (upstream: `beta_fx12=` / `alpha_fx8=`) and returns four
  values, `(vertices, faces, L_dev, vertices_color)`, not three.

The bundled `flexicubes/examples/optimize.py` is upstream's and no longer matches the class, so
don't copy from it directly — the regularizer weights in `FlexiCubesGeometry.reg_loss` are
taken from it, though.

## The point of this run

It is the **ablation partner** of `cubefeats/`. Same objects, same alignment, same cameras,
same losses, same metrics, same output layout, same FlexiCubes extractor on the same grid. The
loss code is *imported* from `cubefeats.hypothesis_mesh_cubefeats`, not copied, so the two
cannot drift apart. The only difference:

| | `cubefeats/` | `flexicubes/` (this) |
|---|---|---|
| what is optimized | cubefeats `[M, 101]` out of `SLatMeshDecoder(..., return_hidden=True)` | `sdf`, `deform`, `weight`, `colour` on the grid |
| initialized from | mesh -> voxels -> DINOv2 -> SLAT -> decoder | mesh -> SDF |
| constrained by | the TRELLIS decoder's prior; only the occupied 64^3 shell exists | nothing — any SDF the grid can express |
| models loaded | pipeline + SLAT encoder + DINOv2 | none |

So this run tells you how much of the cubefeats run's behaviour is the TRELLIS prior helping
(or hurting) versus free-form geometry fitting.

## The SDF initialization is the thing to watch

**The recgen and SAM3D meshes are not watertight** — both are open, double-sided surfaces. The
usual closest-triangle-normal signed distance therefore marks only a thin crust as inside.
Measured on Birdhouse at res 64, fraction of grid vertices with `sdf < 0`:

| mesh | `raycast` | `floodfill` | notes |
|---|---|---|---|
| recgen | 6.76% | **18.94%** | `sdf.min` −0.028 vs −0.055 |
| SAM3D | 4.06% | **14.12%** | |
| DTC GT | 18.98% | **24.10%** | same `sdf.min` (−0.209) either way — the GT is well behaved |

`raycast` on recgen would hand FlexiCubes a hollow crust: it extracts both an outer and an
inner wall, and there is almost no interior for the SDF to move through. `floodfill` — unsigned
distance, then flood-fill the free cells from the grid boundary and call everything unreached
"inside" — is the Sparc3D algorithm (arXiv:2505.14521) and recovers a solid. It is the default.

`sdf_init: sparc3d` calls the exact backend the trellis2 DMTet run uses
(`sparc3d_sdf.sdf.compute_sdf_on_grid`) if you point `sparc3d_path` at
`trellis2/TRELLIS.2/third_parties/Sparc3Dsdf/src`; it needs kaolin and a GPU. Same algorithm,
different implementation.

Sign convention: FlexiCubes uses `occ_n = scalar_field < 0`, i.e. **negative inside**.

## Files

```
flexicubes/flexicubes_geometry.py        FlexiCubesGeometry  (the dmtet/dmtet.py counterpart)
flexicubes/hypothesis_mesh_flexicubes.py the run
flexicubes/generate_flexicubes_configs.py
flexicubes/run_flexicubes.sh             sbatch, one config
flexicubes/run_all_flexicubes.py / .sh   all configs, one mesh source
flexicubes_configs/<object>.yaml         one per object, holding BOTH mesh paths
```

## Running

```bash
cd /home/fr/fr_fr/fr_ba122/projects/trellis1/TRELLIS
conda activate trellis

python flexicubes/generate_flexicubes_configs.py --overwrite

# sanity check: SDF init + the mesh it already gives, no optimization
python flexicubes/hypothesis_mesh_flexicubes.py \
    --yaml-config flexicubes_configs/Birdhouse_Lighting001.yaml --check

sbatch flexicubes/run_flexicubes.sh flexicubes_configs/Birdhouse_Lighting001.yaml
sbatch flexicubes/run_flexicubes.sh flexicubes_configs/Birdhouse_Lighting001.yaml --mesh-source sam3d
sbatch flexicubes/run_all_flexicubes.sh recgen
```

`--check` writes `init_mesh/flexicubes_init.glb`. **If that does not look like the source mesh,
stop there** — no amount of optimization fixes a bad SDF seed.

## Output layout

Identical to the cubefeats run, under `optimization_output/flexicubes/<recgen|sam3d>/<object>/`:
`alignment.npy`, `init_mesh/`, `opt_IIII_VVV.png`, `test/`, `metrics.jsonl` (iter −1 is the
initialization), `final_mesh/flexicubes_opt.glb` (+ `_world.glb`) and
`final_mesh/flexicubes_grid.pt` with the optimized `sdf`/`deform`/`weight`/`colour`.
So `metrics.jsonl` from the two runs can be compared directly.

## Knobs

* `grid_res: 128` — parameters scale as res^3. At 128: 2,146,689 grid verts / 2,097,152 cubes,
  ~59M parameters ≈ 236 MB, ~950 MB with gradients and Adam state, plus ~290 MB for the cube
  and edge index tensors. 192 is about 3.5 GB; 256 is not worth it.
* `lr: 1e-2` — this is an SDF in world units, not a latent, so it wants a much smaller step
  than the cubefeats run's `6e-2`.
* `optimize_weights` / `optimize_color` — turn the dual weights or the colour off to isolate
  geometry.
* `tsdf_band: 3.0` — the TSDF loss only uses grid vertices within this many voxels of the
  surface, which keeps its cost comparable to the cubefeats run (where `cube2mesh` only exposes
  the surface cubes' vertices). The full dense grid would be ~2M points per view.
* `sdf_l2_weight` — pull the SDF back towards its initialization; the analogue of
  `latent_l2_weight`.

## One difference from the trellis2 DMTet version, on purpose

There, colour is `attr = mesh_.visual.vertex_colors` (one entry per *input mesh* vertex) handed
to `Mesh(vertices=verts, faces=faces, vertex_attrs=attr)` where `verts` comes from marching
tetrahedra — the two have different vertex counts, so it is worth checking that the colours
land where you expect. Here colour lives on the **grid** vertices and FlexiCubes interpolates
it onto the extracted vertices itself (`voxelgrid_colors`, which it puts through `sigmoid`, so
the parameter stores logits), which keeps the counts consistent by construction.
