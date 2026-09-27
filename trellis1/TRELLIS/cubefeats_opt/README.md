# TRELLIS1 cubefeats optimization from RecGen / SAM3D / TRELLIS / TRELLIS.2 meshes

The same experiment as `slat_opt/`, with a different variable: instead of the SLAT, the mesh
decoder's output before FlexiCubes (the "cubefeats", the upsampled SLAT) is optimized.

```
source mesh: generative_runs/<tag>/{recgen,sam3d,trellis,trellis2}/<obj>/mesh.{obj,glb}
   |  refit to [-0.5, 0.5]^3, 64^3 voxels, DINOv2 features, SLAT encoder      (as slat_opt)
   v
SLAT [N, 8] @ 64^3
   |  SLatMeshDecoder: transformer + 2 subdivision blocks, run ONCE (no grad)
   v
cubefeats [M, 101] @ 256^3   <- optimized
   |  sdf 8 | deform 24 | flexicubes weights 21 | color 48
   |  FlexiCubes -> mesh -> nvdiffrast renders vs. the aligned GT -> 2D losses -> backprop
```

The alignment, frames, losses (mask, depth, normal, color, SSIM, LPIPS, TSDF, FlexiCubes reg),
logging and outputs are `slat_opt/hypothesis_mesh_slat.py`: `hypothesis_mesh_cubefeats.py` only
calls it with `optimize: cubefeats`, so the two experiments differ in the variable alone.

Differences from the SLAT variable:

* 16x more cells (256^3 instead of 64^3) and 101 instead of 8 channels: far more freedom, and no
  decoder in the graph, so the TRELLIS prior acts only through the initialization.
* The SDF channels are in world units (~1/256 per voxel), deform / weights / color are O(1), so the
  SDF is optimized in voxel units (`cubefeats_sdf_voxel_units: true`) and one Adam lr (1e-2) fits
  all channels. With `latent_l2_weight > 0` the cubefeats are pulled back towards the decoded ones.
* The sparse structure is still fixed: cells can change their surface, not appear where the
  decoder produced none.

## Running

```bash
cd /home/fr/fr_fr/fr_ba122/projects/trellis1/TRELLIS
sbatch cubefeats_opt/run_all_cubefeats_opt.sh   # everything: configs, all objects x all 4 sources

# by hand (trellis env)
python cubefeats_opt/generate_cubefeats_opt_configs.py                 # --overwrite --lr ... to change
python cubefeats_opt/hypothesis_mesh_cubefeats.py --yaml-config cubefeats_opt_configs/Birdhouse_Lighting001.yaml --mesh-source recgen --check
python cubefeats_opt/run_all_cubefeats_opt.py --mesh-source recgen sam3d trellis trellis2
```

Missing source meshes and finished runs (`final_mesh/cubefeats_opt_world.glb`) are skipped.

## Outputs

```
optimization_output/cubefeats_opt/<tag>/<source>/<object>/
    alignment.npy, metrics.jsonl (iter -1 = decoded init, with source + init chamfer; "final")
    init_mesh/   cubefeats_init_world.glb, source_mesh_world.glb, test-view panels
    opt_*.png    train-view panels;  test/ test-view panels every eval_interval
    final_mesh/  cubefeats_opt.glb (mesh frame), cubefeats_opt_world.glb (DTC world), cubefeats.pt
```

Benchmark a result like any method:

```bash
python benchmark/dtc_orb_benchmark.py all --custom-method recgen_cubefeats \
    --custom-mesh "trellis1/TRELLIS/optimization_output/cubefeats_opt/v2/recgen/{obj}/final_mesh/cubefeats_opt_world.glb" \
    --custom-frame world
```
