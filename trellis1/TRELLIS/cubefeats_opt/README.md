# TRELLIS cubefeats optimization from existing reconstructions

Same as `slat_opt/`, but optimizes the decoded FlexiCubes features (256³) instead of the SLAT.

```bash
cd trellis1/TRELLIS
sbatch cubefeats_opt/run_all_cubefeats_opt.sh
python cubefeats_opt/generate_cubefeats_opt_configs.py
python cubefeats_opt/hypothesis_mesh_cubefeats.py --yaml-config cubefeats_opt_configs/Birdhouse_Lighting001.yaml --mesh-source recgen --check
python cubefeats_opt/run_all_cubefeats_opt.py --mesh-source recgen sam3d trellis trellis2
```

## Outputs

```
optimization_output/cubefeats_opt/<tag>/<source>/<object>/
    alignment.npy, metrics.jsonl, init_mesh/, test/
    final_mesh/   cubefeats_opt.glb, cubefeats_opt_world.glb, cubefeats.pt
```
