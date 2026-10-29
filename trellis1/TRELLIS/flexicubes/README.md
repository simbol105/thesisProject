# FlexiCubes optimization from existing reconstructions

Converts a RecGen / SAM3D mesh to an SDF and optimizes it directly with FlexiCubes, without the TRELLIS decoder.

```bash
cd trellis1/TRELLIS
python flexicubes/generate_flexicubes_configs.py --overwrite
python flexicubes/hypothesis_mesh_flexicubes.py --yaml-config flexicubes_configs/Birdhouse_Lighting001.yaml --check
sbatch flexicubes/run_flexicubes.sh flexicubes_configs/Birdhouse_Lighting001.yaml --mesh-source sam3d
sbatch flexicubes/run_all_flexicubes.sh recgen
```

`--check` writes the initial mesh only (`init_mesh/flexicubes_init.glb`).

## Outputs

```
optimization_output/flexicubes/<source>/<object>/
    alignment.npy, metrics.jsonl, init_mesh/, test/
    final_mesh/   flexicubes_opt.glb, flexicubes_opt_world.glb, flexicubes_grid.pt
```
