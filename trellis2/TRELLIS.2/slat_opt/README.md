# TRELLIS.2 SLAT and o-voxel optimization from existing reconstructions

Encodes a RecGen / SAM3D / TRELLIS / TRELLIS.2 mesh into TRELLIS.2's shape and texture SLATs and refines the SLATs or the decoded o-voxel.

```bash
cd trellis2/TRELLIS.2
sbatch slat_opt/run_all_slat_opt.sh
python slat_opt/generate_slat_opt_configs.py
python slat_opt/hypothesis_mesh_slat.py --yaml-config slat_opt_configs/Birdhouse_Lighting001.yaml --mesh-source recgen --check
python slat_opt/run_all_slat_opt.py --mesh-source recgen sam3d trellis trellis2
```

`--check` encodes and decodes only; `--only slat|ovoxel` runs one refinement.

## Outputs

```
optimization_output/slat_opt/<tag>/<source>/<object>/
    init_state.pt               encoded SLATs and alignment
    encode_check/               roundtrip.json, source and decoded meshes
    slat/  ovoxel/              metrics.jsonl, summary.json, test/, mesh_{init,final}_world.ply
```
