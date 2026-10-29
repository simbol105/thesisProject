# TRELLIS SLAT optimization from existing reconstructions

Encodes a RecGen / SAM3D / TRELLIS / TRELLIS.2 mesh into a TRELLIS SLAT and refines it with a rendering loss.

```bash
cd trellis1/TRELLIS
python slat_opt/generate_slat_opt_configs.py --overwrite
python slat_opt/hypothesis_mesh_slat.py --yaml-config slat_opt_configs/Birdhouse_Lighting001.yaml --mesh-source recgen --check
python slat_opt/hypothesis_mesh_slat.py --yaml-config slat_opt_configs/Birdhouse_Lighting001.yaml --mesh-source recgen
python slat_opt/run_all_slat_opt.py --mesh-source recgen      # also: sam3d, trellis, trellis2
```

`--check` stops after alignment and encoding. `run_slat_opt.sh` / `run_all_slat_opt.sh` are the sbatch wrappers.

## Outputs

```
optimization_output/slat_opt/<tag>/slat/<source>/<object>/
    alignment.npy               mesh frame -> world
    init_mesh/                  source and encoded meshes, test-view panels
    test/                       test-view panels
    metrics.jsonl               psnr, mask_iou, chamfer_mm
    final_mesh/                 slat_opt.glb, slat_opt_world.glb, slat.pt
```
