# TRELLIS.2 topology study

Tests whether refining TRELLIS.2's SLAT or o-voxel with a rendering loss can change the mesh topology.

```bash
cd trellis2/TRELLIS.2
python topology_study/run_both.py --yaml-config topology_study/config.yaml
python topology_study/slat_refinement.py  --yaml-config topology_study/config.yaml
python topology_study/ovoxel_refinement.py --yaml-config topology_study/config.yaml
```

## Outputs

```
init_state.pt                   latents and alignment
slat/  ovoxel/
    metrics.jsonl, summary.json   topology, gate changes, PSNR, IoU, chamfer
    mesh_{init,final}_world.ply
    test/                         test-view panels
    latents_final.pt / ovoxel_final.pt
```
