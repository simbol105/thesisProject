# TRELLIS.2 SLAT optimization from RecGen / SAM3D / TRELLIS / TRELLIS.2 meshes

The TRELLIS.2 counterpart of `trellis1/TRELLIS/slat_opt`: an existing reconstruction is encoded into
TRELLIS.2's latents and then refined with the rendering loss.

```
source mesh: generative_runs/<tag>/{recgen,sam3d,trellis,trellis2}/<obj>/mesh.{obj,glb}
   |  (the meshes benchmark/run_all.py generated and benchmarked; tag from benchmark/prompts.yaml)
   |  refit to [-0.5, 0.5]^3                                data_toolkit/dual_grid.py
   |  o_voxel.convert.mesh_to_flexible_dual_grid @ 512     o-voxel: voxels, dual vertices, quad flags
   |  shape_enc_next_dc_f16c32_fp16                          -> shape SLAT   [N, 32] @ 32^3
   |  PBR voxels on the same voxels (base color sampled from the texture / vertex colors,
   |  metallic 0, roughness 1, alpha 1)
   |  tex_enc_next_dc_f16c32_fp16                            -> texture SLAT [N, 32] @ 32^3
   v
<out>/init_state.pt
   |-- topology_study/slat_refinement.py    both SLATs through the decoders       -> <out>/slat/
   |-- topology_study/ovoxel_refinement.py  the decoded o-voxel directly          -> <out>/ovoxel/
```

The encoders are TRELLIS.2's own (published in `microsoft/TRELLIS.2-4B`, now in the HF cache), the
o-voxel conversion is TRELLIS.2's own data pipeline, and the refinement is the one of the topology
study (`topology_study/`): alignment search, GT and rigid cameras in the TRELLIS.2 frame, normalized
latents, GradScaler, topology and gate statistics. `o_voxel`'s own PBR voxelizer only accepts
textured PBR meshes, so the PBR voxels are built here from surface samples, which also works for
RecGen's vertex colors.

## Running

```bash
cd /home/fr/fr_fr/fr_ba122/projects/trellis2/TRELLIS.2
sbatch slat_opt/run_all_slat_opt.sh   # everything: configs, encoding check, all objects x all 4 sources

# by hand (trellis2 env, CUDA 12.8)
python slat_opt/generate_slat_opt_configs.py          # configs from benchmark/prompts.yaml
python slat_opt/hypothesis_mesh_slat.py --yaml-config slat_opt_configs/Birdhouse_Lighting001.yaml --mesh-source recgen --check
python slat_opt/hypothesis_mesh_slat.py --yaml-config slat_opt_configs/Birdhouse_Lighting001.yaml --mesh-source recgen
python slat_opt/run_all_slat_opt.py --mesh-source recgen sam3d trellis trellis2
```

`--check` encodes the mesh, decodes it again and writes `encode_check/roundtrip.json`
(chamfer between source and decoded mesh, in the normalized frame) plus both meshes: the cost of
the encoding before any optimization. `--only slat|ovoxel` runs one refinement. Missing source
meshes, an existing `init_state.pt` and finished refinements (`summary.json`) are skipped.

## Outputs

```
optimization_output/slat_opt/<tag>/<source>/<object>/
    init_state.pt                 encoded SLATs (+ the alignment, added on first use)
    run_config.yaml               the config the refinements ran with
    encode_check/                 (--check) roundtrip.json, source/decoded meshes
    slat/  ovoxel/                topology_study outputs: metrics.jsonl, summary.json (chamfer,
                                  topology, gate changes), test/ panels, mesh_init_world.ply,
                                  mesh_final_world.ply (DTC world frame, vertex colors)
```

Benchmark a refined output like any method (already in the world frame):

```bash
python benchmark/dtc_orb_benchmark.py all --custom-method recgen_trellis2slat \
    --custom-mesh "trellis2/TRELLIS.2/optimization_output/slat_opt/v2/recgen/{obj}/slat/mesh_final_world.ply" \
    --custom-frame world
```

## Knobs

* `encode_res: 512` (SLAT at 32^3, like the 512 pipeline). 1024 gives a 64^3 SLAT and finer
  o-voxels but much more memory in the refinement.
* `optimize: [slat, ovoxel]`, `num_iters`, `lr` (normalized latents), `lr_ovoxel`, loss weights:
  as in `topology_study/config.yaml`.
* The TRELLIS.2 topology findings apply here too: the rendering loss does not reach the
  subdivision and quad-flag gates, so topology changes only as a side effect in the SLAT
  refinement and not at all in the o-voxel refinement.
