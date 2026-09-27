# SLAT optimization from recgen / SAM3D / TRELLIS meshes

(Formerly `cubefeats/`. The cubefeats hypothesis is still available with `optimize: cubefeats`.)

`hypothesis_mesh_slat.py` encodes an existing reconstruction into a TRELLIS SLAT and optimizes
that SLAT against the DTC ground truth with a rendering loss, at full resolution (no
downsampling). The fixes that made SLAT optimization work are explained in
`../zz_claude_optimization.md`; this folder applies them to encoded meshes.

```
source meshes: the outputs benchmark/run_all.py generated (and benchmarked) for the prompt set in
benchmark/prompts.yaml (tag v2):
    generative_runs/<tag>/recgen/<obj>/mesh.obj      (RecGen, vertex colors)
    generative_runs/<tag>/sam3d/<obj>/mesh.glb       (SAM3D, texture)
    generative_runs/<tag>/trellis/<obj>/mesh.glb     (TRELLIS official export)
    generative_runs/<tag>/trellis2/<obj>/mesh.glb    (TRELLIS.2 official export)
   |  (a missing mesh stops the run; generate it with benchmark/run_all.py --methods <source>)
   |  refit to [-0.5, 0.5]^3                   ("mesh frame")
   |  64^3 surface voxelization                dataset_toolkits/voxelize.py
   |  DINOv2 tokens of 150 renders             TRELLIS' own rig: sphere Hammersley, r=2, fov 40
   |  SLatEncoder                              slat_enc_swin8_B_64l8_fp16
   v
SLAT [N, 8] @ 64^3
   |  optimize: slat       z = (feats - mean) / std is the variable; the full mesh decoder
   |                       (fp16 transformer + upsampling + flexicubes) runs every iteration
   |  optimize: cubefeats  the decoder output [M, 101] @ 256^3 is the variable (flexicubes only)
   v
mesh -> MeshRenderer -> losses vs. the GT mesh (mask, depth, normal, color, TSDF, reg) -> backprop
```

## Frames and alignment

`alignment: search` finds the similarity `x_world = s R x_mesh + t`: ICP with scale from each
of the 24 axis-aligned rotations (plus the old FPFH-RANSAC as a candidate), picked by
silhouette IoU on 16 training views. Then everything is done in the **mesh frame**: the GT mesh
is mapped with A^-1 and each camera becomes the rigid `(E @ A)[:3] / s`. Prediction and GT normals,
depth and TSDF then share one frame and one unit system. (The old version folded A into the
cameras only, which left the predicted normals in a different frame from the GT normals.)

## Running

```bash
cd /home/fr/fr_fr/fr_ba122/projects/trellis1/TRELLIS

# (re)generate the configs -- the six thesis objects by default
python slat_opt/generate_slat_opt_configs.py --overwrite

# sanity check: alignment, encoding and the initial mesh, no optimization
python slat_opt/hypothesis_mesh_slat.py --yaml-config slat_opt_configs/Birdhouse_Lighting001.yaml --mesh-source recgen --check

# one object
python slat_opt/hypothesis_mesh_slat.py --yaml-config slat_opt_configs/Birdhouse_Lighting001.yaml --mesh-source recgen

# every object, one mesh source per call
python slat_opt/run_all_slat_opt.py --mesh-source recgen
python slat_opt/run_all_slat_opt.py --mesh-source trellis
python slat_opt/run_all_slat_opt.py --mesh-source sam3d
python slat_opt/run_all_slat_opt.py --mesh-source trellis2
```

The configs are written by `generate_slat_opt_configs.py` from `benchmark/prompts.yaml`; rerun it
with `--overwrite` after changing the prompt set. Runs from the first prompt set stay in
`optimization_output/slat_opt/slat/`.

If `python` is the wrong env after `conda activate trellis`, use `~/miniforge3/envs/trellis/bin/python`.
`run_slat_opt.sh` / `run_all_slat_opt.sh` are the sbatch wrappers.

`run_all_slat_opt.py` skips objects whose source mesh is missing and runs whose final mesh already
exists (`--force` reruns them, `--objects <stem> ...` restricts the set).

## Output layout

```
optimization_output/slat_opt/<tag>/<slat|cubefeats>/<recgen|sam3d|trellis|trellis2>/<object>/
    alignment.npy                        mesh frame -> world (4x4)
    init_mesh/source_mesh_world.glb      the input reconstruction, aligned
    init_mesh/<target>_init_world.glb    after encode -> decode, before optimization
    init_mesh/init_VVV.png               test-view panels before optimization
    check/                               (--check) feature_views.png + train-view panels
    opt_IIII_VVV.png                     train-view panels during the run
    test/test_IIII_VVV.png               test-view panels every eval_interval
    metrics.jsonl                        psnr / mask_iou per eval; chamfer_mm at iter -1 and final
    final_mesh/<target>_opt.glb          result in the mesh frame (+ _world.glb)
    final_mesh/<target>.pt               optimized variable {coords, feats, alignment}
```

Panels are `pred | GT | pred normals | GT normals | overlay` (overlay: yellow = both,
red = prediction only, green = GT only).

`metrics.jsonl` iter -1 records two chamfers: `source_chamfer_mm` (the input mesh itself) and
`chamfer_mm` (after encode/decode). Their difference is the round-trip cost of the SLAT
encoding, independent of the optimization.

## Knobs worth knowing

* `lr: 0.06` for `optimize: slat` is in normalized latent units. For `cubefeats` the old runs
  used 1e-2.
* `lambda_mask`, `lambda_depth`, `lambda_normal`, `lambda_color`: at iteration 0 the log prints
  each term's gradient on the variable. On the birdhouse the geometry terms (mask/depth) were
  30-100x weaker than normal/color; raise them if chamfer barely moves.
* `latent_l2_weight`: pull back towards the initial SLAT if the mesh gets noisy.
* `feature_cameras` / `feature_source` / `fill_interior`: unchanged from the cubefeats version.
* `alignment_path`: reuse a saved `alignment.npy` to skip the search.

**Memory:** encoded meshes voxelize to ~30-40k voxels (38,467 for the recgen birdhouse). With
`optimize: slat` the decoder's activations are kept for backprop; use an 80 GB A100 or an H100
and drop `frames_per_iter` first if it is tight.

**Limit:** the voxel coordinates are fixed, so the surface can only form where the source mesh
has voxels. Parts the reconstruction missed cannot appear by optimizing the features alone.

**Prompts:** all four generative models were run on the same training image per object
(`benchmark/prompts.yaml`); each config records it as `prompt_image` / `prompt_idx`.
