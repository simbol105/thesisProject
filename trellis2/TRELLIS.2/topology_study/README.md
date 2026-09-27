# TRELLIS.2 topology check

Hypothesis (thesis): refining TRELLIS.2's latents (SLAT) or its o-voxel with a rendering loss
cannot change the mesh topology, because of how the mesh is extracted (flexible dual grid, dual
contouring). This folder re-tests it with the fixes that made TRELLIS1 SLAT optimization work.

```bash
cd /home/fr/fr_fr/fr_ba122/projects/trellis2/TRELLIS.2
python topology_study/run_both.py --yaml-config topology_study/config.yaml          # both, in order
python topology_study/slat_refinement.py  --yaml-config topology_study/config.yaml   # or one at a time
python topology_study/ovoxel_refinement.py --yaml-config topology_study/config.yaml
```

Use the `trellis2` env (`~/miniforge3/envs/trellis2/bin/python` if `python` resolves elsewhere).

## Where the topology is decided (from the code)

```
shape SLAT (32 ch, fp16 decoder)
  -> upsampling blocks: to_subdiv(x) -> subdiv > 0          GATE 1: which child voxels exist
  -> h [N, 7] @ res
       0:3  dual vertex = 1.5*sigmoid - 0.5 (in its voxel)  continuous, bounded to +-0.5 voxel
       3:6  intersected = logit > 0                         GATE 2: which quads (faces) exist
       6:7  split weight = softplus                         GATE 3 at inference (torch.where picks
                                                            the diagonal); continuous with train=True
  -> flexible_dual_grid_to_mesh
```

Gates 1 and 2 decide the topology and have **no gradient path** from a rendering loss: the logits
only enter the graph through `> 0`. Gradients reach the latent only through the dual vertex
positions, the split weights and the texture. So topology can change in the SLAT run only as a
side effect (a logit crossing 0 because the shared latent moved for another reason), and in the
o-voxel run not at all. In the o-voxel run the geometry is additionally pinned: each dual vertex
can move at most 1.5 voxels (printed in mm at startup).

## What the scripts measure

* iteration 0: the gradient of every loss term on the latents / o-voxel channels. Expect 0% non-zero
  on `intersected` and on the subdivision logits.
* every `eval_interval` (`metrics.jsonl`) and at the end (`summary.json`):
  * topology of the inference mesh: Euler characteristic, connected components, boundary edges,
    non-manifold edges (genus is only meaningful when there are no boundary edges)
  * gate changes vs. the start: voxels added / removed (gate 1), quad flags flipped (gate 2)
  * `intersected_margin` / `subdiv_margin`: fraction of gate logits within 0.5 of the threshold,
    i.e. how close the gates are to flipping
  * PSNR, silhouette IoU on the test views; chamfer to the GT (mm) at the start and end

So the result is a number, not an impression: "0 voxels added, 0 quads flipped, same Euler
characteristic, chamfer X -> Y" supports the hypothesis; incidental flips in the SLAT run
qualify it ("topology can change, but the loss does not steer it").

## Fixes carried over from TRELLIS1 (see trellis1/TRELLIS/zz_claude_optimization.md)

* alignment: 24 axis-aligned rotations + ICP with scale, picked by silhouette IoU
* one frame: GT mapped into the TRELLIS.2 frame, cameras made rigid `(E @ A)[:3] / s`
* latents optimized normalized, `z = (feats - mean) / std` (raw std is ~5 for shape, ~2.7 for
  texture: the old `lr: 1e-3` on raw feats moved them by ~0.0002 std per step)
* GradScaler, because both decoders run in fp16

## Also different from the old scripts

* the decode used for optimization skips `fill_holes()` (non-differentiable, and it changes
  topology on its own) and uses `train=True` quad splitting so the split weight gets a gradient;
  metrics and exports use the real inference mesh (`train=False`)
* the GT is rendered once per view and cached; pred and GT use the same renderer, cameras and
  loss terms (mask, depth, normal, color with SSIM + LPIPS)
* the o-voxel run starts from the decoder's own o-voxel for the same latents (not a re-voxelized
  GLB with a point renderer), so both runs share the starting mesh and differ only in the variable
* TRELLIS.2's renderer computes normals in camera space, so the normal-frame bug from TRELLIS1
  does not apply here; depth units did (fixed by the rigid cameras)

## Outputs (`out_root`)

```
init_state.pt                     latents from pipeline.run + alignment (shared)
slat/   ovoxel/
    metrics.jsonl, summary.json
    mesh_init_world.ply, mesh_final_world.ply
    opt_IIII_VVV.png              pred | GT | pred normal | GT normal | overlay
    test/                         test-view panels at every eval
    latents_final.pt / ovoxel_final.pt
```

## Possible next step

To test whether topology *could* be recovered if the gates were trainable, the intersected flags
need a soft relaxation (e.g. render each quad with opacity sigmoid(logit) via depth peeling) or
a straight-through estimator on a surrogate. That is a change to the representation, not to the
optimization, and is left out here on purpose.
