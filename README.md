# 3D Object Reconstruction via Latent Priors and Renderings

Thesis code: correcting the hallucinations of image-to-3D generative models by refining their
**latent** representation with a differentiable rendering loss, and benchmarking the result against
the unrefined outputs and against a multi-view optimization baseline (nvdiffrec).

An existing reconstruction (RecGen, SAM3D, TRELLIS or TRELLIS.2) is encoded into TRELLIS's or
TRELLIS.2's published latent space, that latent is optimized against multi-view 2D losses through
the frozen decoder, and the refined mesh is scored with the Stanford-ORB metrics on DTC objects.

---

## 1. Layout

| path | what it is |
|---|---|
| `benchmark/` | generation of every baseline, the Stanford-ORB benchmark on DTC, comparison figures |
| `trellis1/TRELLIS/` | TRELLIS checkout + our refinement code (`slat_optimization.py`, `slat_opt/`, `cubefeats_opt/`) |
| `trellis2/TRELLIS.2/` | TRELLIS.2 checkout + our code (`slat_opt/`, `topology_study/`) |
| `recgen/`, `sam3d/` | the two other generative baselines, with `scripts/` driving them |
| `nvdiffrec/`, `nvdiffrecmc/` | multi-view optimization baselines |
| `Stanford-ORB/` | ORB metric code (used unchanged by the benchmark) |
| `dataset/DTC/` | the six DTC objects: GT mesh, cameras, photos |
| `thesis_runs.sh` | one Slurm job running every refinement experiment |
| `miscellaneous/` | private notes, logs, assistant config (git-ignored) |

Six objects throughout: `Birdhouse_Lighting001`, `Cup_B08TWHJ33Q_Tan_Lighting001`,
`Figurine_B08FYFNYP4_LionKing_Lighting001`, `Gargoyle_B08SQMBDXY_HandsOnKnees_Lighting001`,
`Mallard_B082D168CK_MintGreen_Lighting001`, `Mouse_B0CHNVBBLF_Honeycomb_1_Lighting001`.

## 2. Environments

One conda env per model; `run_script.sh` shows the exact activation used on the cluster
(`source conda.sh`, `module load devel/cuda/<ver>`, `conda activate <env>`, then that env's own
`$CONDA_PREFIX/bin/python`, since plain `python` can resolve elsewhere).

| env | CUDA | used for |
|---|---|---|
| `trellis` | 12.8 | TRELLIS generation and all TRELLIS refinements |
| `trellis2` | 12.8 | TRELLIS.2 generation, refinements, **and the benchmark** |
| `recgen` | 11.8 | RecGen (runs inside its own pixi project) |
| `sam3d` | 12.8 | SAM3D (sets `CUDA_HOME` from `CONDA_PREFIX`) |

TRELLIS needs an H100: its Gaussian rasterizer is built for sm_90 only.

## 3. Baselines: generate and benchmark

```bash
python benchmark/run_all.py                      # any python; each step activates its own env
python benchmark/run_all.py --methods trellis2   # subset
python benchmark/run_all.py --skip-generate      # re-score existing meshes only
```

Prompt images live in `benchmark/prompts.yaml` (tag `v2`, one raised three-quarter view per
object); `benchmark/prompts.py` builds contact sheets to choose them.

Outputs: `generative_runs/v2/<method>/<object>/mesh.{obj,glb}` and
`benchmark_output/<method>_v2/{scores.json, view/, albedo/, depth/, normal/, mesh_world.obj}`,
with every method's row appended to `benchmark_output/summary.md`.

The benchmark alone (ORB targets, rendering, metrics):

```bash
python benchmark/dtc_orb_benchmark.py all --methods nvdiffrec
python benchmark/dtc_orb_benchmark.py all --custom-method NAME \
    --custom-mesh "path/with/{obj}/mesh.ply" --custom-frame world
python benchmark/dtc_orb_benchmark.py summary
```

Steps are `prepare-gt`, `predict`, `evaluate`, `all`, `summary`. Existing files are skipped
(`--overwrite` to redo). `--custom-frame align` registers the mesh to the GT first; `world` takes
it as already registered.

## 4. Experiment 1 — TRELLIS SLAT refinement of a sampled latent (single object)

```bash
cd trellis1/TRELLIS
python slat_optimization.py --yaml-config yaml_config/config.yaml
```

Samples a SLAT with `pipeline.run`, aligns it to the DTC frame, then optimizes the normalized SLAT
features against the reference renders. Outputs under `optimization_output/`: per-iteration panels,
`metrics.jsonl` (PSNR, mask IoU, chamfer) and the final mesh.

## 5. Experiment 2 — refinement of encoded reconstructions (main result)

Four optimized variables × four source meshes × six objects.

```bash
sbatch thesis_runs.sh          # everything below, two GPUs, one job
```

or per variable:

```bash
cd trellis1/TRELLIS
python slat_opt/generate_slat_opt_configs.py            # -> slat_opt_configs/*.yaml
python slat_opt/run_all_slat_opt.py --mesh-source recgen        # SLAT (T1)
python cubefeats_opt/generate_cubefeats_opt_configs.py
python cubefeats_opt/run_all_cubefeats_opt.py --mesh-source recgen sam3d trellis trellis2

cd ../../trellis2/TRELLIS.2
python slat_opt/generate_slat_opt_configs.py
python slat_opt/run_all_slat_opt.py --mesh-source recgen sam3d trellis trellis2   # SLAT (T2) + o-voxel (T2)
python slat_opt/hypothesis_mesh_slat.py --yaml-config slat_opt_configs/Birdhouse_Lighting001.yaml \
    --mesh-source recgen --check                        # encode + decode round trip only
```

Encoding: TRELLIS voxelizes the mesh at 64³ and projects DINOv2 features through the published SLAT
encoder; TRELLIS.2 converts it to an o-voxel at 512³ (`slat_opt/encode_mesh.py`) and uses the
published shape and texture encoders.

Outputs:

```
trellis1/TRELLIS/optimization_output/slat_opt/v2/slat/<source>/<object>/
    init_mesh/, final_mesh/{slat_opt.glb, slat_opt_world.glb, slat.pt}, metrics.jsonl, alignment.npy
trellis1/TRELLIS/optimization_output/cubefeats_opt/v2/<source>/<object>/       (same layout)
trellis2/TRELLIS.2/optimization_output/slat_opt/v2/<source>/<object>/
    init_state.pt, encode_check/roundtrip.json, slat/ and ovoxel/ with
    mesh_{init,final}_world.ply, metrics.jsonl, summary.json (chamfer, topology, gate changes)
```

Runs are skipped when their final mesh exists, so a timed-out job can simply be resubmitted.

Benchmark the refined meshes (all 16 combinations, then the table):

```bash
python benchmark/run_all_opt.py            # --list to preview, --experiments / --sources to subset
```

## 6. Experiment 3 — does the rendering loss change TRELLIS.2 topology?

```bash
cd trellis2/TRELLIS.2
python topology_study/run_both.py --yaml-config topology_study/config.yaml
python topology_study/run_both.py --only slat        # or --only ovoxel
```

`slat_refinement.py` optimizes the shape and texture SLATs; `ovoxel_refinement.py` optimizes the
decoded o-voxel directly, with the voxel set and quad flags fixed by construction. Both report
per-term gradients, gate changes (voxels added/removed, quad flips), topology statistics
(components, Euler characteristic, boundary and non-manifold edges) and a decoder noise floor
(the same latent decoded twice). Outputs go to `optimization_output/` next to the SLAT runs.

## 7. Figures

```bash
python benchmark/make_tiles.py --overview                       # baselines: overview_{color,normals}.png
python benchmark/make_tiles.py --overview --experiments trellis1_slat trellis1_cubefeats \
    trellis2_slat trellis2_ovoxel                               # one pair per refinement
python benchmark/make_tiles.py --methods nvdiffrec recgen_v2 --views 3    # per-object tiles
```

Written to `benchmark_images/`, built from the images the benchmark scored, so every column is the
same DTC test camera: ground truth in the first row, one row per method, one column per object.

## 8. Metrics

Chamfer is bidirectional; the runs report it in millimetres, the benchmark in ORB's squared scale
(×2e-3). The ORB protocol also reports view and relighting PSNR/SSIM/LPIPS, normal error (1−cos),
scale-invariant depth MSE and albedo PSNR, on 8 held-out test views per object. Supervision during
refinement is purely 2D, following the nvdiffrec convention: the GT mesh is rendered into reference
images (mask, depth, normal, unlit base colour) where available, and the GT mesh is used for
alignment and for the chamfer metric only, never inside the optimization objective.
