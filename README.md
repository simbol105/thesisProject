# 3D Object Reconstruction via Latent Priors and Renderings

Refining the latents of image-to-3D generative models (feed forward models) with a differentiable inverse rendering loss, benchmarked with Stanford-ORB on DTC objects.

## Layout

| path | contents |
|---|---|
| `benchmark/` | baseline generation, Stanford-ORB benchmark, figures and tables |
| `trellis1/TRELLIS/` | TRELLIS + refinement code (`slat_optimization.py`, `slat_opt/` incl. dilation, `cubefeats_opt/`, `flexicubes/`) |
| `trellis2/TRELLIS.2/` | TRELLIS.2 + refinement code (`slat_opt/`, `topology_study/`) |
| `recgen/`, `sam3d/` | generative baselines |
| `nvdiffrec/`, `nvdiffrecmc/` | multi-view optimization baselines |
| `Stanford-ORB/` | ORB metric code (unchanged) |
| `dataset/DTC/` | DTC objects: GT mesh, cameras, photos |
| `thesis_app/` | `thesis3d` CLI wrapping all experiments |
| `thesis_runs.sh` | Slurm job running every refinement experiment |

Objects: `Birdhouse_Lighting001`, `Cup_B08TWHJ33Q_Tan_Lighting001`, `Figurine_B08FYFNYP4_LionKing_Lighting001`,
`Gargoyle_B08SQMBDXY_HandsOnKnees_Lighting001`, `Mallard_B082D168CK_MintGreen_Lighting001`, `Mouse_B0CHNVBBLF_Honeycomb_1_Lighting001`.

## Environments

| env | CUDA | used for |
|---|---|---|
| `trellis` | 12.8 | TRELLIS generation and refinements (H100 only) |
| `trellis2` | 12.8 | TRELLIS.2 generation and refinements, benchmark, figures |
| `recgen` | 11.8 | RecGen (pixi project) |
| `sam3d` | 12.8 | SAM3D |

## Baselines

```bash
python benchmark/run_all.py                      # generate + benchmark every method
python benchmark/run_all.py --methods trellis2
python benchmark/run_all.py --skip-generate      # re-score existing meshes
```

## Experiment 1: TRELLIS SLAT refinement

```bash
cd trellis1/TRELLIS
python slat_optimization.py --yaml-config yaml_config/config.yaml
```

## Experiment 2: refinement of encoded reconstructions

```bash
sbatch thesis_runs.sh                    # all variables, sources and objects
python benchmark/run_all_opt.py          # benchmark the refined meshes
```

Per variable: see `trellis1/TRELLIS/slat_opt/`, `trellis1/TRELLIS/cubefeats_opt/` and `trellis2/TRELLIS.2/slat_opt/`.

## Experiment 3: TRELLIS.2 topology

```bash
cd trellis2/TRELLIS.2
python topology_study/run_both.py --yaml-config topology_study/config.yaml
```

## Experiment 4: growing missing structure by SLAT dilation

```bash
cd trellis1/TRELLIS
sbatch slat_opt/run_slat_dilation.sh                                # all arms: baseline, shell, random, grow, guided
python slat_opt/hypothesis_slat_dilation.py --arms baseline guided   # a subset; --check for a short test
```

Config: `slat_opt_configs/dilation/Birdhouse_chimney.yaml`; results in `optimization_output/slat_dilation/Birdhouse_chimney/summary.md`.

## Figures and tables

```bash
python benchmark/make_tiles.py --overview
python benchmark/make_tiles.py --overview --experiments trellis1_slat trellis1_cubefeats trellis2_slat trellis2_ovoxel
python benchmark/make_tables.py --out tables.md
```
