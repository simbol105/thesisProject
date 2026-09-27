# thesis3d

A command-line app for the thesis experiments: **correcting 3D reconstructions by refining a
generative model's latent with a differentiable rendering loss**, and scoring the result with the
Stanford-ORB protocol on Digital Twin Catalog objects.

It wraps the existing project rather than replacing it: every command calls the same scripts the
thesis used, in the right conda environment, and reads the same outputs. The project directory is
never modified by this app except where an experiment writes its own results.

```
thesis_app/
├── thesis3d/           the CLI package
├── results/            scores, figures and tables as used in the thesis (snapshot)
├── third_party/        symlinks to the method repositories (TRELLIS, TRELLIS.2, RecGen, SAM3D, ...)
├── requirements/       one requirement file per environment
├── requirements.txt    the CLI's own dependencies
└── pyproject.toml
```

## 1. Install

The CLI itself needs only Python, `typer` and `pyyaml`:

```bash
cd thesis_app
pip install -e .            # provides the `thesis3d` command
# or, without installing:
PYTHONPATH=. python -m thesis3d.cli --help
```

The **experiments** need the four method environments, which cannot share one interpreter (RecGen
is on CUDA 11.8, the rest on 12.8, and several packages are compiled against the local CUDA):

| env | CUDA | used for | install |
|---|---|---|---|
| `trellis2` | 12.8 | TRELLIS.2 generation and refinement, **the benchmark, figures and tables** | `pip install -r requirements/trellis2.txt` |
| `trellis` | 12.8 | TRELLIS generation and refinement (needs an H100) | `third_party/trellis/setup.sh` (see `requirements/trellis.txt`) |
| `sam3d` | 12.8 | SAM3D generation | `pip install -r requirements/sam3d.txt` |
| `recgen` | 11.8 | RecGen generation | `cd third_party/recgen && pixi install` |

Only `trellis2` is needed to read results, rebuild the tables or rerun the benchmark. Check what is
present with:

```bash
thesis3d env
```

By default the app looks for the project one directory above itself; override with
`export THESIS3D_PROJECT=/path/to/projects`.

## 2. What the experiment is

```
prompt image ──► RecGen | SAM3D | TRELLIS | TRELLIS.2         (source mesh)
                        │
                        ├─ encode into a generative latent    (TRELLIS SLAT or TRELLIS.2 SLAT)
                        │
                        └─ refine one of four variables against multi-view reference images:
                             slat-t1        TRELLIS's 8-channel SLAT            64³
                             cubefeats-t1   its decoded FlexiCubes features    256³
                             slat-t2        TRELLIS.2's shape + texture SLAT    32³
                             ovoxel-t2      its decoded o-voxel                512³
                        │
                        └─ score with Stanford-ORB against the DTC ground truth
```

Six objects were used in the thesis (`thesis3d objects`), but **any DTC object in `dataset/DTC/`
can be run** (`thesis3d objects --dataset`, 50 available here). Four sources and four variables give
the 16 refined methods, plus the un-refined baselines and the nvdiffrec multi-view reference.

## 3. Objects and prompt images

Objects can be named by short name (`birdhouse`), full DTC directory name, a unique prefix
(`Bowl`), `all` (the six thesis objects) or `all-dataset` (everything present).

Every object needs one **prompt image**: the single training photo all generative methods are
conditioned on. The six thesis objects already have theirs; a new object needs one chosen first:

```bash
thesis3d objects --dataset            # what exists, and which objects have a prompt
thesis3d prompts views Bowl           # the training views, sorted by camera elevation
thesis3d prompts auto Bowl            # pick a raised three-quarter view (~30 deg elevation)
thesis3d prompts set Bowl im_0065     # or choose one yourself (image stem or frame index)
thesis3d prompts show Bowl            # which image is chosen, and its camera angles
thesis3d prompts list --missing       # objects still without a prompt
```

`prompts auto` reproduces how the thesis prompt set was chosen: the raised three-quarter view that
showed the most of each object. It writes to `thesis_app/prompts.yaml`, the app's own file — the
project's `benchmark/prompts.yaml` is never modified. All commands read the app's file, so new
objects flow through generation, refinement and benchmarking unchanged.

## 4. Running experiments

Every command takes `--source/-s`, `--object/-o` and (where it applies) `--variable/-v`, each
repeatable, each accepting `all`. Add `--dry-run` to print the commands without running them.

```bash
# generate the source meshes from the prompt images
thesis3d generate --source trellis2 --object cup
thesis3d generate --source all --object all            # the six thesis objects
thesis3d generate --source trellis2 --object Bowl      # any other DTC object
thesis3d generate --source all --object all-dataset    # everything available

# refine: one variable, one source, one object ...
thesis3d refine -v slat-t1 -s recgen -o birdhouse
# ... or everything (this is the main experiment; hours on a GPU)
thesis3d refine -v all -s all -o all
# encode and decode only, to check the encoding without optimizing
thesis3d refine -v slat-t2 -s sam3d -o cup --check

# score with Stanford-ORB
thesis3d benchmark -s recgen -v slat-t1 -o all
thesis3d benchmark -s all -v all --nvdiffrec          # everything, including the baselines
thesis3d benchmark -s trellis --variable none         # baselines only

# score one baseline method on one new object, nothing else
thesis3d prompts auto Calculator
thesis3d generate -s sam3d -o Calculator
thesis3d benchmark -s sam3d --variable none -o Calculator
thesis3d results object Calculator

# what exists so far, for any set of objects
thesis3d status
thesis3d status -o Bowl -o Calculator
```

Runs that already finished are skipped, so any of these can be repeated after an interruption;
`--overwrite` (benchmark, generate) and `--force` (refine) redo them.

**Expected outputs** (written in the project, not in the app):

| command | writes |
|---|---|
| `generate` | `generative_runs/v2/<source>/<object>/mesh.{obj,glb}` |
| `refine` | `trellis1/TRELLIS/optimization_output/...`, `trellis2/TRELLIS.2/optimization_output/...`: final meshes, `metrics.jsonl`, `summary.json`, progress renders |
| `benchmark` | `benchmark_output/<method>/{scores.json, view/, albedo/, depth/, normal/}` and `benchmark_output/summary.md` |

## 5. Reading results

`results` reads the snapshot in `results/scores/` by default, so the thesis numbers are available
without running anything. Add `--live` to read the project's current `benchmark_output/` instead.

```bash
thesis3d results summary                        # one row per method
thesis3d results summary -s recgen              # that source, its baseline and refinements
thesis3d results summary --refined-only -m chamfer -m albedo
thesis3d results compare sam3d                  # baseline vs. its four refinements, with +/- markers
thesis3d results method trellis1_slat_recgen_v2 # one method, per object
thesis3d results object gargoyle                # one object, every method
thesis3d results path                           # where the numbers come from
```

Metrics: `chamfer` (ORB scale, ×2e-3), `normal` (1−cos), `depth` (scale-invariant MSE),
`albedo`, `view-psnr`, `view-ssim`, `view-lpips`, `light-psnr`. Lower is better for chamfer,
normal, depth and LPIPS; higher for PSNR and SSIM.

Regenerating the thesis artefacts (needs the `trellis2` env and the live outputs):

```bash
thesis3d results tables                          # -> results/tables.md, the full appendix
thesis3d results figures                         # -> results/figures/, baseline comparison images
thesis3d results figures -v slat-t1 -v ovoxel-t2 # one image pair per refinement
```

## 6. What is in `results/`

| path | contents |
|---|---|
| `results/scores/<method>/scores.json` | per-object and mean scores for all 25 benchmarked methods |
| `results/summary.md` | the one-row-per-method table |
| `results/tables.md` | the full appendix: per-object tables, encoding cost, topology, noise floor |
| `results/figures/overview_*.png` | comparison grids: ground truth on top, one column per object |

Meshes and renders stay in the project (they are tens of gigabytes); the paths are printed by
`thesis3d status` and `thesis3d results path`.

## 7. Notes and limits

- **Supervision is privileged.** The reference images (silhouette, depth, normal, base colour) are
  rendered from the DTC ground-truth mesh, and the model-to-world alignment is estimated against it.
  The numbers therefore measure refinement under ground-truth multi-view supervision.
- **`refine -v slat-t1` needs an H100** (TRELLIS's rasterizer is built for sm_90).
- **One cell is incomplete** in the shipped results: `slat-t2` from RecGen has 5 of 6 objects; the
  Cup run ran out of GPU memory. `thesis3d status` shows it as `5/6`.
- **New objects have no ground-truth comparison problem**: the benchmark renders its targets from
  each object's own DTC scan, so any object in `dataset/DTC/` is scored the same way as the six.
- The app never edits the project's scripts or its prompt file; it only runs them and reads their
  outputs.
