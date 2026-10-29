# thesis3d

Command-line app that runs the thesis experiments (generation, latent refinement, Stanford-ORB benchmark) and reads their results.

## Install

```bash
cd thesis_app
pip install -e .                 # provides `thesis3d`
thesis3d env                     # check the method environments
```

| env | CUDA | used for | install |
|---|---|---|---|
| `trellis2` | 12.8 | TRELLIS.2, benchmark, figures, tables | `pip install -r requirements/trellis2.txt` |
| `trellis` | 12.8 | TRELLIS (H100 only) | `third_party/trellis/setup.sh` |
| `sam3d` | 12.8 | SAM3D | `pip install -r requirements/sam3d.txt` |
| `recgen` | 11.8 | RecGen | `cd third_party/recgen && pixi install` |

The project is expected one directory above the app; override with `THESIS3D_PROJECT=/path/to/projects`.

## Objects and prompts

```bash
thesis3d objects --dataset
thesis3d prompts auto Bowl
thesis3d prompts set Bowl im_0065
thesis3d prompts show Bowl
```

## Experiments

Variables: `slat-t1`, `cubefeats-t1`, `slat-t2`, `ovoxel-t2`. Sources: `recgen`, `sam3d`, `trellis`, `trellis2`. Every option accepts `all`; `--dry-run` prints the commands.

```bash
thesis3d generate -s all -o all
thesis3d refine -v all -s all -o all
thesis3d refine -v slat-t2 -s sam3d -o cup --check
thesis3d benchmark -s all -v all --nvdiffrec
thesis3d status
```

Finished runs are skipped; `--overwrite` (generate, benchmark) and `--force` (refine) redo them.

## Results

```bash
thesis3d results summary
thesis3d results compare sam3d
thesis3d results object gargoyle
thesis3d results tables
thesis3d results figures
```

`results` reads the snapshot in `results/` by default and `--live` reads `benchmark_output/`.

| path | contents |
|---|---|
| `results/scores/<method>/scores.json` | per-object and mean scores |
| `results/summary.md` | one row per method |
| `results/tables.md` | appendix tables |
| `results/figures/` | comparison images |
