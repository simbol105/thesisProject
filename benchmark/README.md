# Stanford-ORB benchmark on DTC

Scores every method's mesh with the Stanford-ORB metrics on the DTC test views (`trellis2` env, GPU).

```bash
python benchmark/dtc_orb_benchmark.py all --methods recgen sam3d nvdiffrec
python benchmark/dtc_orb_benchmark.py all --custom-method NAME \
    --custom-mesh "path/with/{obj}/mesh.glb" --custom-frame world
python benchmark/dtc_orb_benchmark.py summary
```

Steps: `prepare-gt`, `predict`, `evaluate`, `all`, `summary`; finished objects are skipped (`--overwrite` to redo).

## Outputs

```
benchmark_output/
    _gt_orb/<object>/       ORB targets, shared by all methods
    <method>/<object>/      view/ albedo/ depth/ normal/ mesh_world.obj alignment.npy
    <method>/scores.json    per-object and mean scores
    summary.md              one row per method
```

## Notes

- DTC has no per-view environment maps, so relighting scores the same render as novel-view synthesis.
- Chamfer in `scores.json` is ORB's squared distance (m²); `summary.md` reports it at ORB's 2e3 scale.
