# Stanford-ORB benchmark on the DTC thesis objects

```bash
cd /home/fr/fr_fr/fr_ba122/projects
conda activate trellis2          # or ~/miniforge3/envs/trellis2/bin/python below
python benchmark/dtc_orb_benchmark.py all --methods recgen sam3d nvdiffrec
```

Needs a GPU (nvdiffrast rendering, ORB's LPIPS/SSIM). Results:

```
benchmark_output/
    _gt_orb/<object>/           ORB-format targets, shared by all methods (built once)
    <method>/<object>/          view/ albedo/ depth/ normal/ mesh_world.obj alignment.npy
    <method>/orb_input.json     ORB's input format (paths to predictions and targets)
    <method>/scores.json        per-object and mean scores
    summary.md                  one row per method
```

Steps can be run separately: `prepare-gt`, `predict --methods ...`, `evaluate --methods ...`,
`summary`. Finished objects are skipped (`--overwrite` to redo). Missing meshes are skipped
(nvdiffrec has no Mouse output; SAM3D only has the objects it has generated).

Other methods (e.g. the refined SLAT meshes, already in the world frame):

```bash
python benchmark/dtc_orb_benchmark.py all --custom-method slat_opt_recgen \
    --custom-mesh "trellis1/TRELLIS/optimization_output/slat_opt/slat/recgen/{obj}/final_mesh/slat_opt_world.glb" \
    --custom-frame world
```

## How the ORB inputs are made

- **Targets** (2048x2048, ORB downsizes x4): test photos (sRGB -> linear `.exr`) and their alpha
  masks; z-depth, camera-space normals and base color rendered from the GT mesh with the DTC test
  cameras (nvdiffrec convention); the GT mesh in the DTC world frame.
- **Predictions** (512x512): depth and normals rendered from each method's mesh with the same
  renderer and cameras; view image = the method's own render if it has one (nvdiffrec
  `validate/val_*_opt.png`), otherwise the mesh's base color; albedo = nvdiffrec `val_*_kd.png` or
  the mesh's base color.
- **Frames:** each mesh is mapped into the DTC world frame with the same procedure for every method
  (similarity: ICP with scale from the 24 axis-aligned rotations, picked by silhouette IoU against
  the training-photo masks). `--custom-frame world` skips this for meshes already in that frame.
- ORB's metric code is used unchanged. `benchmark/shim/pyexr.py` stands in for `pyexr` (not
  installed) using OpenCV's EXR codec.

## Things to state when reporting

- Relighting: DTC has no per-view environment maps, so the relighting entry scores the same
  rendered image as novel-view synthesis against the GT photo, with ORB's scale-invariant metric.
- View metrics compare with the real (lit) photos. nvdiffrec renders with its learned lighting;
  the generative methods only have base color, which puts them at a disadvantage on PSNR/SSIM/LPIPS.
- Depth and normals are rendered from every method's mesh with one renderer (nvdiffrec's own
  `geometry/*_depth.exr` are valid, but stored in a `Z` channel that OpenCV cannot read, and using
  one renderer keeps the camera and normal conventions identical across methods).
- ORB's chamfer uses the GT mesh vertices as GT samples and 30k samples of the prediction; the json
  value is a squared distance in m^2 (the ORB paper tables report 2e3 x this value; `summary.md`
  shows that scale).
- Depth is scale-invariant (one scale per object); normals and chamfer are not.
- Method outputs are evaluated as stored, errors included. SAM3D's GLB textures appear to hold
  linear values although glTF defines base color as sRGB (converted, they match the GT albedo on
  4/6 objects), so SAM3D renders come out too dark. RecGen's colors are dark in its own Gaussian
  output as well (e.g. Gargoyle mean 0.02), so this is its prediction, not an export issue.
- ORB's PSNR never goes below that of a flat grey image; predictions darker than that get the same
  floor value, which is why RecGen and SAM3D tie on some objects.
