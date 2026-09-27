#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_h100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=flexicubes_batch
#SBATCH --output=flexicubes_batch_%j.out

# Every config, sequentially, for one mesh source.
#   sbatch flexicubes/run_all_flexicubes.sh            # recgen meshes
#   sbatch flexicubes/run_all_flexicubes.sh sam3d      # SAM3D meshes

set -euo pipefail

TRELLIS_ROOT="/home/fr/fr_fr/fr_ba122/projects/trellis1/TRELLIS"
MESH_SOURCE="${1:-recgen}"

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
source "${HOME}/.bashrc"

echo "Job ${SLURM_JOB_ID:-local} started on $(hostname) (mesh source: ${MESH_SOURCE})"
cd "${TRELLIS_ROOT}"
module load devel/cuda/12.8
conda activate trellis

python flexicubes/run_all_flexicubes.py --mesh-source "${MESH_SOURCE}"
