#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100_il
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=flexicubes_opt
#SBATCH --output=flexicubes_opt_%j.out

# One FlexiCubes optimization (no TRELLIS model is loaded).
#   sbatch flexicubes/run_flexicubes.sh flexicubes_configs/Birdhouse_Lighting001.yaml
#   sbatch flexicubes/run_flexicubes.sh flexicubes_configs/Birdhouse_Lighting001.yaml --mesh-source sam3d
# Everything after the config is forwarded (--mesh-source, --out-dir, --check).
# The --check pass is cheap enough for an interactive allocation:
#   salloc -p dev_gpu_a100_il -n 1 --gres=gpu:1 -t 00:30:00

set -euo pipefail

TRELLIS_ROOT="/home/fr/fr_fr/fr_ba122/projects/trellis1/TRELLIS"
CONFIG="${1:?usage: run_flexicubes.sh <config.yaml> [--mesh-source recgen|sam3d] [--check]}"
shift

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
source "${HOME}/.bashrc"

echo "Job ${SLURM_JOB_ID:-local} started on $(hostname)"
cd "${TRELLIS_ROOT}"
module load devel/cuda/12.8
conda activate trellis

python flexicubes/hypothesis_mesh_flexicubes.py --yaml-config "${CONFIG}" "$@"
