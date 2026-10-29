#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100_il
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=slat_opt
#SBATCH --output=slat_opt_%j.out

# usage: sbatch scripts/run_slat_optimization.sh [config] [--out-dir DIR]
set -euo pipefail

TRELLIS_ROOT="/home/fr/fr_fr/fr_ba122/projects/trellis1/TRELLIS"
CONFIG="${1:-${TRELLIS_ROOT}/yaml_config/config.yaml}"
shift || true

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
source "${HOME}/.bashrc"

echo "Job ${SLURM_JOB_ID:-local} started on $(hostname)"
cd "${TRELLIS_ROOT}"
module load devel/cuda/12.8
conda activate trellis

"${HOME}/miniforge3/envs/trellis/bin/python" slat_optimization.py --yaml-config "${CONFIG}" "$@"
