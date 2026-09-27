#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100_il
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=trellis_uco3d
#SBATCH --output=trellis_uco3d_%j.out

# usage: sbatch scripts/run_uco3d.sh [config] [--check]
set -euo pipefail

TRELLIS_ROOT="/home/fr/fr_fr/fr_ba122/projects/trellis1/TRELLIS"
CONFIG="${1:-${TRELLIS_ROOT}/yaml_config/config_uco3d.yaml}"
shift || true

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
source "${HOME}/.bashrc"

echo "Job ${SLURM_JOB_ID:-local} started on $(hostname)"
cd "${TRELLIS_ROOT}"
module load devel/cuda/12.8
conda activate trellis

python hypothesis_slat_uco3d.py --yaml-config "${CONFIG}" "$@"
