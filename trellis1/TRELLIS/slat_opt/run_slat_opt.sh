#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100_il
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=slat_opt
#SBATCH --output=slat_opt_%j.out

# One SLAT optimization.

set -euo pipefail

TRELLIS_ROOT="/home/fr/fr_fr/fr_ba122/projects/trellis1/TRELLIS"
CONFIG="${1:?usage: run_slat_opt.sh <config.yaml> [--mesh-source recgen|sam3d|trellis|trellis2] [--check]}"
shift

set +u    # /etc/bashrc and conda activate break under set -u
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
source "${HOME}/.bashrc"

echo "Job ${SLURM_JOB_ID:-local} started on $(hostname)"
cd "${TRELLIS_ROOT}"
module load devel/cuda/12.8
export PATH=/opt/bwhpc/common/devel/cuda/12.8/bin:$PATH
conda activate trellis
set -u

"${HOME}/miniforge3/envs/trellis/bin/python" slat_opt/hypothesis_mesh_slat.py --yaml-config "${CONFIG}" "$@"
