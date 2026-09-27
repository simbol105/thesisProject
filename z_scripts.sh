#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_h100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=sam3d_trellis_batch
#SBATCH --output=sam3d_trellis_batch_%j.out

set -euo pipefail

PROJECT_ROOT="/home/fr/fr_fr/fr_ba122/projects"

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
source "${HOME}/.bashrc"

echo "Job ${SLURM_JOB_ID:-local} started on $(hostname)"

cd "${PROJECT_ROOT}/sam3d/sam-3d-objects"
module load devel/cuda/12.8
conda activate sam3d
python scripts/generate_sam3d_configs.py
python scripts/run_all_sam3d_configs.py

cd "${PROJECT_ROOT}/trellis1/TRELLIS"
module load devel/cuda/12.8
conda activate trellis
python scripts/generate_trellis_configs.py
python scripts/run_all_trellis_configs.py

echo "SAM3D and TRELLIS batch completed"
