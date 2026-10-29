#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_h100_il
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=slat_dilation
#SBATCH --output=slat_dilation_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=ba122@email.uni-freiburg.de

# The dilation experiment: grow the Birdhouse chimney

set -uo pipefail

TRELLIS_ROOT="${TRELLIS_ROOT:-/home/fr/fr_fr/fr_ba122/projects/trellis1/TRELLIS}"
cd "${TRELLIS_ROOT}"

set +u    # /etc/bashrc and conda activate break under set -u
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
source "${HOME}/.bashrc" >/dev/null 2>&1 || true
module load devel/cuda/12.8
export PATH=/opt/bwhpc/common/devel/cuda/12.8/bin:$PATH
conda activate trellis || { echo "conda env 'trellis' not found" >&2; exit 1; }
set -u

echo "Job ${SLURM_JOB_ID:-local} started on $(hostname) at $(date '+%F %T')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

START=$(date +%s)
"${HOME}/miniforge3/envs/trellis/bin/python" slat_opt/hypothesis_slat_dilation.py "$@"
STATUS=$?
END=$(date +%s)

echo "Job finished at $(date '+%F %T'), wall clock $(( (END - START) / 60 )) min (exit ${STATUS})"
exit ${STATUS}
