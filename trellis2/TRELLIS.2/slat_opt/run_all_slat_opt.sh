#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_h100_il
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=slat_opt2
#SBATCH --output=slat_opt2_%j.out

# TRELLIS.2 SLAT / o-voxel optimization: every object, every mesh source, in one job.

set -uo pipefail

ROOT="/home/fr/fr_fr/fr_ba122/projects/trellis2/TRELLIS.2"
PYTHON="${HOME}/miniforge3/envs/trellis2/bin/python"   # `python` can resolve to another env
SOURCES=(recgen sam3d trellis trellis2)
CHECK_OBJECT="Birdhouse_Lighting001"

set +u    # /etc/bashrc and conda activate break under set -u
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
source "${HOME}/.bashrc"

echo "Job ${SLURM_JOB_ID:-local} started on $(hostname) (mesh sources: ${SOURCES[*]})"
cd "${ROOT}"
module load devel/cuda/12.8
export PATH=/opt/bwhpc/common/devel/cuda/12.8/bin:$PATH
conda activate trellis2
set -u

echo "=== configs"
"${PYTHON}" slat_opt/generate_slat_opt_configs.py

echo "=== encoding check on ${CHECK_OBJECT}"
"${PYTHON}" slat_opt/run_all_slat_opt.py --mesh-source "${SOURCES[@]}" \
    --objects "${CHECK_OBJECT}" --check

echo "=== optimization"
"${PYTHON}" slat_opt/run_all_slat_opt.py --mesh-source "${SOURCES[@]}"
STATUS=$?

echo "=== done (exit ${STATUS}); results under optimization_output/slat_opt/"
exit ${STATUS}
