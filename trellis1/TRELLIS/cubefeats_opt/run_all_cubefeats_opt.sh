#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100_il
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=cubefeats_opt
#SBATCH --output=cubefeats_opt_%j.out

# TRELLIS1 cubefeats optimization for every object

set -uo pipefail

TRELLIS_ROOT="/home/fr/fr_fr/fr_ba122/projects/trellis1/TRELLIS"
PYTHON="${HOME}/miniforge3/envs/trellis/bin/python"   # `python` can resolve to another env
SOURCES=(recgen sam3d trellis trellis2)

set +u    # /etc/bashrc and conda activate break under set -u
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
source "${HOME}/.bashrc"

echo "Job ${SLURM_JOB_ID:-local} started on $(hostname) (mesh sources: ${SOURCES[*]})"
cd "${TRELLIS_ROOT}"
module load devel/cuda/12.8
export PATH=/opt/bwhpc/common/devel/cuda/12.8/bin:$PATH
conda activate trellis
set -u

"${PYTHON}" cubefeats_opt/generate_cubefeats_opt_configs.py

"${PYTHON}" cubefeats_opt/run_all_cubefeats_opt.py --mesh-source "${SOURCES[@]}"
STATUS=$?

echo "=== done (exit ${STATUS}); results under optimization_output/cubefeats_opt/"
exit ${STATUS}
