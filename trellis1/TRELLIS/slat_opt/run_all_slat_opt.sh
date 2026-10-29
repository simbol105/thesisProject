#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100_il
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=slat_opt_batch
#SBATCH --output=slat_opt_batch_%j.out

# SLAT optimization for every object

set -uo pipefail

TRELLIS_ROOT="/home/fr/fr_fr/fr_ba122/projects/trellis1/TRELLIS"
PYTHON="${HOME}/miniforge3/envs/trellis/bin/python"   # `python` can resolve to another env
if [ "$#" -gt 0 ]; then SOURCES=("$@"); else SOURCES=(recgen sam3d trellis trellis2); fi

set +u    # /etc/bashrc and conda activate break under set -u
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
source "${HOME}/.bashrc"

echo "Job ${SLURM_JOB_ID:-local} started on $(hostname) (mesh sources: ${SOURCES[*]})"
cd "${TRELLIS_ROOT}"
module load devel/cuda/12.8
export PATH=/opt/bwhpc/common/devel/cuda/12.8/bin:$PATH
conda activate trellis
set -u

"${PYTHON}" slat_opt/generate_slat_opt_configs.py

FAILED=()
for SOURCE in "${SOURCES[@]}"; do
    echo "=== mesh source: ${SOURCE}"
    "${PYTHON}" slat_opt/run_all_slat_opt.py --mesh-source "${SOURCE}" || FAILED+=("${SOURCE}")
done

if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "finished with failures in: ${FAILED[*]}"
    exit 1
fi
echo "all mesh sources finished"
