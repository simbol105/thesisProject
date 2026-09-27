#!/usr/bin/env bash
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu_h100_il
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=thesis_runs
#SBATCH --output=thesis_runs_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=ba122@email.uni-freiburg.de

# Every latent-optimization run of the thesis, in one job on two GPUs.

set -uo pipefail

PROJECTS="/home/fr/fr_fr/fr_ba122/projects"
TRELLIS1="${PROJECTS}/trellis1/TRELLIS"
TRELLIS2="${PROJECTS}/trellis2/TRELLIS.2"
PY1="${HOME}/miniforge3/envs/trellis/bin/python"      # `python` can resolve to another env
PY2="${HOME}/miniforge3/envs/trellis2/bin/python"
SOURCES=(recgen sam3d trellis trellis2)
LOGS="${PROJECTS}/thesis_runs_logs/${SLURM_JOB_ID:-local}"
mkdir -p "${LOGS}"

set +u    # /etc/bashrc and conda activate break under set -u
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
source "${HOME}/.bashrc"
module load devel/cuda/12.8
export PATH=/opt/bwhpc/common/devel/cuda/12.8/bin:$PATH
set -u

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
GPU_A=0
GPU_B=$(( NGPU > 1 ? 1 : 0 ))    # one GPU: both streams share it (slower, still correct)
echo "Job ${SLURM_JOB_ID:-local} on $(hostname), ${NGPU} GPU(s); streams on GPU ${GPU_A} and ${GPU_B}"
nvidia-smi -L
echo "logs: ${LOGS}"

run_step() {
    local name="$1" workdir="$2"; shift 2
    local log="${LOGS}/${name}.log"
    echo "[$(date +%H:%M:%S)] start ${name} (GPU ${CUDA_VISIBLE_DEVICES:-?}) -> ${log}"
    ( cd "${workdir}" && "$@" ) >>"${log}" 2>&1
    local status=$?
    echo "[$(date +%H:%M:%S)] done  ${name} (exit ${status})"
    return 0
}

stream_trellis1() {
    export CUDA_VISIBLE_DEVICES=${GPU_A}
    set +u; conda activate trellis; set -u

    run_step "t1_slat_configs" "${TRELLIS1}" "${PY1}" slat_opt/generate_slat_opt_configs.py
    for src in "${SOURCES[@]}"; do
        run_step "t1_slat_${src}" "${TRELLIS1}" "${PY1}" slat_opt/run_all_slat_opt.py --mesh-source "${src}"
    done

    run_step "t1_cubefeats_configs" "${TRELLIS1}" "${PY1}" cubefeats_opt/generate_cubefeats_opt_configs.py
    run_step "t1_cubefeats" "${TRELLIS1}" "${PY1}" cubefeats_opt/run_all_cubefeats_opt.py \
        --mesh-source "${SOURCES[@]}"
}

stream_trellis2() {
    export CUDA_VISIBLE_DEVICES=${GPU_B}
    set +u; conda activate trellis2; set -u

    run_step "t2_configs" "${TRELLIS2}" "${PY2}" slat_opt/generate_slat_opt_configs.py
    run_step "t2_encode_check" "${TRELLIS2}" "${PY2}" slat_opt/run_all_slat_opt.py \
        --mesh-source "${SOURCES[@]}" --objects Birdhouse_Lighting001 --check
    run_step "t2_slat" "${TRELLIS2}" "${PY2}" slat_opt/run_all_slat_opt.py --mesh-source "${SOURCES[@]}"
}

stream_trellis1 &
PID1=$!
stream_trellis2 &
PID2=$!
wait "${PID1}"; echo "=== TRELLIS1 stream finished"
wait "${PID2}"; echo "=== TRELLIS.2 stream finished"

echo "=== all streams done"
echo "logs:    ${LOGS}"
echo "results:"
echo "  ${TRELLIS1}/optimization_output/slat_opt/"
echo "  ${TRELLIS1}/optimization_output/cubefeats_opt/"
echo "  ${TRELLIS2}/optimization_output/slat_opt/"
