#!/usr/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100_short
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --job-name=trellis_setup
#SBATCH --output=trellis_setup.out

# prepare sam3d project
source ~/miniforge3/etc/profile.d/conda.sh
source ~/.bashrc

cd ~/projects/trellis2

cd TRELLIS.2
conda activate trellis2

CUDA_HOME=$(dirname $(dirname $(which nvcc)))
echo $CUDA_HOME

. ./setup.sh  --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
