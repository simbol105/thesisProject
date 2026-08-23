#!/usr/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100_il 
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=trellis_optimization
#SBATCH --output=trellis_optimization.out

source ~/miniforge3/etc/profile.d/conda.sh
source ~/.bashrc



cd trellis2/TRELLIS.2
module load devel/cuda/12.8
echo $CUDA_HOME
conda activate trellis2



# pip install glfw
# pip install PyOpenGL PyOpenGL_accelerate
# conda install -c conda-forge rtree
# python test_mesh_attrs.py
# python -m pip install libigl
# pip install scikit-learn


python refine_latents.py --yaml-config "/home/fr/fr_fr/fr_ba122/projects/trellis2/TRELLIS.2/yaml_config/config.yaml"   

# python voxel_opt.py --yaml-config "/home/fr/fr_fr/fr_ba122/projects/trellis2/TRELLIS.2/yaml_config/config.yaml"


