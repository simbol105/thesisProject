#!/usr/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100_il 
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=trellis_optimization
#SBATCH --output=trellis_optimization.out


# salloc -p dev_gpu_a100_il -n 1 --gres=gpu:1 -t 00:30:00


source ~/miniforge3/etc/profile.d/conda.sh
source ~/.bashrc



cd trellis2/TRELLIS.2
module load devel/cuda/12.8
echo $CUDA_HOME
conda activate trellis2

# recgen
cd recgen
module load devel/cuda/11.8
echo $CUDA_HOME
conda activate recgen
export PIXI_CACHE_DIR="/scratch/pixi-cache-$USER"
export UV_CACHE_DIR="/scratch/uv-cache-$USER"

python generate.py

# pip install glfw
# pip install PyOpenGL PyOpenGL_accelerate
# conda install -c conda-forge rtree
# python test_mesh_attrs.py
# python -m pip install libigl
# pip install scikit-learn
# conda install pytorch-sparse -c pyg # or  pip install torch-sparse


# python hypothesis_dense_voxel.py --yaml-config "/home/fr/fr_fr/fr_ba122/projects/trellis2/TRELLIS.2/yaml_config/config.yaml"
# python hypothesis_recgen_ovoxel_refinement.py --yaml-config "/home/fr/fr_fr/fr_ba122/projects/trellis2/TRELLIS.2/yaml_config/config.yaml"
# python hypothesis_ovoxel_mesh_refinement.py --yaml-config "/home/fr/fr_fr/fr_ba122/projects/trellis2/TRELLIS.2/yaml_config/config.yaml"




# python hypothesis_ovoxel_refinement.py --yaml-config "/home/fr/fr_fr/fr_ba122/projects/trellis2/TRELLIS.2/yaml_config/config.yaml" 

# python hypothesis_voxel_optimization.py --yaml-config "/home/fr/fr_fr/fr_ba122/projects/trellis2/TRELLIS.2/yaml_config/config.yaml" 

# python hypothesis_latent_refinement.py --yaml-config "/home/fr/fr_fr/fr_ba122/projects/trellis2/TRELLIS.2/yaml_config/config.yaml" 
# python hypothesis_trellis_init.py --yaml-config "/home/fr/fr_fr/fr_ba122/projects/trellis2/TRELLIS.2/yaml_config/config.yaml" 

python refine_latents.py --yaml-config "/home/fr/fr_fr/fr_ba122/projects/trellis2/TRELLIS.2/yaml_config/config.yaml"   

# python voxel_opt.py --yaml-config "/home/fr/fr_fr/fr_ba122/projects/trellis2/TRELLIS.2/yaml_config/config.yaml"

# Setup for sam3d 
cd sam3d/sam-3d-objects
module load devel/cuda/12.8
echo $CUDA_HOME
conda activate sam3d
python demo.py 


# Setup for hunyuan3d
cd hunyuan3d/Hunyuan3D-2
module load devel/cuda/12.8
echo $CUDA_HOME
conda activate hunyuan3d




# Trellis 1 setup
cd trellis1/TRELLIS
module load devel/cuda/12.8
echo $CUDA_HOME
export PATH=/opt/bwhpc/common/devel/cuda/12.8/bin:$PATH
conda activate trellis

python hypothesis_latent_optimization.py --yaml-config /home/fr/fr_fr/fr_ba122/projects/trellis1/TRELLIS/yaml_config/config.yaml

python hypothesis_slat_optimization.py --yaml-config /home/fr/fr_fr/fr_ba122/projects/trellis1/TRELLIS/yaml_config/config.yaml

python hypothesis_test.py --yaml-config /home/fr/fr_fr/fr_ba122/projects/trellis1/TRELLIS/yaml_config/config.yaml


python hypothesis_cubefeats.py --yaml-config /home/fr/fr_fr/fr_ba122/projects/trellis1/TRELLIS/yaml_config/config.yaml

python hypothesis_gaussian_splat.py  --yaml-config /home/fr/fr_fr/fr_ba122/projects/trellis1/TRELLIS/yaml_config/config.yaml
python hypothesis_gaussian_cubefeats.py  --yaml-config /home/fr/fr_fr/fr_ba122/projects/trellis1/TRELLIS/yaml_config/config.yaml


