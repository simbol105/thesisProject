#!/usr/bin/bash
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=sam3d_setup
#SBATCH --output=sam3d_setup.out

# prepare sam3d project
source ~/miniforge3/etc/profile.d/conda.sh
source ~/.bashrc

cd ~/projects/sam3d 

cd * 
mamba activate sam3d-objects 

python demo.py
echo "Done with Sam3d demo processing, output saved in splat.ply"
