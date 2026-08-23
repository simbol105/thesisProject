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
# Install mamba 
#curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
#bash Miniforge3-$(uname)-$(uname -m).sh
#echo "Finished downloading and installing mamba"

# Clone the sam3d repo
#git init 
#git clone https://github.com/facebookresearch/sam-3d-objects.git
#echo "Finished cloning the sam3d repo"
cd * 
# Following setup instruction 
#mamba env create -f environments/default.yml
mamba activate sam3d-objects 
#export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
#pip install -e '.[dev]'
#pip install -e '.[p3d]'
#export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
#pip install -e '.[inference]'
#./patching/hydra
#echo "Finished Setting up SAM3d"
#install huggingface cli
#export HF_TOKEN="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" 
#pip install 'huggingface-hub[cli]<1.0' 
#huggingface-cli login --token $HF_TOKEN --add-to-git-credential
#TAG=hf
#hf download \
#  --repo-type model \
#  --local-dir checkpoints/${TAG}-download \
#  --max-workers 1 \
#  facebook/sam-3d-objects
#mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}
#rm -rf checkpoints/${TAG}-download
#echo "Finished downloading SAM3D model from hugginface" 

# try the setup with the demo.py file 
#touch demo.py
#cat <<EOF >demo.py
#import sys
 # import inference code 
#sys.path.append("notebook")
#from inference import Inference, load_image, load_single_mask

# load model
#tag = "hf"
#config_path = f"checkpoints/{tag}/pipeline.yaml"
#inference = Inference(config_path, compile=False)

# load image and mask
#image = load_image("~/projects/dataset/dtc/test/im_0086.png")

#mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)
#mask = load_image("~/projects/dataset/dtc/test/mask_im_0086.png")
#mask = load_single_mask("~/projects/dataset/dtc/test/mask_im_0086.png", index=14) 
# run model
#output = inference(image, mask, seed=42)

# export gaussian splat
#output["gs"].save_ply(f"splat.ply")
#EOF
python demo.py
echo "Done with Sam3d demo processing, output saved in splat.ply"
