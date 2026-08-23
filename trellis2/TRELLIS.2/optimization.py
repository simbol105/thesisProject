import os 
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Can save GPU memory
# os.environ["ATTN_BACKEND"] = 'xformers' # I couldn't get flash attention to work here. 
import sys 
sys.path.append("/home/simbol/Documents/thesis/thesis/projects/trellis/TRELLIS.2/")  
import cv2
import imageio
from PIL import Image
import torch
import torch.nn as nn
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel
import time 
import yaml
from tqdm import tqdm
from  trellis2.representations import Mesh, MeshWithVoxel
from trellis2.renderers.mesh_renderer import MeshRenderer
from trellis2.renderers.pbr_mesh_renderer import PbrMeshRenderer
import json
from pathlib import Path
from typing import List
import numpy as np

'''
TODO: Steps in my optimization loop
1. Use a random image from the dataset and run the trellis pipeline on it to get the latent representation 
2. Decode the latent to mesh, render the mesh to image using the same camera parameters as the reference image, 
3. Compute the image loss between the rendered image and the reference image, like nvdiffrec loss
4. Backpropagate to update the latent representation. 
5. Repeat 2-4 until convergence. 
'''
'''
config should contain, path to the train dataset with configuration similar to nvdiffrecmc, 
and 
yaml config {
    "lr": 1e-3,
    "num_iters": 500,
    "envmap_path": "path/to/envmap.exr"
    }
'''
def optimize(config_path: Path, config_yaml: dict)-> List[MeshWithVoxel]: 
    assert config_path.exists(), f"Config file {config_path} does not exist."
    with open(config_path, 'r') as f:
        json_config = json.load(f)
    
    # yaml_config = yaml.safe_load(config_yaml)
    with open(config_yaml, "r") as f:
        yaml_config = yaml.safe_load(f)
    path_dataset = Path(config_path).parent / "train"
    assert path_dataset.exists(), f"Dataset path {path_dataset} does not exist."

    # load a random image from the dataset to use for extracting latent 
    image_paths = list(path_dataset.glob("*.png"))
    camera_matrices = json_config["frames"]
    intr = json_config["intrinsic"]
   
    intrinsics = torch.tensor([
        [intr['fx'], 0, intr['cx']],
        [0, intr['fy'], intr['cy']],
        [0, 0, 1]
    ], dtype=torch.float32).cuda()
    
    assert len(image_paths) > 0, f"No images found in dataset path {image_paths}."
    N = len(image_paths)
    input_image = Image.open(image_paths[torch.randint(0, N, (1,)).item()])
    # load the pipeline
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()

    _, (shape_slat, tex_slat, res) = pipeline.run(input_image, return_latent=True, 
                                        pipeline_type='512')
    # convert the latents to trainable parameters 
    shape_feats = nn.Parameter(shape_slat.feats.clone())
    tex_feats   = nn.Parameter(tex_slat.feats.clone())
    assert Path(yaml_config['envmap_path']).exists(), f"Envmap path {yaml_config['envmap_path']} doesn't exist!" 
    env_map = EnvMap(torch.tensor(
        cv2.cvtColor(cv2.imread(Path(yaml_config['envmap_path']), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB), 
        dtype=torch.float32, device='cuda'
    ))
    render_options = {
        "resolution": res,
        "near": 0.01,
        "far": 100.0
    }
    
    lr = float(yaml_config['lr'])
    optimizer = torch.optim.Adam([shape_feats, tex_feats], lr=lr)
    num_iters = int(yaml_config['num_iters'])
    outer = tqdm(range(num_iters), desc="Iterations")
    for iter in outer:
        
        total_loss = 0.0
        
        shape_feats = shape_feats.detach().requires_grad_(True)
        tex_feats   = tex_feats.detach().requires_grad_(True)
        
        optimizer.zero_grad()
        indices = np.random.uniform(low=1, high=N, size=N).astype(int)
        for i in tqdm(indices, desc="Frames", leave=False):
            renderer = PbrMeshRenderer(device="cuda", rendering_options=render_options)

            # shape_slat_opt = shape_slat.replace(feats=shape_feats)
            # tex_slat_opt   = tex_slat.replace(feats=tex_feats)
            
            shape_slat_opt = shape_slat.replace(
                feats=shape_feats,
                coords=shape_slat.coords.detach()
            )

            tex_slat_opt = tex_slat.replace(
                feats=tex_feats,
                coords=tex_slat.coords.detach()
            )

            with torch.no_grad():
                mesh_list = pipeline.decode_latent(
                    shape_slat_opt,
                    tex_slat_opt,
                    res
                )
            mesh = mesh_list[0]

            extr = torch.tensor(camera_matrices[i]["transform_matrix"]).float().cuda()
            intr = torch.tensor(intrinsics).float().cuda()

            buffers = renderer.render(
                mesh=mesh,
                extrinsics=extr,
                intrinsics=intr,
                envmap=env_map
            )
            rendered_image = buffers["shaded"].clone()  # [3, H, W]
            rendered_image = rendered_image.detach().requires_grad_(True)
            ref = cv2.imread(image_paths[i])[:, :, ::-1]  
            ref = cv2.resize(ref, (res, res))
            ref = torch.tensor(ref).float().cuda() / 255.0
            ref = ref.permute(2, 0, 1).detach()  # [3, H, W]

            loss = torch.nn.functional.mse_loss(rendered_image, ref)
            # print("Rendered requires_grad:", rendered_image.requires_grad)

            #total_loss += loss
            loss.backward()
           
           
       
        # total_loss.backward() 
        
            optimizer.step()

        # outer.set_postfix(loss=total_loss.item() / N)
    
    # decode the final mesh and the latents and return 
    final_shape_slat = shape_slat.replace(feats=shape_feats.detach())
    final_tex_slat   = tex_slat.replace(feats=tex_feats.detach())
    final_mesh = pipeline.decode_latent(final_shape_slat, final_tex_slat, res)[0]
    return final_mesh, final_shape_slat, final_tex_slat


if __name__ == "__main__":
    config_path = Path("/home/fr/fr_fr/fr_ba122/projects/dataset/DTC/Birdhouse_Lighting001/transforms_train.json")
    config_yaml = "yaml_config/config.yaml"
    envmap_path = "/home/fr/fr_fr/fr_ba122/projects/dataset/DTC/Birdhouse_Lighting001/env.exr"
    env_map = EnvMap(torch.tensor(
        cv2.cvtColor(cv2.imread(envmap_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB), 
        dtype=torch.float32, device='cuda'
    ))
    mesh, _, _  = optimize(config_path=config_path, config_yaml = config_yaml)
    #final_mesh.simplify(16777216) # nvdiffrast limit
    video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh,
                                                                        envmap=env_map))
    imageio.mimsave("optimization_result_20.mp4", video, fps=15)
    
    # Export the final result to .glb
    glb = o_voxel.postprocess.to_glb(
    vertices            =   mesh.vertices,
    faces               =   mesh.faces,
    attr_volume         =   mesh.attrs,
    coords              =   mesh.coords,
    attr_layout         =   mesh.layout,
    voxel_size          =   mesh.voxel_size,
    aabb                =   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    decimation_target   =   1000000,
    #texture_size        =   4096,
    texture_size        =   1024,
    remesh              =   True,
    remesh_band         =   1,
    remesh_project      =   0,
    verbose             =   True
    )
    glb.export("optimization_result_20.glb", extension_webp=False)

    