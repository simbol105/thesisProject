# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys
import numpy as np
import time 

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask,  display_image, make_scene, ready_gaussian_for_video_rendering, render_video

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

# load image (RGBA only, mask is embedded in the alpha channel)
# image = load_image("notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")
# mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)
# image = load_image("/home/fr/fr_fr/fr_ba122/projects/dataset/DTC/Birdhouse_Lighting001/train/im_0005.png")
image = load_image("/home/fr/fr_fr/fr_ba122/projects/dataset/DTC/Cup_B08TWHJ33Q_Tan_Lighting001/train/im_0005.png")

# run model
mask = image[:, :, 3] > 0 
start_time = time.time()
pipeline = inference._pipeline 
output = pipeline.run(image, 
                #    mask,  
                None,
                   seed=42,
                   with_mesh_postprocess=True,  # default  
                   with_texture_baking=True)
sam3d_runtime = time.time() - start_time

with open('sam3d_runtime.txt', 'a') as f: 
    f.write(f"Runtime for sam3d on DTC image: {sam3d_runtime:.2f} seconds\n")
# export gaussian splat
# output["gs"].save_ply(f"dtc_splat.ply")
mesh_glb = output["glb"] 
mesh_glb.export("sam3d_cup_mesh.glb")


scene_gs = make_scene(output)
scene_gs = ready_gaussian_for_video_rendering(scene_gs)

video = render_video(
    scene_gs,
    r=1,
    fov=60,
    pitch_deg=15,
    yaw_start_deg=-45,
    resolution=512,
)["color"]

# save video as gif
import imageio

imageio.mimsave(
    "dtc_cup_splat.gif",
    video,
    format="GIF",
    duration=1000 / 30,  
    loop=0,  
)

print("Your reconstruction has been saved to dtc_cup_splat.ply")
