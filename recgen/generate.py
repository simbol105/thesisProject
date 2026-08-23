import os
os.environ["ATTN_BACKEND"] = 'xformers'


from PIL import Image
import numpy as np


from recgen_inference import build_recgen, generate

SIZE = 1024

def load_image(path):
    image = Image.open(path)
    image = image.resize((SIZE, SIZE), Image.NEAREST)  # resize correctly
    image = np.array(image).astype(np.uint8)
    return image


pipeline = build_recgen.build("recgen_base.multiview_stereo")

rgb = load_image("/home/fr/fr_fr/fr_ba122/projects/dataset/DTC/Cup_B08TWHJ33Q_Tan_Lighting001/train/im_0071.png")
depth = load_image("/home/fr/fr_fr/fr_ba122/projects/dataset/DTC/Cup_B08TWHJ33Q_Tan_Lighting001/depth/im_0070_depth.png")
mask = load_image("/home/fr/fr_fr/fr_ba122/projects/dataset/DTC/Cup_B08TWHJ33Q_Tan_Lighting001/mask/im_0070.png")

rgb = rgb[:, :, :3]
mask = mask[:,:,  :1].reshape(SIZE, SIZE)
# convert depth to float32
depth = depth.astype(np.float32) / 1000.0 
print(rgb.shape, depth.shape, mask.shape )
fx = 4016.6871096
fy = 4016.6871096
cx = 1000.0
cy = 1000.0

K = np.array(
    [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32,
)
result = generate(
    pipeline, 
    image=rgb,
    depth=depth,
    mask=mask,
    intrinsics=K,
)

print(result.pose_matrix)           # (4, 4) float64 pose in the camera frame
print(result.pose_quat)              # (7,) [tx, ty, tz, qx, qy, qz, qw]
print(result.mesh.vertices.shape)    # final textured mesh

result.save("./out", save_splat=True, save_glb=False)
