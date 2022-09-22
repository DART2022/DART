import os
import csv
import numpy as np
import torch
from pytorch3d.io import load_objs_as_meshes, load_obj, save_obj
import matplotlib.pyplot as plt
# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)

DATA_DIR = r"./data"


def modify_vertices(verts):
    # Set paths
    obj_filename = os.path.join(DATA_DIR, r"hand_02.obj")
    out_filename = os.path.join(DATA_DIR, r"test_hand.obj")

    with open(obj_filename) as fin, open(out_filename, 'w+') as fout:
        cnt = 0
        read_vert = False
        for line in fin.readlines():
            if line.startswith("v "):
                cnt += 1
                read_vert = True
            else:
                if read_vert:
                    read_vert = False
                    for v in verts:
                        fout.write("v {:.6f} {:.6f} {:.6f}\n".format(v[0], v[1], v[2]))
                fout.write(line)
        print(cnt)


def render_mesh_image():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Set paths
    obj_filename = os.path.join(DATA_DIR, "test_hand.obj")
    # obj_filename = os.path.join(DATA_DIR, "hand_wrist/hand_02.obj")

    # Load obj file
    mesh = load_objs_as_meshes([obj_filename], device=device)

    # Initialize a camera.
    # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction.
    # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow.
    R, T = look_at_view_transform(1.0, 0, 0)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
    # the difference between naive and coarse-to-fine rasterization.
    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Place a point light in front of the object. As mentioned above, the front of the cow is facing the
    # -z direction.
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and
    # apply the Phong lighting model
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )
    images = renderer(mesh)
    plt.figure(figsize=(10, 10))
    plt.imshow(images[0, ..., :3].cpu().numpy())
    plt.show()
    # plt.axis("off")


if __name__ == "__main__":
    render_mesh_image()
