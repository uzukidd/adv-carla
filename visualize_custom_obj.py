from pytorch3d.io import load_obj

from pytorch3d.structures.meshes import Meshes

import torch
import trimesh
import meshio

# f = load_obj("/home/chw/Documents/sdf/out.obj")

verts, faces_all, file_property = load_obj("image.obj")
verts, faces = verts, faces_all[0]

image_mesh = Meshes(verts=[verts], faces=[faces])

# stl_mesh = meshio.read('out.stl')
mesh_obj = meshio.read('image.obj')

print(mesh_obj.points, type(mesh_obj.points), max(mesh_obj.points[:, 0]), min(mesh_obj.points[:, 0]))
print(mesh_obj.points.shape)

# visualize the mesh
plotly_mesh = trimesh.Trimesh(
    verts.detach().cpu().numpy(),
    faces.detach().cpu().numpy(),
    vertex_colors=[100, 100, 255],
)

plotly_mesh.show()

def rotate_points(points, angle):
    # Rotate 3D points around the z-axis
    cos_theta = torch.cos(angle)
    sin_theta = torch.sin(angle)

    rotation_matrix = torch.tensor([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ], dtype=points.dtype, device=points.device)

    rotated_points = torch.matmul(points, rotation_matrix)
    return rotated_points

rotate_verts = rotate_points(verts, torch.Tensor([0.5]))

print(rotate_verts)


plotly_mesh = trimesh.Trimesh(
    rotate_verts.detach().cpu().numpy(),
    faces.detach().cpu().numpy(),
    vertex_colors=[100, 100, 255],
)

# TODO: What coordinate system is this? Z-axis is forward out the screen?

plotly_mesh.show()


