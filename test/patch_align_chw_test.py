from pytorch3d.utils import ico_sphere
import trimesh
import torch

import math

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

device = torch.device("cuda:0")

# Initialize an ico_sphere mesh and set its device to cuda:0
sphere_mesh = ico_sphere(0, device)

# Get the verts and faces of the mesh
verts = sphere_mesh.verts_packed()
faces = sphere_mesh.faces_packed()

print(verts)

verts_rotated = rotate_points(verts[:, :3], torch.tensor(math.pi / 4))

print(verts_rotated)

# visualize the mesh
plotly_mesh = trimesh.Trimesh(
    verts.detach().cpu().numpy(),
    faces.detach().cpu().numpy(),
    vertex_colors=[100, 100, 255],
)

plotly_mesh.show()

plotly_mesh_rotated = trimesh.Trimesh(
    verts_rotated.detach().cpu().numpy(),
    faces.detach().cpu().numpy(),
    vertex_colors=[100, 100, 255],
)

plotly_mesh_rotated.show()


### Verify that the distance from the origin is preserved after rotation

# 计算每个顶点到原点的距离
dist_before = torch.norm(verts[:, :3], dim=1)
dist_after = torch.norm(verts_rotated[:, :3], dim=1)

# 打印距离
print("Distance before rotation:", dist_before)
print("Distance after rotation:", dist_after)
