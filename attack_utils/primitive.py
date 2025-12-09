import torch
import torch.nn as nn
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes, laplacian
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere

from typing import Optional, Dict


# class learnable_cube:
#     CUBE_VERTEICE = torch.tensor(
#         [
#             [-1, -1, -1],  # 0
#             [-1, -1, 1],  # 1
#             [-1, 1, -1],  # 2
#             [-1, 1, 1],  # 3
#             [1, -1, -1],  # 4
#             [1, -1, 1],  # 5
#             [1, 1, -1],  # 6
#             [1, 1, 1],  # 7
#         ],
#         dtype=torch.float32,
#     )

#     CUBE_FACES = torch.tensor(
#         [
#             [0, 1, 2],
#             [1, 3, 2],
#             [4, 6, 5],
#             [5, 6, 7],
#             [0, 4, 1],
#             [1, 4, 5],
#             [2, 3, 6],
#             [3, 7, 6], 
#             [0, 2, 4],
#             [2, 6, 4], 
#             [1, 5, 3],
#             [3, 5, 7],
#         ],
#         dtype=torch.int64,
#     )

#     def __init__(self) -> None:
#         pass


# class learnable_sphere_legacy:
#     def __init__(self, scale: torch.Tensor, level: int = 2, eps: float = 0.0):
#         self.scale: torch.Tensor = torch.tensor(scale).float().cuda()
#         self.semi_scale: torch.Tensor = self.scale / 2
#         self.basic_mesh = self.generate_basic_mesh(level=level, scale=scale, eps=eps)
#         self.training = True

#         # Vertex deforming parameter
#         self.init_vert_quadrant: torch.Tensor = (
#             torch.sign(self.basic_mesh.verts_packed()).detach().clone()
#         )
#         self.deform_vert_logit: torch.Tensor = (
#             torch.zeros_like(self.basic_mesh.verts_packed(), device=self.init_vert_quadrant.device).contiguous()
#         )
#         self.deform_vert_logit.requires_grad_(True)

#         self.init_vert_logit: torch.Tensor = (
#             torch.logit(torch.abs(self.basic_mesh.verts_packed() / self.scale[None, :]))
#             .detach()
#             .clone()
#         )
#         print(f"mesh vertex size : {self.basic_mesh.verts_packed().size()}")

#     def get_parameters(self) -> list[torch.Tensor]:
#         return [self.deform_vert_logit]

#     def get_basic_meshes(self) -> Meshes:
#         return self.basic_mesh

#     def get_deformed_meshes(
#         self, deform_vert_logit: Optional[torch.Tensor] = None
#     ) -> Meshes:
#         basic_mesh = self.get_basic_meshes()
#         if deform_vert_logit is None:
#             deform_vert_logit = self.deform_vert_logit

#         if not self.training:
#             deform_vert_logit = deform_vert_logit.detach()

#         deformed_vert = (
#             self.scale[None, :]
#             * self.init_vert_quadrant
#             * torch.sigmoid(self.init_vert_logit + deform_vert_logit)
#         )

#         return basic_mesh.update_padded(deformed_vert.unsqueeze(0))

#     # def get_transformed_meshes(self, translate:torch.Tensor, deform_vert_logit:torch.Tensor = None):
#     #     deformed_meshes = self.get_deformed_meshes(deform_vert_logit = deform_vert_logit)
#     #     verts = deformed_meshes.verts_padded()
#     #     verts = verts + translate[None, :]

#     #     return deformed_meshes.update_padded(verts)

#     def get_base_coord(self):
#         base_z = self.basic_mesh.verts_packed()[:, 2].min()

#         return self.deform_vert.new_tensor([0.0, 0.0, base_z], requires_grad=False)

#     @staticmethod
#     def generate_basic_mesh(level: int, scale: list, eps):
#         mSphere = ico_sphere(level).cuda()

#         new_vert = mSphere.verts_padded()
#         new_vert[:, :, 0] = new_vert[:, :, 0] * scale[0] * (1 - eps)
#         new_vert[:, :, 1] = new_vert[:, :, 1] * scale[1] * (1 - eps)
#         new_vert[:, :, 2] = new_vert[:, :, 2] * scale[2] * (1 - eps)

#         mSphere = mSphere.update_padded(new_vert)
#         return mSphere


class learnable_sphere(nn.Module):
    def __init__(self, semi_scale: torch.Tensor, level: int = 2, eps: float = 0.001, device:Optional[torch.device]=None):
        super().__init__()
        self.semi_scale: torch.Tensor = semi_scale.detach().clone()

        self.basic_mesh:Meshes = self.generate_basic_mesh(
            level=level, semi_scale=semi_scale, eps=eps, device=device
        )

        # Vertex deforming parameter
        self.init_vert_quadrant: torch.Tensor = (
            torch.sign(self.basic_mesh.verts_packed()).detach().clone()
        )
        self.deform_vert_logit: torch.Tensor = nn.Parameter(
            torch.zeros_like(self.basic_mesh.verts_packed()).to(device).contiguous()
        , True)

        self.init_vert_logit: torch.Tensor = (
            torch.logit(torch.abs(self.basic_mesh.verts_packed() / self.semi_scale[None, :]))
            .detach()
            .clone()
        )
        print(f"mesh vertex count : {self.basic_mesh.verts_packed().size()}")

    def load_parameters(self, parameters: Dict[str, torch.Tensor]):
        self.deform_vert_logit = parameters["deform_vert_logit"]

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        return {"deform_vert_logit": self.deform_vert_logit}

    def get_basic_meshes(self) -> Meshes:
        return self.basic_mesh

    def get_deformed_meshes(
        self, deform_vert_logit: Optional[torch.Tensor] = None
    ) -> Meshes:
        basic_mesh = self.get_basic_meshes()
        if deform_vert_logit is None:
            deform_vert_logit = self.deform_vert_logit

        deformed_vert = (
            self.semi_scale[None, :]
            * self.init_vert_quadrant
            * torch.sigmoid(self.init_vert_logit + deform_vert_logit)
        )
        # packed parameters update to padded parameters, single-batch meshes should be OK
        return basic_mesh.update_padded(deformed_vert.unsqueeze(0))

    @staticmethod
    def generate_basic_mesh(level: int, semi_scale: list, eps, device:torch.device = None):
        mSphere = ico_sphere(level).to(device)

        new_vert = mSphere.verts_padded()
        new_vert[:, :] = new_vert[:, :] * semi_scale * (1 - eps)

        mSphere = mSphere.update_padded(new_vert)
        return mSphere
