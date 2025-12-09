import numpy as np
import torch

from pytorch3d.ops import sample_points_from_meshes, laplacian
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.utils import ico_sphere
from pytorch3d.transforms import (
    euler_angles_to_matrix,
    Rotate,
    RotateAxisAngle,
    Scale,
    Transform3d,
    Translate,
)

from .base import adversarial_patch_3d
from .primitive import learnable_sphere

from typing import Optional, Union


class single_sphere(adversarial_patch_3d):
    def __init__(self, semi_scale: Union[list, np.ndarray, torch.Tensor] = [0.7, 0.7, 0.5], level: int = 2, eps=0.01, device:torch.device=None):
        super().__init__(device)
        if isinstance(semi_scale, list): 
            semi_scale = torch.tensor(semi_scale, device=device)
        elif isinstance(semi_scale, np.ndarray):
            semi_scale = torch.from_numpy(semi_scale).to(device)
            
        self.sphere: learnable_sphere = learnable_sphere(
            semi_scale=semi_scale, level=level, eps=eps, device=device
        )
        self.base_coord = torch.tensor([[0.0, 0.0, -self.sphere.basic_mesh.verts_packed()[:, 2].amin()]]
                                       , device=device)
        self.base_tranlate = Translate(self.base_coord)
        

    def get_regularization_loss(self):
        deformed_mesh = self.get_deformed_meshes()
        return mesh_laplacian_smoothing(deformed_mesh)

    def get_parameters(self) -> dict[torch.Tensor]:
        parameters: dict[torch.Tensor] = super().get_parameters()
        parameters.update(self.sphere.get_parameters())
        return parameters

    def load_parameter(self, parameters: list):
        super().load_parameter(parameters)
        self.sphere.load_parameters(parameters)

    def get_basic_meshes(self) -> Meshes:
        return self.sphere.get_basic_meshes()

    def get_deformed_meshes(
        self,
        deform_vert_logit: Optional[torch.Tensor] = None,
    ) -> Meshes:
        return self.sphere.get_deformed_meshes(deform_vert_logit=deform_vert_logit)

    def get_transformed_meshes(
        self,
        pos: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        adversarial_parameters: Optional[torch.Tensor] = None,
    ) -> Meshes:
        """
        Args:
            pos: [B, 3]
            theta: [B,]
        """

        # deform_vert_logit = None
        # global_translation = self.global_translation
        # global_theta = self.theta

        if adversarial_parameters is not None:
            raise NotImplementedError
            global_translation = adversarial_parameters[0]
            global_theta = adversarial_parameters[1]
            deform_vert_logit = adversarial_parameters[2]

        deformed_mesh = self.get_deformed_meshes()
        verts = deformed_mesh.verts_padded()

        # global transformation
        T, R = self.encode_parameters(self._T, self._theta)
        verts = self.base_tranlate.transform_points(verts)
        verts = R.transform_points(verts)
        verts = T.transform_points(verts)

        # translate to the rooftop of the vehicle
        if pos is not None and theta is not None:
            local_R = RotateAxisAngle(theta, "Z", degrees=False).to(verts.device)
            local_T = Translate(pos).to(verts.device)
            verts = local_R.transform_points(verts)
            verts = local_T.transform_points(verts)
            
        transformed_mesh = deformed_mesh.update_padded(verts)
        return transformed_mesh

    def constrain_grad(self):
        super().constrain_grad(allow_rotate=False)
        if self.sphere.deform_vert_logit.grad is not None:
            self.sphere.deform_vert_logit.grad[:, 2] = 0.0