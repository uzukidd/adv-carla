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
from .primitive import learnable_sphere, learnable_sphere_legacy

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
        deform_vert_logit: torch.Tensor = None,
    ) -> Meshes:
        return self.sphere.get_deformed_meshes(deform_vert_logit=deform_vert_logit)

    def get_transformed_meshes(
        self,
        pos: torch.Tensor = None,
        theta: torch.Tensor = None,
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
        super().constrain_grad()
        if self.sphere.deform_vert_logit.grad is not None:
            self.sphere.deform_vert_logit.grad[:, 2] = 0.0


class single_sphere_legacy(adversarial_patch_3d):
    def __init__(self, scale: list = [0.7, 0.7, 0.5], level: int = 2, eps=0.0):
        super().__init__(scale=scale)
        self.sphere = learnable_sphere_legacy(scale=self.scale, level=level, eps=eps)

    def set_training(self, enable: bool):
        super().set_training(enable)
        self.sphere.training = enable

    def get_regularization_loss(self):
        deformed_mesh = self.get_deformed_meshes()
        return mesh_laplacian_smoothing(deformed_mesh)

    def get_parameters(self) -> list[torch.Tensor]:
        parameters = super().get_parameters()
        parameters.append(self.sphere.deform_vert_logit)
        return parameters

    def load_parameter(self, parameters: list):
        super().load_parameter(parameters)
        self.sphere.deform_vert_logit = (
            parameters[2].clone().detach().to(self.sphere.deform_vert_logit.device)
        )
        self.sphere.deform_vert_logit.requires_grad_(True)

    def get_basic_meshes(self) -> Meshes:
        return self.sphere.get_basic_meshes()

    def get_deformed_meshes(
        self,
        deform_vert_logit: torch.Tensor = None,
    ) -> Meshes:
        return self.sphere.get_deformed_meshes(deform_vert_logit=deform_vert_logit)

    def get_transformed_meshes(
        self,
        pos: torch.Tensor = None,
        theta: torch.Tensor = None,
        adversarial_parameters: Optional[torch.Tensor] = None,
    ) -> Meshes:
        """
        Args:
            pos: [3,]
            theta: [1,]
        """

        deform_vert_logit = None
        global_translation = self.global_translation
        global_theta = self.theta

        if adversarial_parameters is not None:
            global_translation = adversarial_parameters[0]
            global_theta = adversarial_parameters[1]
            deform_vert_logit = adversarial_parameters[2]

        if not self.training:
            global_translation = global_translation.detach()
            global_theta = global_theta.detach()

        deformed_mesh = self.get_deformed_meshes(deform_vert_logit=deform_vert_logit)
        verts = deformed_mesh.verts_padded()
        verts = (
            verts
            + (self.offset_limit * torch.tanh(global_translation / self.offset_limit))[
                None, :
            ]
        )
        global_R = self.generate_rotate_matrix(global_theta)
        verts = torch.matmul(verts, global_R.T)

        # translate to the rooftop of the vehicle
        if pos is not None and theta is not None:
            local_R = self.generate_rotate_matrix(theta)
            verts = torch.matmul(verts, local_R.T)
            verts = verts + (pos - self.base_coord)

            transformed_mesh = deformed_mesh.update_padded(verts)

        else:
            transformed_mesh = deformed_mesh

        return transformed_mesh

    def constrain_grad(self):
        super().constrain_grad()
        if self.sphere.deform_vert_logit.grad is not None:
            self.sphere.deform_vert_logit.grad[:, 2] = 0.0

    def generate_rotate_matrix(self, theta: torch.Tensor) -> torch.Tensor:
        tensor_0 = torch.zeros(1).cuda()
        RZ = euler_angles_to_matrix(
            torch.concatenate([tensor_0, tensor_0, theta]), ["X", "Y", "Z"]
        )

        return RZ


class simple_cubic_lattice:
    def __init__(self, cubic_level: int = 2, scale: list = [0.7, 0.7, 0.5], eps=0.0):
        self.internal_atom: list[learnable_sphere] = None
        self.scale = np.array(scale)
        self.cubic_level = cubic_level

        self.offset_limit: torch.Tensor = torch.tensor([0.1]).cuda()
        self.global_translation: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]).cuda()
        self.global_translation.requires_grad_(True)

        self.base_coord: torch.Tensor = (
            torch.tensor([0.0, 0.0, -scale[2]]).float().cuda()
        )

        self.theta: torch.Tensor = torch.tensor([0.0]).cuda()
        self.theta.requires_grad_(True)

        self._init_internal_atoms()
        self.lattice_grid = torch.stack(
            [grid_pos.contiguous().view(-1) for grid_pos in self.generate_mesh_grid()],
            dim=1,
        )

    def _init_internal_atoms(self):
        dscale = self.scale / self.cubic_level
        self.internal_atoms = []
        for _ in range(self.cubic_level**3):
            self.internal_atoms.append(learnable_sphere(level=2, scale=dscale))

    def get_laplacian_loss(self):
        deformed_mesh = self.get_deformed_lattice()
        return mesh_laplacian_smoothing(deformed_mesh)

    def get_parameters(self):
        parameters = [self.global_translation, self.theta]
        for internal_atom in self.internal_atoms:
            parameters.append(internal_atom.deform_vert_logit)

        return parameters

    def load_parameter(self, parameters: list):
        self.global_translation = parameters[0]
        self.theta = parameters[1]
        for i, internal_atom in enumerate(self.internal_atoms):
            internal_atom.deform_vert_logit = parameters[2 + i]

    def generate_mesh_grid(self):
        dscale = self.scale / self.cubic_level
        x = torch.arange(
            -self.scale[0] + dscale[0], self.scale[0], dscale[0] * 2.0
        ).cuda()
        y = torch.arange(
            -self.scale[1] + dscale[1], self.scale[1], dscale[1] * 2.0
        ).cuda()
        z = torch.arange(
            -self.scale[2] + dscale[2], self.scale[2], dscale[2] * 2.0
        ).cuda()

        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
        return grid_x, grid_y, grid_z

    def get_deformed_lattice(self) -> Meshes:
        atoms_meshes = []
        for i in range(self.lattice_grid.size(0)):
            atoms_meshes.append(
                self.internal_atoms[i].get_transformed_meshes(self.lattice_grid[i])
            )
            # atoms_meshes.append(self.internal_atoms[i].get_deformed_meshes())
        atoms_meshes = join_meshes_as_batch(atoms_meshes)
        return atoms_meshes

    def get_transformed_meshes(self, pos: torch.Tensor, theta: torch.Tensor) -> Meshes:
        deformed_mesh = self.get_deformed_lattice()
        verts = deformed_mesh.verts_padded()
        verts = (
            verts
            + (
                self.offset_limit
                * torch.tanh(self.global_translation / self.offset_limit)
            )[None, :]
        )
        global_R = self.generate_rotate_matrix(self.theta)
        verts = torch.matmul(verts, global_R.T)

        # translate to the rooftop of the vehicle
        local_R = self.generate_rotate_matrix(theta)
        verts = torch.matmul(verts, local_R.T)
        verts = verts + (pos - self.base_coord)

        transformed_mesh = deformed_mesh.update_padded(verts)

        return transformed_mesh

    def constrain_z_grad(self):
        self.global_translation.grad[2] = 0.0
        for internal_atom in self.internal_atoms:
            internal_atom.deform_vert_logit.grad[:, 2] = 0.0

    def generate_rotate_matrix(self, theta: torch.Tensor) -> torch.Tensor:
        tensor_0 = torch.zeros(1).cuda()
        RZ = euler_angles_to_matrix(
            torch.concatenate([tensor_0, tensor_0, theta]), ["X", "Y", "Z"]
        )

        return RZ
