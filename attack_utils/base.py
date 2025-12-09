import torch
import torch.nn as nn
import numpy as np

import pytorch3d
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

from abc import ABC, abstractmethod
from typing import List, Dict


class adversarial_patch_3d(nn.Module, ABC):
    """
    3D adversarial patch
        pipeline:
            base meshes + deforming parameters (deforming + translate + rotate)->
            deformed meshes + transforming parameters (position + rotation) ->
            transformed meshes (final meshes)
        parameter:
            _T, _theta
    """

    def __init__(self, device:torch.device):
        super().__init__()
        self.device = device
        # global transforming parameters
        self._b: torch.Tensor = torch.tensor([[0.1]]).to(device)
        self._T: torch.Tensor = nn.Parameter(torch.tensor([[0.0, 0.0, 0.0]]).to(device), True)

        self._theta: torch.Tensor = nn.Parameter(torch.tensor([0.0]).to(device), True)

    def encode_parameters(self, T, theta):
        return Translate(self._b * torch.tanh(T)), RotateAxisAngle(
            theta, "Z", degrees=False
        )

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        parameters = {"_T": self._T, "_theta": self._theta}
        return parameters

    def load_parameter(self, parameters: Dict[str, torch.Tensor]) -> None:
        self._theta = parameters["_theta"]
        self._T = parameters["_T"]

    @abstractmethod
    def get_basic_meshes(self) -> Meshes:
        ...

    @abstractmethod
    def get_deformed_meshes(self) -> Meshes:
        ...

    @abstractmethod
    def get_transformed_meshes(
        self,
        pos: torch.Tensor,
        theta: torch.Tensor,
        adversarial_parameters: list[torch.Tensor],
    ) -> Meshes:
        ...

    @abstractmethod
    def get_regularization_loss(self) -> torch.Tensor:
        ...

    def constrain_grad(self, allow_rotate: bool = False):
        if self._T.grad is not None:
            self._T.grad[:, 2] = 0.0

        if not allow_rotate:
            self._theta.grad = None
