import json
from functools import partial
from typing import Union, Optional, Dict

import lightning as L
import numpy as np
import pytorch3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
from pcdet.datasets.processor.data_processor import DataProcessor
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes, join_meshes_as_batch, join_meshes_as_scene
from raytorch.LiDAR import LiDAR_base
from voxel_ops import pcdet_registry

from . import eval_utils
from .base import adversarial_patch_3d


class physical_adversary:
    def __init__(self, adversary_config: EasyDict):
        self.config = adversary_config
        self.car_adv_patch_scale = self.config.car_adv_patch_scale
        self.car_adv_patch_level = self.config.car_adv_patch_level

        self.benchmark = None
        if self.config.benchmark_path is not None:
            with open(self.config.benchmark_path) as file:
                self.benchmark = json.load(file)

        self.adversarial_patch = None
        self.rooftop_approximate = None
        self.lidar = None
        self.enbaled_adversary = True

    def enable_adversary(self, adversary: bool = True):
        self.enbaled_adversary = adversary

    def configure_adversary(
        self, adversarial_patch: adversarial_patch_3d, lidar: Optional[LiDAR_base] = None
    ):
        self.adversarial_patch = adversarial_patch
        self.lidar = lidar

    def evaluate_adversary(self, result_dict, dataset: str, log_dir: str):
        dataset_eval: eval_utils.dataset_evaluation = eval_utils.__all__[dataset]
        asr_str, asr_dict = dataset_eval.evaluate_ASR(
            self.benchmark, result_dict, log_dir
        )
        asr_dict = {f"ASR/{k}": v for k, v in asr_dict.items()}

        return asr_str, asr_dict

    @staticmethod
    def evaluate_summary(result_dict, dataset: str, log_dir: str):
        dataset_eval: eval_utils.dataset_evaluation = eval_utils.__all__[dataset]
        res_str, res_dict = dataset_eval.parse_raw_result(result_dict, log_dir)

        return res_str, res_dict

    def check_tensor(self, input: Union[np.ndarray, torch.Tensor]):
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input).to(self.adversarial_patch.device)

        return input

    def collate_gtboxes(self, data_dict: Optional[dict] = None, config: Optional[EasyDict] = None):
        if data_dict is None:
            return partial(self.collate_gtboxes, config=config)

        if not isinstance(config.target_label, torch.Tensor):
            config.target_label = torch.tensor(config.target_label).to(
                self.adversarial_patch.device
            )

        if self.enbaled_adversary:

            data_dict["gt_boxes"] = self.check_tensor(data_dict["gt_boxes"])

            # Filter ground truth boxes by label
            mask = torch.isin(data_dict["gt_boxes"][:, -1], config.target_label)
            data_dict["gt_boxes"] = data_dict["gt_boxes"][mask]
            data_dict["gt_boxes"] = data_dict["gt_boxes"].detach().cpu().numpy()

            # We already filtered the ground truth boxes in dataset infos loading
            # Filter ground truth boxes by min points
            # if config.min_points is not None:
            #     point_masks = roiaware_pool3d_utils.points_in_boxes_gpu(
            #             data_dict["points"][None, :, 0:3], data_dict["gt_boxes"][None, :, :7]
            #         )
            #     point_masks.squeeze_(0)

            #     new_gt_boxes = []
            #     for n_mask in range(point_masks.size(0)):
            #         num = (point_masks == n_mask).sum()
            #         if num > config.min_points:
            #             new_gt_boxes.append(data_dict["gt_boxes"][n_mask])
            #     if new_gt_boxes.__len__() > 0:
            #         data_dict["gt_boxes"] = torch.stack(new_gt_boxes)

        return data_dict

    def physical_adversary(self, data_dict: Optional[dict] = None, config: Optional[EasyDict] = None):
        if data_dict is None:
            return partial(self.physical_adversary, config=config)

        if self.enbaled_adversary:

            data_dict["gt_boxes"] = self.check_tensor(data_dict["gt_boxes"])
            data_dict["points"] = self.check_tensor(data_dict["points"])
            if data_dict["gt_boxes"].size(1) == 8:
                pos, lwh, theta, label = torch.split(
                    data_dict["gt_boxes"].clone(), [3, 3, 1, 1], dim=1
                )
            elif data_dict["gt_boxes"].size(1) == 10:  # includes velocity
                pos, lwh, theta, vel, label = torch.split(
                    data_dict["gt_boxes"].clone(), [3, 3, 1, 2, 1], dim=1
                )
            else:
                raise NotImplementedError
            pos[:, 2] += lwh[:, 2] / 2.0 # move coordinate to the rooftop
            data_dict["points"] = self.attach_adv_patch_scene_car_aux(
                data_dict["points"],
                self.adversarial_patch,
                pos,
                theta,
            )
            data_dict["gt_boxes"] = data_dict["gt_boxes"].detach().cpu().numpy()

        return data_dict

    # def fast_intensity(points,
    #                lidar_origin:torch.Tensor,
    #                alpha=1e-4,
    #                gain=1.0,
    #                sigma=0.02):
    #     r = np.linalg.norm(points - lidar_origin, axis=1)

    #     I = gain * np.exp(-alpha * (r ** 2))

    #     I += np.random.normal(0, sigma, size=I.shape)

    #     I = torch.clip(I, 0.0, 1.0)
    #     return I

    # @staticmethod
    # @torch.no_grad
    # def fast_intensity_torch(
    #     points: torch.Tensor,
    #     lidar_origin: Optional[torch.Tensor] = None,
    #     alpha: float = 1e-4,
    #     gain: float = 1.0,
    #     sigma: float = 0.02,
    # ) -> torch.Tensor:
    #     """
    #     Fast approximate LiDAR intensity (0~1)

    #     Parameters
    #     ----------
    #     points : Tensor [N,3]
    #         Point cloud coordinates
    #     lidar_origin : Tensor [3], optional
    #         LiDAR origin, default is [0,0,0]
    #     alpha : float
    #         Distance decay coefficient (default 1e-4)
    #     gain : float
    #         Intensity gain (default 1.0)
    #     sigma : float
    #         Gaussian noise standard deviation (default 0.02)

    #     Returns
    #     -------
    #     intensity : Tensor [N]
    #         Intensity values in range [0,1]
    #     """

    #     if lidar_origin is None:
    #         lidar_origin = torch.zeros(3, device=points.device, dtype=points.dtype)

    #     # compute distance from LiDAR origin
    #     vecs = points - lidar_origin
    #     r = torch.norm(vecs, dim=1)

    #     # distance decay
    #     intensity = gain * torch.exp(-alpha * r**2)

    #     # add Gaussian noise
    #     noise = torch.randn_like(intensity) * sigma
    #     intensity = intensity + noise

    #     # clamp intensity to [0,1]
    #     intensity = torch.clamp(intensity, 0.0, 1.0)

    #     return intensity

    def attach_adv_patch_scene_car_aux(
        self,
        points: torch.Tensor,
        adv_patch: adversarial_patch_3d,
        pos: torch.Tensor,  # [N, 3]
        theta: torch.Tensor,  # [N, 1]
        sample_amount=50,  # when lidar is None
        adversarial_parameters: Optional[Dict[str, torch.Tensor]] =None,
    ):
        pts_set = [points]
        meshes_batch = []
        n = pos.size(0)
        for i in range(n):
            transformed_mesh = adv_patch.get_transformed_meshes(
                pos[i : i + 1], theta[i], adversarial_parameters
            )
            meshes_batch.append(transformed_mesh)

        if meshes_batch.__len__() != 0:
            meshes_batch = join_meshes_as_batch(meshes_batch)

            if self.lidar is not None:
                extend_pts = self.lidar.scan_triangles(
                    meshes_batch, compute_intensity=True
                )
            else:
                extend_pts = sample_points_from_meshes(meshes_batch, sample_amount * n)

                # extend_pts = F.pad(extend_pts,  (0, 1), "constant", 1.0)
            # reflectness = torch.ones(extend_pts.shape[0], 1).to(
            #     extend_pts.device
            # )  # (N, 1)
            # reflectness = torch.rand(extend_pts.shape[0], 1).to(
            #     extend_pts.device
            # )  # (N, 1)

            if points.size(1) == 3:  # no intensity
                # reflectivity : [0, 1]
                reflectness = torch.rand(extend_pts.shape[0], 1).to(
                    extend_pts.device
                )  # (N, 1)
                extend_pts = torch.cat([extend_pts, reflectness], dim=1)
                # print(points[:, 3].mean())

            if points.size(1) == 5:  # include timestamp
                timestamp = points[0, 4]
                extend_pts = F.pad(extend_pts, (0, 1), "constant", timestamp)

            pts_set.append(extend_pts)

        return torch.concatenate(pts_set)
