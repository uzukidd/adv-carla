from collections import defaultdict
from functools import partial
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict
from pcdet.datasets.processor.data_processor import DataProcessor
from pcdet.utils import common_utils


def resgister_data_processor(name: str, module: callable):
    setattr(DataProcessor, name, module)


def constant_reflectness(
    data_processor: DataProcessor, data_dict: dict = None, config: EasyDict = None
):
    if data_dict is None:
        return partial(data_processor.constant_reflectness, config=config)

    def _check_tensor(input: Union[np.ndarray, torch.Tensor]):
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input).cuda()

        return input

    data_dict["points"]: torch.Tensor = _check_tensor(data_dict["points"])  # type: ignore

    cosntant_value: float = config["constant"]

    reflectness = data_dict["points"].new_full(
        data_dict["points"].size(), cosntant_value
    )
    mask = data_dict["points"].new_zeros(data_dict["points"].size())
    mask[:, 3] = 1.0

    data_dict["points"] = data_dict["points"] * (1 - mask) + reflectness * mask

    return data_dict


resgister_data_processor("constant_reflectness", constant_reflectness)


def sample_points_aux(
    data_processor: DataProcessor, data_dict: dict = None, config: EasyDict = None
):
    if data_dict is None:
        return partial(sample_points_aux, data_processor=data_processor, config=config)

    def _check_tensor(input: Union[np.ndarray, torch.Tensor]):
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input).cuda()

        return input

    num_points = config.NUM_POINTS[data_processor.mode]
    if num_points == -1:
        return data_dict

    data_dict["points"]: torch.Tensor = _check_tensor(data_dict["points"])  # type: ignore

    points = data_dict["points"]
    if num_points < points.size(0):
        pts_depth = torch.linalg.norm(points[:, 0:3], dim=1)
        pts_near_flag = pts_depth < 40.0
        far_idxs_choice = torch.where(~pts_near_flag)[0]
        near_idxs = torch.where(pts_near_flag)[0]
        choice = []
        if num_points > far_idxs_choice.size(0):
            near_idxs_choice = torch.randperm(near_idxs.size(0))[
                : num_points - far_idxs_choice.size(0)
            ]
            near_idxs_choice = near_idxs[near_idxs_choice]

            # near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
            choice = (
                torch.concatenate((near_idxs_choice, far_idxs_choice), dim=0)
                if far_idxs_choice.size(0) > 0
                else near_idxs_choice
            )
            choice = choice[torch.randperm(choice.size(0))]
        else:
            # choice = np.arange(0, len(points), dtype=np.int32)
            # choice = np.random.choice(choice, num_points, replace=False)
            choice = torch.randperm(points.size(0))[:num_points]

    else:
        choice = torch.arange(0, points.size(0))
        if num_points > points.size(0):
            # extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
            extra_choice = torch.randperm(points.size(0))[: num_points - points.size(0)]
            choice = torch.concatenate((choice, extra_choice), dim=0)
        # np.random.shuffle(choice)
        choice = choice[torch.randperm(choice.size(0))]

    data_dict["points"] = points[choice]
    return data_dict


resgister_data_processor("sample_points_aux", sample_points_aux)
