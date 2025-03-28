import pickle
import time

from abc import ABC

import numpy as np
import torch
import tqdm

from typing import Callable, Optional, Tuple, Dict
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils

class dataset_evaluation(ABC):

    @staticmethod
    def eval_asr(clean_dict:dict, res_dcit:dict):
        raise NotImplementedError

class kitti_dataset(dataset_evaluation):

    @staticmethod
    def eval_asr(clean_dict:dict, res_dcit:dict):
        asr_dict = dict()
        for key, val in res_dcit.items():
            asr_dict[key] = (clean_dict[key] - res_dcit[key]) / clean_dict[key] * 100
        
        asr_str = ""
        for class_name in ["Car"]:
            for difficulty in ["moderate"]:
                for node in ["", "_R40"]:
                    asr_str += f"{class_name} AP{node} {difficulty}\t3d\tbev\timage\n"
                    for metric in ["3d", "bev", "image"]:
                        asr_str += f"{asr_dict[f"{class_name}_{metric}/{difficulty}{node}"]:.2f}\t"
                    asr_str += "\n"

        return asr_str, asr_dict

class nuscenes_dataset(dataset_evaluation):

    @staticmethod
    def eval_asr(clean_dict:dict, res_dcit:dict):
        pass


__all__: dict[dataset_evaluation] = {
    'KittiDataset': kitti_dataset,
    'NuScenesDataset': nuscenes_dataset,
}
