import json
import math
import os
import pickle
import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import tqdm
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils


def convert_abs_to_percent(clean_dict: dict, res_dict: dict, exclude_key=[]):
    asr_dict = dict()
    for key, val in res_dict.items():
        if key in exclude_key:
            continue

        if isinstance(val, dict):
            asr_dict[key] = convert_abs_to_percent(clean_dict[key], val, exclude_key)
        elif isinstance(val, float):
            if clean_dict[key] == 0.0:
                clean_dict[key] = 1e-6

            asr_dict[key] = (clean_dict[key] - res_dict[key]) / clean_dict[key] * 100

    return asr_dict


class dataset_evaluation(ABC):

    @staticmethod
    @abstractmethod
    def evaluate_ASR(clean_dict: dict, res_dict: dict, log_dir: str):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def parse_raw_result(res_dict: dict, log_dir: str):
        raise NotImplementedError


class kitti_dataset(dataset_evaluation):

    @staticmethod
    def evaluate_ASR(clean_dict: dict, res_dict: dict, log_dir: str):
        asr_dict = convert_abs_to_percent(clean_dict, res_dict)
        asr_str = ""
        for class_name in ["Car"]:
            for difficulty in ["moderate"]:
                for node in ["", "_R40"]:
                    asr_str += f"{class_name} AP{node} {difficulty}\t3d\tbev\timage\n"
                    for metric in ["3d", "bev", "image"]:
                        asr_str += f"{asr_dict[f"{class_name}_{metric}/{difficulty}{node}"]:.2f}\t"
                    asr_str += "\n"

        return asr_str, asr_dict

    @staticmethod
    def parse_raw_result(res_dict: dict, log_dir: str):
        new_res_str = ""
        new_res_dict = {}
        for class_name in ["Car"]:
            for difficulty in ["moderate"]:
                for node in ["", "_R40"]:
                    new_res_str += (
                        f"{class_name} AP{node} {difficulty}\t3d\tbev\timage\n"
                    )
                    for metric in ["3d", "bev", "image"]:
                        new_res_str += f"{res_dict[f"{class_name}_{metric}/{difficulty}{node}"]:.2f}\t"
                        new_res_dict[f"{class_name}_{metric}/{difficulty}{node}"] = (
                            res_dict[f"{class_name}_{metric}/{difficulty}{node}"]
                        )
                    new_res_str += "\n"

        return new_res_str, new_res_dict


class nuscenes_dataset(dataset_evaluation):

    @staticmethod
    def evaluate_ASR(clean_dict: dict, res_dict: dict, log_dir: str):
        with open(os.path.join(log_dir, "metrics_summary.json"), "r") as f:
            res_dict = json.load(f)

        asr_dict = convert_abs_to_percent(clean_dict, res_dict, ["meta", "cfg"])
        asr_str = ""
        return str(json.dumps(asr_dict["label_aps"]["car"], indent=4)), asr_dict

    @staticmethod
    def parse_raw_result(res_dict: dict, log_dir: str):
        with open(os.path.join(log_dir, "metrics_summary.json"), "r") as f:
            res_dict = json.load(f)

        new_res_str = ""
        new_res_dict = dict()
        for class_name in ["car"]:
            for metric in ["label_aps", "mean_dist_aps", "label_tp_errors"]:
                new_res_dict[f"{class_name}/{metric}"] = res_dict[metric][class_name]

        return new_res_str, new_res_dict


__all__: Dict[str, dataset_evaluation] = {
    "KittiDataset": kitti_dataset,
    "NuScenesDataset": nuscenes_dataset,
}
