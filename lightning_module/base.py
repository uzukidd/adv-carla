from collections import defaultdict
from typing import List
import wandb
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict
from pcdet.datasets import DistributedSampler
from pcdet.datasets.dataset import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu, model_fn_decorator
from pcdet.models.detectors import Detector3DTemplate
from pcdet.utils import common_utils
from torch.utils.data import DataLoader, Dataset


class print_logger:
    def info(self, *args, **kwargs):
        print(*args, **kwargs)


def convert_to_easydict(config: EasyDict):
    if not isinstance(config, EasyDict):
        config = EasyDict(config)

    if "_BASE_CONFIG_" in config:
        yaml_config = None
        with open(config["_BASE_CONFIG_"], "r") as f:
            try:
                yaml_config = yaml.safe_load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.safe_load(f)

        new_config = EasyDict(yaml_config)
        for key, val in new_config.items():
            if not isinstance(val, dict):
                if key not in config:
                    config[key] = val
                continue

            if key not in config:
                config[key] = convert_to_easydict(val)

    return config


class pcdet_dataset(L.LightningDataModule):
    def __init__(
        self,
        pcdet_dataset_config: dict,
        class_names: List[str],
        batch_size: int,
        workers: int,
        disabled_processing: list = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.pcdet_dataset_config = convert_to_easydict(pcdet_dataset_config)

        self.class_names = class_names
        self.dataset = None
        self.batch_size = batch_size
        self.workers = workers

        self.disabled_processing = disabled_processing

        if disabled_processing is not None:
            self.pcdet_dataset_config.DATA_PROCESSOR = [
                config for config in self.pcdet_dataset_config.DATA_PROCESSOR if config.NAME not in disabled_processing
            ]

    def setup(self, stage: str) -> None:
        from pcdet import datasets
        if wandb.run is not None:
            wandb.run.summary.update({"disabled_processing": self.disabled_processing})
        print(f"STAGE: {stage}")
        if self.dataset is not None:
            return

        if stage in ("fit", "validate"):
            self.dataset: DatasetTemplate = datasets.__all__[
                self.pcdet_dataset_config.DATASET
            ](
                dataset_cfg=self.pcdet_dataset_config,
                class_names=self.class_names,
                root_path=None,
                training=False,
                logger=print_logger(),
            )
        elif stage == "test":
            self.dataset: DatasetTemplate = datasets.__all__[
                self.pcdet_dataset_config.DATASET
            ](
                dataset_cfg=self.pcdet_dataset_config,
                class_names=self.class_names,
                root_path=None,
                training=False,
                logger=print_logger(),
            )

    def transfer_batch_to_device(self, batch_dict, device, dataloader_idx):
        load_data_to_gpu(batch_dict)
        return batch_dict

    def ddp_sampler(self):
        if self.trainer._accelerator_connector.use_distributed_sampler:
            return None
        return DistributedSampler(
            self.dataset,
            self.trainer.world_size,
            self.trainer.local_rank,
            shuffle=False,
        )

    def build_datalaoder(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.ddp_sampler(),
            collate_fn=gpu_collate_batch,
        )

    def train_dataloader(self) -> DataLoader:
        return self.build_datalaoder()

    def test_dataloader(self) -> DataLoader:
        return self.build_datalaoder()

    def val_dataloader(self) -> DataLoader:
        return self.build_datalaoder()


class pcdet_model(L.LightningModule):
    def __init__(self, pcdet_model_config: EasyDict, num_class: int, dataset: Dataset):
        super().__init__()
        self.model: Detector3DTemplate = build_network(
            model_cfg=pcdet_model_config, num_class=num_class, dataset=dataset
        )
        if pcdet_model_config.ckpt_path is not None:
            self.model.load_params_from_file(
                filename=pcdet_model_config.ckpt_path,
                logger=print_logger(),
                to_cpu=True,
            )
        self.model.cuda()
        self.model.eval()
        # print(self.model)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)


def gpu_collate_batch(batch_list, _unused=False):
    data_dict = defaultdict(list)
    for cur_sample in batch_list:
        for key, val in cur_sample.items():
            data_dict[key].append(val)
    batch_size = len(batch_list)
    ret = {}
    batch_size_ratio = 1

    for key, val in data_dict.items():
        try:
            if key in ["voxels", "voxel_num_points"]:
                if isinstance(val[0], list):
                    batch_size_ratio = len(val[0])
                    val = [i for item in val for i in item]

                # Retain gradient graph
                if isinstance(val[0], torch.Tensor):
                    ret[key] = torch.concatenate(val, axis=0)
                elif isinstance(val[0], np.ndarray):
                    ret[key] = np.concatenate(val, axis=0)

            elif key in ["points", "voxel_coords"]:
                coors = []
                if isinstance(val[0], list):
                    val = [i for item in val for i in item]

                # Retain gradient graph
                if isinstance(val[0], torch.Tensor):
                    for i, coor in enumerate(val):
                        coor_pad = F.pad(coor, (1, 0), mode="constant", value=i)
                        coors.append(coor_pad)
                    ret[key] = torch.concatenate(coors, axis=0)

                elif isinstance(val[0], np.ndarray):
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(
                            coor, ((0, 0), (1, 0)), mode="constant", constant_values=i
                        )
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
            elif key in ["gt_boxes"]:
                max_gt = max([len(x) for x in val])
                if isinstance(val[0], np.ndarray):
                    batch_gt_boxes3d = np.zeros(
                        (batch_size, max_gt, val[0].shape[-1]), dtype=np.float32
                    )
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, : val[k].__len__(), :] = val[k]

                elif isinstance(val[0], torch.Tensor):
                    batch_gt_boxes3d = val[0].new_zeros(
                        (batch_size, max_gt, val[0].shape[-1]), dtype=torch.float
                    )
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, : val[k].__len__(), :] = val[k]
                ret[key] = batch_gt_boxes3d

            elif key in ["roi_boxes"]:
                max_gt = max([x.shape[1] for x in val])
                batch_gt_boxes3d = np.zeros(
                    (batch_size, val[0].shape[0], max_gt, val[0].shape[-1]),
                    dtype=np.float32,
                )
                for k in range(batch_size):
                    batch_gt_boxes3d[k, :, : val[k].shape[1], :] = val[k]
                ret[key] = batch_gt_boxes3d

            elif key in ["roi_scores", "roi_labels"]:
                max_gt = max([x.shape[1] for x in val])
                batch_gt_boxes3d = np.zeros(
                    (batch_size, val[0].shape[0], max_gt), dtype=np.float32
                )
                for k in range(batch_size):
                    batch_gt_boxes3d[k, :, : val[k].shape[1]] = val[k]
                ret[key] = batch_gt_boxes3d

            elif key in ["gt_boxes2d"]:
                max_boxes = 0
                max_boxes = max([len(x) for x in val])
                batch_boxes2d = np.zeros(
                    (batch_size, max_boxes, val[0].shape[-1]), dtype=np.float32
                )
                for k in range(batch_size):
                    if val[k].size > 0:
                        batch_boxes2d[k, : val[k].__len__(), :] = val[k]
                ret[key] = batch_boxes2d
            elif key in ["images", "depth_maps"]:
                # Get largest image size (H, W)
                max_h = 0
                max_w = 0
                for image in val:
                    max_h = max(max_h, image.shape[0])
                    max_w = max(max_w, image.shape[1])

                # Change size of images
                images = []
                for image in val:
                    pad_h = common_utils.get_pad_params(
                        desired_size=max_h, cur_size=image.shape[0]
                    )
                    pad_w = common_utils.get_pad_params(
                        desired_size=max_w, cur_size=image.shape[1]
                    )
                    pad_width = (pad_h, pad_w)
                    pad_value = 0

                    if key == "images":
                        pad_width = (pad_h, pad_w, (0, 0))
                    elif key == "depth_maps":
                        pad_width = (pad_h, pad_w)

                    image_pad = np.pad(
                        image,
                        pad_width=pad_width,
                        mode="constant",
                        constant_values=pad_value,
                    )

                    images.append(image_pad)
                ret[key] = np.stack(images, axis=0)
            elif key in ["calib"]:
                ret[key] = val
            elif key in ["points_2d"]:
                max_len = max([len(_val) for _val in val])
                pad_value = 0
                points = []
                for _points in val:
                    pad_width = ((0, max_len - len(_points)), (0, 0))
                    points_pad = np.pad(
                        _points,
                        pad_width=pad_width,
                        mode="constant",
                        constant_values=pad_value,
                    )
                    points.append(points_pad)
                ret[key] = np.stack(points, axis=0)
            elif key in ["camera_imgs"]:
                ret[key] = torch.stack(
                    [torch.stack(imgs, dim=0) for imgs in val], dim=0
                )
            else:
                ret[key] = np.stack(val, axis=0)
        except:
            print("Error in collate_batch: key=%s" % key)
            raise TypeError

    ret["batch_size"] = batch_size * batch_size_ratio
    return ret

