from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule


import os
from easydict import EasyDict

import yaml
from torch import optim, nn, utils, Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L

from pcdet.datasets import build_dataloader
from pcdet.datasets.dataset import DatasetTemplate
from pcdet.models import build_network, model_fn_decorator, load_data_to_gpu
from pcdet.models.detectors import Detector3DTemplate
from dummy import MNISTData

from typing import List

class print_logger:
    def info(self, *args, **kwargs):
        print(*args, **kwargs)

def convert_to_easydict(config:EasyDict):
    if not isinstance(config, EasyDict):
        config = EasyDict(config)
    
    if '_BASE_CONFIG_' in config:
        yaml_config = None
        with open(config['_BASE_CONFIG_'], 'r') as f:
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
    def __init__(self, pcdet_dataset_config:dict, 
                 class_names: List[str],
                 batch_size:int, 
                 dist_train:bool, 
                 workers:int,
                 merge_all_iters_to_one_epoch:bool,
                 total_epochs:int,
                ):
        super().__init__()
        self.pcdet_dataset_config = convert_to_easydict(pcdet_dataset_config)
        
        self.class_names = class_names
        self.batch_size = batch_size
        self.dist_train = dist_train
        self.workers = workers
        self.merge_all_iters_to_one_epoch = merge_all_iters_to_one_epoch
        self.total_epochs = total_epochs
    
    def setup(self, stage: str) -> None:
        from pcdet.datasets import __all__
        if stage in ("fit", "validate"):
            self.dataset:DatasetTemplate = __all__[self.pcdet_dataset_config.DATASET](
                dataset_cfg=self.pcdet_dataset_config,
                class_names=self.class_names,
                root_path=None,
                training=False,
                logger=print_logger(),
            )
        elif stage == "test":
            self.dataset:DatasetTemplate = __all__[self.pcdet_dataset_config.DATASET](
                dataset_cfg=self.pcdet_dataset_config,
                class_names=self.class_names,
                root_path=None,
                training=False,
                logger=print_logger(),
            )
        # if stage in ("fit", "validate"):
        #     self.mnist_val = MNIST("./dummy", train=True, download=True, transform=ToTensor())

        # if stage == "test":
        #     self.mnist_test = MNIST("./dummy", train=False, download=True, transform=ToTensor())

        # if stage == "predict":
        #     self.random_predict = MNIST("./dummy", train=False, download=True, transform=ToTensor())


    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, 
                          batch_size=self.batch_size, 
                          collate_fn=self.dataset.collate_batch,)
        
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, 
                          batch_size=self.batch_size, 
                          collate_fn=self.dataset.collate_batch,)
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, 
                          batch_size=self.batch_size, 
                          collate_fn=self.dataset.collate_batch,)

class pcdet_model(L.LightningModule):
    def __init__(self, pcdet_model_config:EasyDict, 
                 num_class:int, 
                 dataset:Dataset):
        super().__init__()
        self.model:Detector3DTemplate = build_network(model_cfg=pcdet_model_config, num_class=num_class, dataset=dataset)
        if pcdet_model_config.ckpt_path is not None:
            self.model.load_params_from_file(filename=pcdet_model_config.ckpt_path, logger=print_logger(), to_cpu=True)
        self.model.cuda()
        self.model.eval()
        
    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

class physical_attack(L.LightningModule):
    def __init__(self, pcdet_model_config):
        """
            Dataset (OpenPCDet) -> Lightning
            Adversarial Dataset -> Lightning
            Adversarial Patch -> Lightning
            Model (OpenPCDet) -> Lightning
            Adversarial Loss -> Lightning
        """
        super().__init__()
        self.pcdet_model_config = convert_to_easydict(pcdet_model_config)
        self.pcdet_model = None
        self.datamodule:pcdet_dataset = None
        
    def prepare_data(self):
        self.datamodule = self.trainer.datamodule
        
    def configure_model(self):
        
        self.pcdet_model = pcdet_model(self.pcdet_model_config, 
                                       self.datamodule.class_names.__len__(),
                                       self.datamodule.dataset)
        self.pcdet_model.freeze()
        # self.pcdet_model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
        # self.pcdet_model = self.pcdet_model(self.model_config, )
    
    def training_step(self, batch_dict, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        # dict_keys(['frame_id', 'calib', 'gt_boxes', 'points', 'flip_x', 'noise_rot', 'noise_scale', 
        # 'lidar_aug_matrix', 'use_lead_xyz', 'voxels', 'voxel_coords', 'voxel_num_points', 
        # 'image_shape', 'batch_size', 'pillar_features', 'spatial_features', 'spatial_features_2d', 
        # 'batch_cls_preds', 'batch_box_preds', 'cls_preds_normalized'])
        
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict = self.pcdet_model(batch_dict)
        # list[dict_keys(['pred_boxes', 'pred_scores', 'pred_labels'])]
        # dict_keys(['gt', 'roi_0.3', 'rcnn_0.3', 'roi_0.5', 'rcnn_0.5', 'roi_0.7', 'rcnn_0.7'])
        import pdb; pdb.set_trace()
        
    def on_test_epoch_start(self):
        self.det_annos = []
    
    def test_step(self, batch_dict, batch_idx):
        load_data_to_gpu(batch_dict)
        pred_dicts, ret_dict = self.pcdet_model(batch_dict)
        annos = self.datamodule.dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, self.datamodule.class_names,
            output_path=None
        )
        self.det_annos += annos
        
    def on_test_epoch_end(self):
        result_str, result_dict = self.datamodule.dataset.evaluation(
            self.det_annos, self.datamodule.class_names,
            eval_metric=self.pcdet_model_config.POST_PROCESSING.EVAL_METRIC,
            output_path=None
        )
        
        print(result_str)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def cli_main():
    cli = LightningCLI(physical_attack, pcdet_dataset)
    # note: don't call fit!!

if __name__ == "__main__":
    cli_main()