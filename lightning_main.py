from lightning.pytorch.cli import LightningCLI

import os
from easydict import EasyDict
import yaml
import json

import torch
from torch import optim, nn, utils, Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torch.distributed as dist
from pytorch3d.io import save_obj


import lightning as L

from pcdet.utils import common_utils, commu_utils
from pcdet.datasets import DistributedSampler
from pcdet.datasets.dataset import DatasetTemplate
from pcdet.models import build_network, model_fn_decorator, load_data_to_gpu
from pcdet.models.detectors import Detector3DTemplate

from raytorch.LiDAR import LiDAR_base

from attack_utils import physical_adversary, single_sphere
from data_utils import voxel_collate_batch, resgister_data_processor
from loss_utils import relevant_bounding_box_loss

from functools import partial
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
                 workers:int,
                ):
        super().__init__()
        self.pcdet_dataset_config = convert_to_easydict(pcdet_dataset_config)
        
        self.class_names = class_names
        self.dataset = None
        self.batch_size = batch_size
        self.workers = workers
    
    def setup(self, stage: str) -> None:
        from pcdet import datasets
        print(f"STAGE: {stage}")
        if self.dataset is not None:
            return
        
        if stage in ("fit", "validate"):
            self.dataset:DatasetTemplate = datasets.__all__[self.pcdet_dataset_config.DATASET](
                dataset_cfg=self.pcdet_dataset_config,
                class_names=self.class_names,
                root_path=None,
                training=False,
                logger=print_logger(),
            )
        elif stage == "test":
            self.dataset:DatasetTemplate = datasets.__all__[self.pcdet_dataset_config.DATASET](
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
        return DistributedSampler(self.dataset, self.trainer.world_size, self.trainer.local_rank, shuffle=False)

    def build_datalaoder(self) -> DataLoader:
        return DataLoader(self.dataset, 
                          batch_size=self.batch_size, 
                          shuffle = False,
                          sampler=self.ddp_sampler(),
                          collate_fn=voxel_collate_batch,)

    def train_dataloader(self) -> DataLoader:
        return self.build_datalaoder()
        
    def test_dataloader(self) -> DataLoader:
        return self.build_datalaoder()

    def val_dataloader(self) -> DataLoader:
        return self.build_datalaoder()


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
    
class adversarial_patch(L.LightningModule):
    def __init__(self, car_adv_patch_scale, car_adv_patch_level):
        super().__init__()
        self.universal_adv_patch_car = single_sphere(semi_scale=torch.Tensor(car_adv_patch_scale).to(self.device),
                                                     level=car_adv_patch_level)

class physical_attack(L.LightningModule):
    def __init__(self, pcdet_model_config, benchmark_path, car_adv_patch_scale, car_adv_patch_level):
        """
            Dataset (OpenPCDet) -> Lightning
            Adversarial Dataset -> Lightning
            Adversarial Patch -> Lightning
            Model (OpenPCDet) -> Lightning
            Adversarial Loss -> Lightning
        """
        super().__init__()

        self.save_hyperparameters()

        self.pcdet_model_config = convert_to_easydict(pcdet_model_config)
        self.pcdet_model = None
        self.print_logger = None
        self.datamodule:pcdet_dataset = None

        self.benchmark = None
        if benchmark_path is not None:
            with open(benchmark_path) as file:
                self.benchmark = json.load(file)

        self.adversarial_patch = None
        self.car_adv_patch_scale = car_adv_patch_scale
        self.car_adv_patch_level = car_adv_patch_level
        
        self.adversary = physical_adversary()
        resgister_data_processor("physical_adversary",  self.adversary.physical_adversary)
        resgister_data_processor("collate_gtboxes",  self.adversary.collate_gtboxes)

        self.loss = relevant_bounding_box_loss(1)

    def print(self, *args, **kwargs):
        self.print_logger.info(*args, **kwargs)

    def visualize_frame(self, point_clouds:torch.Tensor, gt_boxes:torch.Tensor):
        if self.local_rank == 0:
            os.makedirs(os.path.join(self.trainer.log_dir, "tmp"), exist_ok=True)
            points_path = os.path.join(self.trainer.log_dir, "tmp", "points.pt")
            gt_boxes_path = os.path.join(self.trainer.log_dir, "tmp", "gt_boxes.pt")
            torch.save(point_clouds, points_path)
            torch.save(gt_boxes, gt_boxes_path)
            command = ["python", "visual_utils/vis_terminal.py", "-p", points_path, "-g", gt_boxes_path]
            import subprocess
            subprocess.run(
                command,
                check=True,
                text=True,
                capture_output=False
            )
            os.remove(points_path)
            os.remove(gt_boxes_path)
        
    
    def on_save_checkpoint(self, checkpoint):
        adversarial_meshes = self.adversarial_patch.get_transformed_meshes()
        save_obj(os.path.join(self.trainer.log_dir, "adversarial_meshes.obj"), 
                 adversarial_meshes.verts_packed(), 
                 adversarial_meshes.faces_packed())
        
        all_parameter = dict(self.named_parameters())
        for name, param in all_parameter.items():
            if not param.requires_grad:
                del checkpoint['state_dict'][name]

        return super().on_save_checkpoint(checkpoint)
        
    def configure_model(self):
        # Initialize logger
        self.print_logger = common_utils.create_logger(os.path.join(self.trainer.log_dir, "runtime_log.log"), 
                                                       self.local_rank)

        self.datamodule = self.trainer.datamodule
        self.pcdet_model = pcdet_model(self.pcdet_model_config, 
                                       self.datamodule.class_names.__len__(),
                                       self.datamodule.dataset)
        self.pcdet_model.freeze()
        
        self.adversarial_patch = single_sphere(self.car_adv_patch_scale, self.car_adv_patch_level, device=self.device)
        self.adversary.configure_adversary(self.adversarial_patch, 
                                            LiDAR_base(origin=torch.tensor([0.0, 0.0, 0.0]).to(self.device),
                                            azi_range=[-90, 90],
                                            polar_range= [-2.18, 2.0],
                                            polar_num=10, azi_res=0.08))

        # self.pcdet_model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
        # self.pcdet_model = self.pcdet_model(self.model_config, )

    def evaluate_pred_result(self):
        if self.det_annos is None or self.det_annos.__len__() == 0:
            return

        if self.trainer.world_size > 1:
            self.det_annos = common_utils.merge_results_dist(self.det_annos, 
                                                        len(self.datamodule.dataset), 
                                                        tmpdir=os.path.join(self.trainer.log_dir, 'tmpdir'))
        if self.local_rank == 0:
            result_str, result_dict = self.datamodule.dataset.evaluation(
                self.det_annos, self.datamodule.class_names,
                eval_metric=self.pcdet_model_config.POST_PROCESSING.EVAL_METRIC,
                output_path=self.trainer.log_dir
            )
            with open(os.path.join(self.trainer.log_dir, f"epoch_{self.current_epoch}_result_dict.json"), "w", encoding="utf-8") as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=4) 
            self.print(result_str)
            if self.benchmark is not None:
                from eval_utils import eval_utils
                dataset_eval:eval_utils.dataset_evaluation = eval_utils.__all__[self.datamodule.pcdet_dataset_config.DATASET]
                asr_str, asr_dict = dataset_eval.eval_asr(self.benchmark, result_dict)
                with open(os.path.join(self.trainer.log_dir, f"epoch_{self.current_epoch}_adversarial_dict.json"), "w", encoding="utf-8") as f:
                    json.dump(asr_dict, f, ensure_ascii=False, indent=4) 
                self.print(asr_str)
            # dict_keys(['Car_aos/easy_R40', 'Car_aos/moderate_R40', 'Car_aos/hard_R40', 'Car_3d/easy_R40', 'Car_3d/moderate_R40', 'Car_3d/hard_R40', 'Car_bev/easy_R40', 'Car_bev/moderate_R40', 'Car_bev/hard_R40', 'Car_image/easy_R40', 'Car_image/moderate_R40', 'Car_image/hard_R40', 'Pedestrian_aos/easy_R40', 'Pedestrian_aos/moderate_R40', 'Pedestrian_aos/hard_R40', 'Pedestrian_3d/easy_R40', 'Pedestrian_3d/moderate_R40', 'Pedestrian_3d/hard_R40', 'Pedestrian_bev/easy_R40', 'Pedestrian_bev/moderate_R40', 'Pedestrian_bev/hard_R40', 'Pedestrian_image/easy_R40', 'Pedestrian_image/moderate_R40', 'Pedestrian_image/hard_R40', 'Cyclist_aos/easy_R40', 'Cyclist_aos/moderate_R40', 'Cyclist_aos/hard_R40', 'Cyclist_3d/easy_R40', 'Cyclist_3d/moderate_R40', 'Cyclist_3d/hard_R40', 'Cyclist_bev/easy_R40', 'Cyclist_bev/moderate_R40', 'Cyclist_bev/hard_R40', 'Cyclist_image/easy_R40', 'Cyclist_image/moderate_R40', 'Cyclist_image/hard_R40'])

        
        # if self.trainer.world_size > 1:
        #     dist.barrier()
        # commu_utils.synchronize()
    
    """
    -----------------
    Training
    -----------------
    """
    # def on_fit_start(self):
    #     self.trainer.save_checkpoint(os.path.join(self.trainer.log_dir, "checkpoint.ckpt"))
    
    def on_train_epoch_start(self):
        self.adversary.enable_adversary(True)
    
    def on_train_epoch_end(self):
        pass
    
    def training_step(self, batch_dict, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        # dict_keys(['frame_id', 'calib', 'gt_boxes', 'points', 'flip_x', 'noise_rot', 'noise_scale', 
        # 'lidar_aug_matrix', 'use_lead_xyz', 'voxels', 'voxel_coords', 'voxel_num_points', 
        # 'image_shape', 'batch_size', 'pillar_features', 'spatial_features', 'spatial_features_2d', 
        # 'batch_cls_preds', 'batch_box_preds', 'cls_preds_normalized'])
        
        # self.visualize_frame(batch_dict['points'].cpu(), batch_dict['gt_boxes'].cpu())
        pred_dicts, ret_dict = self.pcdet_model(batch_dict)
        
        total_loss = torch.tensor(1e-6, device=self.device, requires_grad=True)
        for batch_mask in range(batch_dict["batch_size"]):
            rrbbox_loss = self.loss.forward(pred_dicts[batch_mask], batch_dict['gt_boxes'][batch_mask])
            total_loss = total_loss + rrbbox_loss
        total_loss = total_loss / batch_dict["batch_size"]

        return total_loss

        # return batch_dict['points'].mean()

        # list[dict_keys(['pred_boxes', 'pred_scores', 'pred_labels'])]
        # dict_keys(['gt', 'roi_0.3', 'rcnn_0.3', 'roi_0.5', 'rcnn_0.5', 'roi_0.7', 'rcnn_0.7'])

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        self.adversarial_patch.constrain_grad()
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm
        )

    """
    -----------------
    Validating
    -----------------
    """
    def on_validation_epoch_start(self):
        self.det_annos = []
        # if self.trainer.sanity_checking:
        #     self.adversary.enable_adversary(False)

    def validation_step(self, batch_dict, batch_idx):
        pred_dicts, ret_dict = self.pcdet_model(batch_dict)
        annos = self.datamodule.dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, self.datamodule.class_names,
            output_path=None
        )
        self.det_annos += annos
    
    def on_validation_epoch_end(self):
        self.evaluate_pred_result()
        # if self.local_rank == 0:
        #     import pdb; pdb.set_trace()
        
    """
    -----------------
    Testing
    -----------------
    """
        
    def on_test_epoch_start(self):
        self.det_annos = []
        self.adversary.enable_adversary(False)
    
    def test_step(self, batch_dict, batch_idx):
        pred_dicts, ret_dict = self.pcdet_model(batch_dict)
        annos = self.datamodule.dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, self.datamodule.class_names,
            output_path=None
        )
        self.det_annos += annos
        
    def on_test_epoch_end(self):
        self.evaluate_pred_result()
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def cli_main():
    cli = LightningCLI(physical_attack, pcdet_dataset)
    # note: don't call fit!!

if __name__ == "__main__":
    cli_main()