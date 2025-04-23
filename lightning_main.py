import sys

global DEBUG_STATUS
DEBUG_STATUS = "pdb" in sys.modules

import json
import os
from typing import Callable, Iterable, List, Optional

import lightning as L
import torch
from lightning.pytorch.cli import LightningCLI
from pcdet.utils import common_utils
from pytorch3d.io import save_obj
from pytorch_lightning.callbacks import ModelCheckpoint
from raytorch.LiDAR import LiDAR_base
from torch.optim import Optimizer

import wandb
from attack_utils import encode_adversarial_target, physical_adversary, single_sphere
from data_utils import resgister_data_processor
from lightning_module import convert_to_easydict, pcdet_dataset, pcdet_model
from loss_utils import relevant_bounding_box_loss

os.environ["WANDB_DISABLE_GPU"] = "true"
os.environ["WANDB_DISABLE_CODE"] = "true"


def breakpoint_if_pdb():
    if DEBUG_STATUS:
        import pdb

        pdb.set_trace()


class adversarial_patch(L.LightningModule):
    def __init__(self, car_adv_patch_scale, car_adv_patch_level):
        super().__init__()
        self.universal_adv_patch_hcar = single_sphere(
            semi_scale=torch.Tensor(car_adv_patch_scale).to(self.device),
            level=car_adv_patch_level,
        )


optimizer_callable = Callable[[Iterable], Optimizer]
target_encoder_callable = Callable[[Iterable], encode_adversarial_target]


class physical_attack(L.LightningModule):
    def __init__(
        self,
        pcdet_model_config,
        adversary_config: Optional[dict] = None,
        optimizer: optimizer_callable = torch.optim.Adam,
        target_encoder: target_encoder_callable = encode_adversarial_target,
    ):
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
        self.datamodule: pcdet_dataset = None
        self.optimizer = optimizer
        self.target_encoder = target_encoder

        self.benchmark = None
        self.adversarial_patch = None
        self.adversary = None

        if adversary_config is not None:
            self.adversary_config = convert_to_easydict(adversary_config)
            self.adversary = physical_adversary(self.adversary_config)
            resgister_data_processor(
                "physical_adversary", self.adversary.physical_adversary
            )
            resgister_data_processor("collate_gtboxes", self.adversary.collate_gtboxes)

        self.loss = relevant_bounding_box_loss(1)

    def get_logging_dir(self):
        if wandb.run is not None:
            return wandb.run.dir
        return None

    def configure_experiment_name(self):
        experiment_name = (
            f"{self.trainer.datamodule.pcdet_dataset_config.DATASET}-"
            f"{self.pcdet_model_config.NAME}-"
            f"{self.trainer.state.fn.value}"
        )
        if getattr(self.trainer.logger, "_wandb_init", None) is not None:
            self.trainer.logger._wandb_init["name"] = experiment_name

    def configure_callbacks(self):
        self.configure_experiment_name()
        return super().configure_callbacks()

    def configure_model(self):
        # Initialize logger
        if self.local_rank == 0:
            self.print_logger = common_utils.create_logger(
                os.path.join(self.get_logging_dir(), "runtime_log.log"), self.local_rank
            )

        self.datamodule = self.trainer.datamodule
        self.pcdet_model = pcdet_model(
            self.pcdet_model_config,
            self.datamodule.class_names.__len__(),
            self.datamodule.dataset,
        )
        self.pcdet_model.freeze()

        if self.adversary is not None:
            if wandb.run is not None:
                wandb.run.summary.update(self.adversary_config)
            self.target_encoder = self.target_encoder(self.pcdet_model.model)
            self.adversarial_patch = single_sphere(
                self.adversary.car_adv_patch_scale,
                self.adversary.car_adv_patch_level,
                device=self.device,
            )
            self.adversary.configure_adversary(
                self.adversarial_patch,
                LiDAR_base(
                    origin=torch.tensor([0.0, 0.0, 0.0]).to(self.device),
                    azi_range=[-90, 90],
                    polar_range=[-2.18, 2.0],
                    polar_num=10,
                    azi_res=0.08,
                ),
            )

    def print(self, *args, **kwargs):
        if self.local_rank == 0:
            self.print_logger.info(*args, **kwargs)

    def visualize_frame(self, point_clouds: torch.Tensor, gt_boxes: torch.Tensor):
        if self.local_rank == 0:
            os.makedirs(os.path.join(self.trainer.log_dir, "tmp"), exist_ok=True)
            points_path = os.path.join(self.trainer.log_dir, "tmp", "points.pt")
            gt_boxes_path = os.path.join(self.trainer.log_dir, "tmp", "gt_boxes.pt")
            torch.save(point_clouds, points_path)
            torch.save(gt_boxes, gt_boxes_path)
            command = [
                "python",
                "visual_utils/vis_terminal.py",
                "-p",
                points_path,
                "-g",
                gt_boxes_path,
            ]
            import subprocess

            subprocess.run(command, check=True, text=True, capture_output=False)
            os.remove(points_path)
            os.remove(gt_boxes_path)

    def on_save_checkpoint(self, checkpoint):
        if self.local_rank == 0:
            adversarial_meshes = self.adversarial_patch.get_transformed_meshes()
            save_obj(
                os.path.join(self.get_logging_dir(), "adversarial_meshes.obj"),
                adversarial_meshes.verts_packed(),
                adversarial_meshes.faces_packed(),
            )

        all_parameter = dict(self.named_parameters())
        for name, param in all_parameter.items():
            if not param.requires_grad:
                del checkpoint["state_dict"][name]

        return super().on_save_checkpoint(checkpoint)

    def iterate_evaluation(self, batch_dict, batch_idx):
        pred_dicts, ret_dict = self.pcdet_model(batch_dict)
        annos = self.datamodule.dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, self.datamodule.class_names, output_path=None
        )
        self.det_annos += annos

    def evaluate_pred_result(self):
        if self.det_annos is None or self.det_annos.__len__() == 0:
            return

        if self.trainer.world_size > 1:
            self.det_annos = common_utils.merge_results_dist(
                self.det_annos,
                len(self.datamodule.dataset),
                tmpdir=os.path.join(self.trainer.log_dir, "tmpdir"),
            )
        if self.local_rank == 0:
            result_str, result_dict = self.datamodule.dataset.evaluation(
                self.det_annos,
                self.datamodule.class_names,
                eval_metric=self.pcdet_model_config.POST_PROCESSING.EVAL_METRIC,
                output_path=self.get_logging_dir(),
            )
            with open(
                os.path.join(
                    self.get_logging_dir(),
                    f"epoch_{self.current_epoch}_result_dict.json",
                ),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=4)
            self.print(result_str)
            # self.logger.log_metrics(result_dict)
            wandb.run.summary.update(
                physical_adversary.evaluate_summary(
                    result_dict,
                    self.datamodule.pcdet_dataset_config.DATASET,
                    self.get_logging_dir(),
                )[1]
            )
            self.logger.log_text("log_text", ["content"], [[result_str]])

            if self.adversary is not None and self.adversary.benchmark is not None:
                asr_str, asr_dict = self.adversary.evaluate_adversary(
                    result_dict,
                    self.datamodule.pcdet_dataset_config.DATASET,
                    self.get_logging_dir(),
                )
                with open(
                    os.path.join(
                        self.get_logging_dir(),
                        f"epoch_{self.current_epoch}_adversarial_dict.json",
                    ),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(asr_dict, f, ensure_ascii=False, indent=4)
                # self.logger.log_text("log_text", ["content"], [[asr_str]])
                # wandb.run.summary.update(asr_dict)
            # dict_keys(['Car_aos/easy_R40', 'Car_aos/moderate_R40', 'Car_aos/hard_R40', 'Car_3d/easy_R40', 'Car_3d/moderate_R40', 'Car_3d/hard_R40', 'Car_bev/easy_R40', 'Car_bev/moderate_R40', 'Car_bev/hard_R40', 'Car_image/easy_R40', 'Car_image/moderate_R40', 'Car_image/hard_R40', 'Pedestrian_aos/easy_R40', 'Pedestrian_aos/moderate_R40', 'Pedestrian_aos/hard_R40', 'Pedestrian_3d/easy_R40', 'Pedestrian_3d/moderate_R40', 'Pedestrian_3d/hard_R40', 'Pedestrian_bev/easy_R40', 'Pedestrian_bev/moderate_R40', 'Pedestrian_bev/hard_R40', 'Pedestrian_image/easy_R40', 'Pedestrian_image/moderate_R40', 'Pedestrian_image/hard_R40', 'Cyclist_aos/easy_R40', 'Cyclist_aos/moderate_R40', 'Cyclist_aos/hard_R40', 'Cyclist_3d/easy_R40', 'Cyclist_3d/moderate_R40', 'Cyclist_3d/hard_R40', 'Cyclist_bev/easy_R40', 'Cyclist_bev/moderate_R40', 'Cyclist_bev/hard_R40', 'Cyclist_image/easy_R40', 'Cyclist_image/moderate_R40', 'Cyclist_image/hard_R40'])

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

        target_dicts = self.target_encoder.encode_target(batch_dict, pred_dicts)
        for batch_mask in range(batch_dict["batch_size"]):
            rrbbox_loss = self.loss.forward(
                target_dicts[batch_mask], batch_dict["gt_boxes"][batch_mask]
            )
            total_loss = total_loss + rrbbox_loss
        total_loss = total_loss / batch_dict["batch_size"]

        return total_loss

        # return batch_dict['points'].mean()

        # list[dict_keys(['pred_boxes', 'pred_scores', 'pred_labels'])]
        # dict_keys(['gt', 'roi_0.3', 'rcnn_0.3', 'roi_0.5', 'rcnn_0.5', 'roi_0.7', 'rcnn_0.7'])

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val, gradient_clip_algorithm
    ):
        self.adversarial_patch.constrain_grad()
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
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
        self.iterate_evaluation(batch_dict, batch_idx)

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
        breakpoint_if_pdb()

    def test_step(self, batch_dict, batch_idx):
        self.iterate_evaluation(batch_dict, batch_idx)

    def on_test_epoch_end(self):
        self.evaluate_pred_result()

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        return optimizer


def cli_main():
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath="checkpoints/",
    #     filename="adversary-{epoch:02d}",
    # )
    cli = LightningCLI(
        physical_attack,
        pcdet_dataset,
        auto_configure_optimizers=False,
        save_config_callback=None,
        # callbacks=[checkpoint_callback],
    )
    # note: don't call fit!!


if __name__ == "__main__":
    cli_main()
