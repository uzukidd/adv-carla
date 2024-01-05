UNI_RANDOM_SEED = 2024

import numpy as np
import torch
import torch.nn.functional as F

np.random.seed(UNI_RANDOM_SEED) 
torch.manual_seed(UNI_RANDOM_SEED)

torch.cuda.manual_seed(UNI_RANDOM_SEED)
torch.cuda.manual_seed_all(UNI_RANDOM_SEED)

import pdb
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

from pcdet.datasets.kitti.kitti_dataset import create_kitti_infos
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import KittiDataset, build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

from eval_utils import eval_utils


EVAL_OUTPUT_DIR = "./eval_output/"
CFG_FILE = "./cfgs/kitti_models/pointrcnn.yaml"
DATA_CONFIG_FILE = "./cfgs/dataset_configs/kitti_dataset.yaml"
DATA_PATH = "/home/ksas/Public/datasets/KITTI"
CKPT_PATH = "/home/ksas/Public/model_zoo/pcdet/pointrcnn_7870.pth"

BATCH_SIZE = 1
WORKERS = 4
DIST_TEST = False

cfg_from_yaml_file(CFG_FILE, cfg)

# BATCH_SIZE = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
logger = common_utils.create_logger()
logger.info('-----------------Gradient Fetching Test-------------------------')

test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=BATCH_SIZE,
        dist=DIST_TEST, workers=WORKERS, logger=logger, training=False
    )
logger.info(f'Class names of samples: \t{test_set.class_names}')

model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
model.load_params_from_file(filename=CKPT_PATH, logger=logger, to_cpu=True)
model.cuda()
model.eval()

for idx, module in enumerate(model.module_list):
    logger.info(f'Module names of model \t({idx}): \t{module._get_name()}')
    
backbone_network = model.module_list[0]
point_headbox = model.module_list[1]
pointrcnn_head = model.module_list[2]


def pseudo_train_test():
    for i, batch_dict in enumerate(test_loader):
        load_data_to_gpu(batch_dict)
        
        model.eval()
        model.pseudo_train()
        model.zero_grad()
        pred_dicts, _ = model(batch_dict)
        
        loss, tb_dict, disp_dict = model.get_training_loss()
        logger.info(f"total loss: \t{loss}")
        
        loss_dict = {}
       
        point_headbox_cls_loss, cls_loss_dict = point_headbox.get_cls_layer_loss()
        point_headbox_box_loss, box_loss_dict = point_headbox.get_box_layer_loss()
        loss_dict.update(cls_loss_dict)
        loss_dict.update(box_loss_dict)
        
        rcnn_cls_loss, cls_loss_dict = pointrcnn_head.get_box_cls_layer_loss()
        rcnn_reg_loss, reg_loss_dict = pointrcnn_head.get_box_reg_layer_loss()
        loss_dict.update(cls_loss_dict)
        loss_dict.update(reg_loss_dict)
        
        logger.info(f"loss dict: \t{loss_dict}")
        
        V.draw_scenes(
            points=batch_dict["points"][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'].detach(),
            ref_scores=pred_dicts[0]['pred_scores'].detach(), ref_labels=pred_dicts[0]['pred_labels'].detach(), gt_boxes=batch_dict["gt_boxes"][0, :, :].detach()
        )

        
pseudo_train_test()