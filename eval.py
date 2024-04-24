import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pickle as pkl
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from raytorch.LiDAR import LiDAR_base

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils

from data_tools import adv_dataset
from eval_utils import eval_utils

import pdb
import argparse
import os
import time

def components_of_model(model, logger):
    for idx, module in enumerate(model.module_list):
        logger.info(f'Module names of model \t({idx}): \t{module._get_name()}')
        
    backbone_network = model.module_list[0]
    point_headbox = model.module_list[1]
    pointrcnn_head = model.module_list[2]

    return backbone_network, point_headbox, pointrcnn_head


def load_annotated_rooftop(ROOFTOP_ANNOTATE, logger):
    rooftop_approximate = None
    try:
        with open(ROOFTOP_ANNOTATE, "rb") as input:
            rooftop_approximate = pkl.load(input)
    except FileNotFoundError as error:
        logger.info(error.__str__())
    except TypeError as error:
        logger.info(error.__str__())
    
    # logger.info(f"rooftop_approximate: {rooftop_approximate}")

    return rooftop_approximate
    


def parse_config():
    ### Set hyperparameters, including random seed, device, dataset path, etc.
    args = argparse.ArgumentParser(description='KITTI Attack Evaluation')
    args.add_argument('--UNI_RANDOM_SEED', type=int, default=2024, help='random seed')
    args.add_argument('--DEVICE', type=int, default=0, help='device')
    args.add_argument('--EVAL_OUTPUT_DIR', type=str, default="./eval_output/", help='evaluation output directory')
    args.add_argument('--CFG_FILE', type=str, default="./cfgs/kitti_models/pointrcnn.yaml", help='configuration file')
    args.add_argument('--DATA_CONFIG_FILE', type=str, default="./cfgs/dataset_configs/kitti_dataset.yaml", help='dataset configuration file')
    args.add_argument('--DATA_PATH', type=str, default="/home/ksas/Public/datasets/KITTI", help='dataset path')
    args.add_argument('--CKPT_PATH', type=str, default="/home/ksas/Public/model_zoo/pcdet/pointrcnn_7870.pth", help='checkpoint path')
    args.add_argument('--ROOFTOP_ANNOTATE', type=str, default="/home/ksas/uzuki_space/vehicle-shape-reconstruction/rooftop_appro_std.pkl", help='rooftop annotation path')
    args.add_argument('--BATCH_SIZE', type=int, default=1, help='batch size')
    args.add_argument('--WORKERS', type=int, default=4, help='workers')
    args.add_argument('--DIST_TEST', action='store_true', help='distributed test')
    args.add_argument('--exp_name', type=str, default=str(int(time.time())), help='name of saving folder')

    args.add_argument('--ckpt_path', required=True, type=str, help='car mesh path')

    args = args.parse_args()

    ### Load the configuration file and set up the logger
    cfg_from_yaml_file(args.CFG_FILE, cfg)
    
    return args, cfg


def set_seed_and_device(args):
    ### Set the seed for numpy, torch, and device for cuda
    np.random.seed(args.UNI_RANDOM_SEED) 
    torch.manual_seed(args.UNI_RANDOM_SEED)

    torch.cuda.manual_seed(args.UNI_RANDOM_SEED)
    torch.cuda.manual_seed_all(args.UNI_RANDOM_SEED)

    torch.cuda.set_device(args.DEVICE)


if __name__ == "__main__":
    ### Setting the parameters and logger
    args, cfg = parse_config()
    ### Set the seed for numpy, torch, and device number for cuda
    set_seed_and_device(args)
    
    SAVE_PATH = f"output/eval/{args.exp_name}"
    os.makedirs(SAVE_PATH)

    logger = common_utils.create_logger(log_file = os.path.join(SAVE_PATH, "exp_log.log"))
    logger.info('-----------------Kitti Attack Evaluation-------------------------')
    logger.info(args)
    ### Load the parameters of annotated rooftop 
    rooftop_approximate = load_annotated_rooftop(args.ROOFTOP_ANNOTATE, logger)

    ### Build the dataloader
    test_set, test_loader, sampler = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=args.BATCH_SIZE,
            dist=args.DIST_TEST,
            workers=args.WORKERS,
            logger=logger,
            training=False
        )
    
    logger.info(f'Class names of samples: \t{test_set.class_names}')
    
    ### Build the neural network and load the checkpoint
    model = build_network(model_cfg=cfg.MODEL,
                          num_class=len(cfg.CLASS_NAMES),
                          dataset=test_set)
    
    model.load_params_from_file(filename=args.CKPT_PATH,
                                logger=logger,
                                to_cpu=True)
    model.cuda()
    model.eval()

    backbone_network, point_headbox, pointrcnn_head = components_of_model(model, logger)

    ### Prepare the adversarial dataset, which contains get_gradients methods and so forth, and optimizer oriented patch 
    lidar = LiDAR_base(origin=torch.tensor([0.0, 0.0, 0.0]).cuda(),
                   azi_range=[-90, 90],
                   polar_range= [-2.18, 2.0],
                   polar_num=10, azi_res=0.08)
    
    kitti_adv_dataset = adv_dataset(test_set,
                                    sample_amount=[50, 25],
                                    rooftop_approximate = rooftop_approximate,
                                    surrogate_model=None,
                                    lidar = lidar,
                                    enable_car = True,
                                    enable_ped = False,
                                    enable_bicycle = False)
    logger.info(f'Loading checkpoint from: \t{args.ckpt_path}')
    kitti_adv_dataset.load_adversarial_parameter(args.ckpt_path)
    ### Evaluate the adversarial examples
    kitti_adv_dataset.enable_adversarial_patch(True)
    eval_utils.eval_one_epoch(
            cfg=cfg,
            args=None, 
            model=model, 
            dataloader=kitti_adv_dataset,
            epoch_id=0, 
            logger=logger,
            dist_test=args.DIST_TEST,
            result_dir=Path(args.EVAL_OUTPUT_DIR),
            infer_time=True
        )

