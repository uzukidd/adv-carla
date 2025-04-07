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
from pytorch3d.vis.plotly_vis import plot_scene

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils

from data_tools import adv_dataset, kitti_carla_dataset

from eval_utils import eval_utils
from loss_utils import relevant_bounding_box_loss

import argparse
import os
import time

def gtbox_wise_cos_compute(mesh_proposal_loss:torch.Tensor, 
                           gtbox_idx:torch.Tensor,
                           gtbox_size:int,
                           optimizer,
                           universal_adv_patch):
    grad_list = []
    for idx in range(gtbox_size):
        gtbox_mask = (gtbox_idx == idx)
        masked_loss = mesh_proposal_loss[gtbox_mask]
        optimizer.zero_grad()
        masked_loss.sum().backward(retain_graph = True)
        grad_list.append(universal_adv_patch.get_mesh_gradient())
    
    # for i in range(gtbox_size):
    #     for j in range(gtbox_size - i - 1):
            
    # pdb.set_trace()

def vis_adv_examples(kitti_adv_dataset):
    fig = plot_scene({
        "original": {
            "mesh_1": kitti_adv_dataset.universal_adv_patch.get_basic_mesh()
        },
        "adversarial": {
            "mesh_1": kitti_adv_dataset.universal_adv_patch.get_deformed_mesh()
        },
    }, ncols=2)
    fig.update_layout(height=400, width=800)
    fig.show()


def eval_data(args, cfg, adv_enabled, model, dataset, logger):
    model.eval()
    dataset.enable_adversarial_patch(adv_enabled)
    eval_utils.eval_one_epoch(
            cfg,
            args = None,
            model = model,
            dataloader = dataset,
            epoch_id = 0,
            logger = logger,
            dist_test = args.DIST_TEST,
            result_dir = Path(args.EVAL_OUTPUT_DIR),
            infer_time = True
        )


def components_of_model(model, logger):
    for idx, module in enumerate(model.module_list):
        logger.info(f'Module names of model \t({idx}): \t{module._get_name()}')

def parse_config():
    ### Set hyperparameters, including random seed, device, dataset path, etc.
    args = argparse.ArgumentParser(description='KITTI Eval Test')
    args.add_argument('--UNI_RANDOM_SEED', type=int, default=2024, help='random seed')
    args.add_argument('--device', type=int, default=0, help='device')
    args.add_argument('--EVAL_OUTPUT_DIR', type=str, default="./eval_output/", help='evaluation output directory')
    
    args.add_argument('--cfg-file', type=str, default="configs/attack_configs/relevant_bounding_box_pointpillar.yaml", help='configuration file')
    
    args.add_argument('--BATCH_SIZE', type=int, default=1, help='batch size')
    args.add_argument('--WORKERS', type=int, default=4, help='workers')
    args.add_argument('--DIST_TEST', action='store_true', help='distributed test')

    args.add_argument('--eval-clean-data', action='store_true', help='evaluate clean data (only)')
    args.add_argument('--eval-init-patch', action='store_true', help='evaluate initial patch (only)')

    args.add_argument('--scale', nargs='*', help='Scale of Patch')
    args.add_argument('--level', type=int, default=2, help='level of Patch')
    args.add_argument('--exp-name', type=str, default=str(int(time.time())), help='name of saving folder')
    args.add_argument('--verbose-epoch', type=int, default=-1, help='verbose per epoch')
    args.add_argument('--visualize', action='store_true')
    
    args.add_argument('--dataset', type=str,  help='choose dataset type (kitti or kitti-carla)')
    args.add_argument("--prepared-gtboxes", action='store_true')

    args = args.parse_args()

    ### Load the configuration file and set up the logger
    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def load_dataset(args, cfg, logger):
    dataset_cfg = cfg.DATA_CONFIG
    attack_cfg = cfg.ATTACK_CONFIG
    if dataset_cfg.DATASET == "KittiDataset":
        dataset, test_loader, sampler = build_dataloader(
                dataset_cfg=dataset_cfg,
                class_names=cfg.CLASS_NAMES,
                batch_size=args.BATCH_SIZE,
                dist=args.DIST_TEST,
                workers=args.WORKERS,
                logger=logger,
                training=False
            )
        
    elif dataset_cfg.DATASET == "KittiCarlaDataset":
        dataset = kitti_carla_dataset(dataset_cfg, 
                                      class_names=cfg.CLASS_NAMES, 
                                      training=False, 
                                      ext=".ply", 
                                    #   gtboxes_path = dataset_cfg.GTBOXES,
                                      logger=logger)
        
    return dataset


def set_seed_and_device(args):
    ### Set the seed for numpy, torch, and device for cuda
    np.random.seed(args.UNI_RANDOM_SEED)
    torch.manual_seed(args.UNI_RANDOM_SEED)

    torch.cuda.manual_seed(args.UNI_RANDOM_SEED)
    torch.cuda.manual_seed_all(args.UNI_RANDOM_SEED)

    torch.cuda.set_device(args.device)


def main():
    ### Setting the parameters and logger
    args, cfg = parse_config()
    
    dataset_cfg = cfg.DATA_CONFIG
    model_cfg = cfg.MODEL
    attack_cfg = cfg.ATTACK_CONFIG

    ### Set the seed for numpy, torch, and device number for cuda
    set_seed_and_device(args,)
    
    args.SAVE_PATH = f"output/eval/{args.exp_name}"
    os.makedirs(args.SAVE_PATH,)
    logger = common_utils.create_logger(log_file = os.path.join(args.SAVE_PATH, "exp_log.log"),)
    logger.info('-----------------Eval Test-------------------------')
    logger.info(args,)
    ### Load the parameters of annotated rooftop 

    ### Build the dataloader
    dataset = load_dataset(args, 
                           cfg, 
                           logger,)
    logger.info(f'Class names of samples: \t{dataset.class_names}')
    
    ### Build the neural network and load the checkpoint
    model = build_network(model_cfg=model_cfg,
                          num_class=len(cfg.CLASS_NAMES),
                          dataset=dataset,)
    
    model.load_params_from_file(filename=model_cfg.CKPT_PATH,
                                logger=logger,
                                to_cpu=True,)
    model.cuda()
    model.eval()

    components_of_model(model, 
                        logger,)

    ### Prepare the adversarial dataset, which contains get_gradients methods and so forth, and optimizer oriented patch 
    lidar = LiDAR_base(origin=torch.tensor([0.0, 0.0, 0.0]).cuda(),
                   azi_range=[-90, 90],
                   polar_range= [-2.18, 2.0],
                   polar_num=10, 
                   azi_res=0.08,)
    kitti_adv_dataset = adv_dataset(dataset,
                                    attack_cfg,
                                    surrogate_model=model,
                                    lidar = lidar,
                                    enable_car = True,
                                    enable_ped = False,
                                    enable_bicycle = False,
                                    car_adv_patch_scale = args.scale,
                                    car_adv_patch_level = args.level,)
    logger.info(f"parameter length:\t{kitti_adv_dataset.get_adversarial_parameter().__len__()}")

    if args.prepared_gtboxes:
        kitti_adv_dataset.prepare_predicted_gtboxes(path = os.path.join(args.SAVE_PATH, "gtboxes.pt"),)
        return
    
    logger.info(f"Try loading checkpoint at {attack_cfg.ADVERSARIAL_PATCH}...")
    kitti_adv_dataset.load_adversarial_parameter(attack_cfg.ADVERSARIAL_PATCH)

    logger.info("Evaluate the adversarial patch")
    eval_data(args = args, 
            cfg = cfg,
            adv_enabled = True, 
            model = model, 
            dataset = kitti_adv_dataset, 
            logger = logger)
    return

if __name__ == "__main__":
    main()