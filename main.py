import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pathlib import Path
torch.autograd.set_detect_anomaly(True)
# try:
#     import open3d
#     from visual_utils import open3d_vis_utils as V
#     OPEN3D_FLAG = True
# except:
#     import mayavi.mlab as mlab
#     from visual_utils import visualize_utils as V
#     OPEN3D_FLAG = False

from raytorch.LiDAR import LiDAR_base
from pytorch3d.vis.plotly_vis import plot_scene

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils

from data_tools import adv_dataset, kitti_carla_dataset
from eval_utils import eval_utils
from loss_utils import relevant_bounding_box_loss
from white_box_attack import run_one_epoch_white_box_attack
from query_attack import run_one_epoch_query_attack

import pdb
import argparse
import os
import time


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
    for idx, module in enumerate(model.module_list):
        module.eval()
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
    args = argparse.ArgumentParser(description='KITTI Attack Test')
    args.add_argument('--UNI_RANDOM_SEED', type=int, default=2024, help='random seed')
    args.add_argument('--device', type=int, default=0, help='device')
    args.add_argument('--EVAL_OUTPUT_DIR', type=str, default="./eval_output/", help='evaluation output directory')
    
    args.add_argument('--patch-ckpt', type=str, default=None, help='checkpoint of adversarial patch')
    args.add_argument('--cfg-file', type=str, default="configs/attack_configs/relevant_bounding_box_pointpillar.yaml", help='configuration file')
    
    args.add_argument('--BATCH_SIZE', type=int, default=1, help='batch size')
    args.add_argument('--WORKERS', type=int, default=4, help='workers')
    args.add_argument('--DIST_TEST', action='store_true', help='distributed test')
    args.add_argument('--optim', type=str, default="ifgsm", choices=["adam", "ifgsm"], help='optimization method')
    args.add_argument('--headbox-attack', action='store_true', help='enable headbox attack')
    args.add_argument('--roihead-attack', action='store_true', help='enable roihead attack')

    args.add_argument('--eval-clean-data', action='store_true', help='evaluate clean data (only)')
    args.add_argument('--eval-init-patch', action='store_true', help='evaluate initial patch (only)')
    args.add_argument('--roi-head-weights', type=float, default=1.0, help='roi head weights')
    args.add_argument('--laplacian-weights', type=float, default=0.001, help='laplacian weights')
    args.add_argument('--learning-rate', type=float, default=0.005, help='learning rate')
    args.add_argument('--scale', nargs='*', help='Scale of Patch')
    args.add_argument('--level', type=int, default=2, help='level of Patch')
    args.add_argument('--iou-frozen', action='store_true', help='freeze iou loss while optimization')
    args.add_argument('--logit-frozen', action='store_true', help='freeze logit loss while optimization')
    args.add_argument('--exp-name', type=str, default=str(int(time.time())), help='name of saving folder')
    args.add_argument('--verbose-epoch', type=int, default=-1, help='verbose per epoch')
    args.add_argument('--visualize', action='store_true')
    
    args.add_argument('--stage-1-loss-reduce-func', type=str, default="physical_loss")
    args.add_argument('--stage-2-loss-reduce-func', type=str, default="score_multiply_iou3d")
    args.add_argument("--surrogate-stage-1", action='store_true')

    args = args.parse_args()
    ### Load the configuration file and set up the logger
    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg



def set_seed_and_device(args):
    ### Set the seed for numpy, torch, and device for cuda
    np.random.seed(args.UNI_RANDOM_SEED) 
    torch.manual_seed(args.UNI_RANDOM_SEED)

    torch.cuda.manual_seed(args.UNI_RANDOM_SEED)
    torch.cuda.manual_seed_all(args.UNI_RANDOM_SEED)

    torch.cuda.set_device(args.device)
    
def build_model_from_cfg(model_cfg,
                         dataset,
                         logger,):
    model = build_network(model_cfg=model_cfg,
                          num_class=len(dataset.class_names),
                          dataset=dataset)
    
    model.load_params_from_file(filename=model_cfg.CKPT_PATH,
                                logger=logger,
                                to_cpu=True)
    model.cuda()
    model.eval()
    
    return model

def load_dataset(args, dataset_cfg, class_names, logger):
    if dataset_cfg.DATASET == "KittiDataset":
        dataset, test_loader, sampler = build_dataloader(
                dataset_cfg=dataset_cfg,
                class_names=class_names,
                batch_size=args.BATCH_SIZE,
                dist=args.DIST_TEST,
                workers=args.WORKERS,
                logger=logger,
                training=False
            )
        
    elif dataset_cfg.DATASET == "KittiCarlaDataset":
        dataset = kitti_carla_dataset(dataset_cfg, 
                                      class_names=class_names, 
                                      training=False, 
                                      ext=".ply", 
                                    #   gtboxes_path = dataset_cfg.GTBOXES,
                                      logger=logger)
        
    return dataset

def build_model_pipeline(args, logger, lidar, dataset_cfg,
                          model_cfg,
                          attack_cfg):
    
    dataset = load_dataset(args, dataset_cfg, model_cfg.CLASS_NAMES, logger)
    logger.info(f'Class names of samples: \t{dataset.class_names}')

    model = build_model_from_cfg(model_cfg,
                         dataset,
                         logger,)
    components_of_model(model, logger)
    
    adv_pipeline = adv_dataset(dataset,
                            attack_cfg,
                            surrogate_model=model,
                            lidar = lidar,
                            enable_car = True,
                            enable_ped = False,
                            enable_bicycle = False,
                            car_adv_patch_scale = args.scale,
                            car_adv_patch_level = args.level,
                            )
    
    logger.info(f"parameter length:\t{[para.size() for para in adv_pipeline.get_adversarial_parameter()]}")
    
    return dataset, model, adv_pipeline

def main():
    ### Setting the parameters and logger
    args, cfg = parse_config()
    
    attack_method = cfg.ATTACK_METHOD
    dataset_cfg = cfg.DATA_CONFIG
    model_cfg = cfg.MODEL
    attack_cfg = cfg.ATTACK_CONFIG

    ### Set the seed for numpy, torch, and device number for cuda
    set_seed_and_device(args)
    
    args.SAVE_PATH = f"output/train/{args.exp_name}"
    os.makedirs(args.SAVE_PATH)
    logger = common_utils.create_logger(log_file = os.path.join(args.SAVE_PATH, "exp_log.log"))
    logger.info('-----------------Kitti Attack Test-------------------------')
    logger.info(args)

    ### Prepare the adversarial dataset, which contains get_gradients methods and so forth, and optimizer oriented patch 
    lidar = LiDAR_base(origin=torch.tensor([0.0, 0.0, 0.0]).cuda(),
                   azi_range=[-90, 90],
                   polar_range= [-2.18, 2.0],
                   polar_num=10, azi_res=0.08)
    
    dataset, model, adv_pipeline = build_model_pipeline(args, logger, lidar, dataset_cfg,
                          model_cfg,
                          attack_cfg)
    
    if attack_method == "query":
        surrogate_dataset_cfg = cfg.SURROGATE_DATA_CONFIG
        surrogate_model_cfg = cfg.SURROGATE_MODEL
        surrogate_attack_cfg = cfg.SURROGATE_ATTACK_CONFIG

        surrogate_dataset, surrogate_model, surrogate_adv_pipeline = build_model_pipeline(args, logger, lidar, surrogate_dataset_cfg,
                          surrogate_model_cfg,
                          surrogate_attack_cfg)
    elif attack_method == "evaluate" or attack_method == "inference":
        if attack_cfg.ADVERSARIAL_PATCH is not None:
            adv_pipeline.load_adversarial_parameter(attack_cfg.ADVERSARIAL_PATCH)
        elif args.patch_ckpt is not None:
            adv_pipeline.load_adversarial_parameter(args.patch_ckpt)

    
    if args.eval_clean_data:
        logger.info("Evaluate the clean data")
        eval_data(args = args, 
                cfg = cfg,
                adv_enabled = False, 
                model = model, 
                dataset = adv_pipeline, 
                logger = logger)
        return
    
    if args.eval_init_patch:
        logger.info("Evaluate the initial patch")
        eval_data(args = args, 
                cfg = cfg,
                adv_enabled = True, 
                model = model, 
                dataset = adv_pipeline, 
                logger = logger)
        return
    
    ### Evaluate patch attack with one epoch
    adv_pipeline.save_adversarial_parameter(os.path.join(args.SAVE_PATH, "initial_patch_checkpoint.pt"))
    logger.info(f'Checkpoint has been saved as: \t{os.path.join(args.SAVE_PATH, "initial_patch_checkpoint.pt")}')

    if attack_method == "optimize":
        run_one_epoch_white_box_attack(args,
                    dataset = adv_pipeline,
                    model = model,
                    enable_adv = True, 
                    update = True, 
                    visualize = True, 
                    logger = logger,
                    verbose_epoch = args.verbose_epoch)
    elif attack_method == "query":
        run_one_epoch_query_attack(args,
                surrogate_dataset = surrogate_adv_pipeline,
                victim_dataset = adv_pipeline,
                surrogate_model = surrogate_model,
                victim_model = model,
                enable_adv = True, 
                update = True, 
                visualize = True, 
                logger = logger,
                verbose_epoch = args.verbose_epoch)
    elif attack_method == "evaluate":
        logger.info("Evaluate final patch")
        eval_data(args = args, 
                cfg = cfg,
                adv_enabled = True, 
                model = model, 
                dataset = adv_pipeline, 
                logger = logger)
        return
    elif attack_method == "inference":
        adv_pipeline.prepare_predicted_gtboxes(path = os.path.join(args.SAVE_PATH, "gtboxes.pt"))
        return
    else:
        raise NotImplementedError

    adv_pipeline.save_adversarial_parameter(os.path.join(args.SAVE_PATH, "final_adversarial_patch_checkpoint.pt"))
    logger.info(f'Checkpoint has been saved as: \t{os.path.join(args.SAVE_PATH, "final_adversarial_patch_checkpoint.pt")}')
    ### Visualize the adversarial examples
    # vis_adv_examples(kitti_adv_dataset)

    ### Evaluate the adversarial examples
    logger.info("Evaluate the final patch")
    eval_data(args = args, 
            cfg = cfg,
            adv_enabled = True, 
            model = model, 
            dataset = adv_pipeline, 
            logger = logger)

if __name__ == "__main__":
    main()