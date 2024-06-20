import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pdb
import pickle as pkl
import matplotlib.pyplot as plt
from tqdm import tqdm
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

from data_tools import adv_dataset
from eval_utils import eval_utils
from loss_utils import relevant_bounding_box_loss

import pdb
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
    
class stage_wise_full_attack(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.car_rbbox_loss_func = relevant_bounding_box_loss(frozen_iou = args.iou_frozen,
                 frozen_logit = args.logit_frozen,
                 confidence_threshold = 0.1,
                 iou_threshold = 0.1, 
                 verbose = False)

    
    def forward(self, gt_boxes, first_dict = None, second_dict = None):
        mesh_loss = torch.zeros(1, requires_grad=True).cuda()
        if self.args.headbox_attack:
            car_mesh_proposal_loss = self.car_rbbox_loss_func(batch_dict = first_dict, 
                                    gt_boxes = gt_boxes, 
                                    target_class = 1,
                                    logit_normal = "sigmoid",
                                    detector_type = "rcnn",
                                    ret_part_loss = False)
            mesh_loss = mesh_loss + car_mesh_proposal_loss
            
        if self.args.roihead_attack:
            car_mesh_single_loss = self.car_rbbox_loss_func(batch_dict = second_dict, 
                                gt_boxes = gt_boxes, 
                                target_class = 1,
                                logit_normal = "sigmoid",
                                detector_type = "single",
                                ret_part_loss = False)
            mesh_loss = mesh_loss + self.args.roi_head_weights * car_mesh_single_loss
    
        return mesh_loss
    
class loss_wise_full_attack(nn.Module):
    
    def __init__(self, args, ):
        super().__init__()
        self.args = args
        self.car_rbbox_loss_func = relevant_bounding_box_loss(frozen_iou = args.iou_frozen,
                 frozen_logit = args.logit_frozen,
                 confidence_threshold = 0.1,
                 iou_threshold = 0.1, 
                 verbose = False)
    
    def forward(self, gt_boxes, first_dict = None, second_dict = None):
        
        mesh_loss = None
        
        if self.args.mode < 2:
            iou3d, masked_cls_preds = self.car_rbbox_loss_func(batch_dict = first_dict, 
                                    gt_boxes = gt_boxes, 
                                    target_class = 1,
                                    logit_normal = "sigmoid",
                                    detector_type = "rcnn",
                                    ret_part_loss = True)
            if self.args.mode == 0:
                mesh_loss = iou3d.sum()
            elif self.args.mode == 1:
                mesh_loss = -torch.log(1 - masked_cls_preds).sum()

        elif self.args.mode >= 2:
            iou3d, masked_cls_preds = self.car_rbbox_loss_func(batch_dict = second_dict, 
                                gt_boxes = gt_boxes, 
                                target_class = 1,
                                logit_normal = "sigmoid",
                                detector_type = "single",
                                ret_part_loss = True)
            if self.args.mode == 2:
                mesh_loss = iou3d.sum()
            elif self.args.mode == 3:
                mesh_loss = -torch.log(1 - masked_cls_preds).sum()
            
        return mesh_loss

    

def run_one_epoch_attack(args, 
                         dataset: adv_dataset, 
                         model, 
                         enable_adv, 
                         update, 
                         visualize,
                         logger,
                         verbose_epoch: int = 100):
    
    adversarial_loss_func = stage_wise_full_attack(args)
    # adversarial_loss_func = loss_wise_full_attack(args)
    
    optimizer = optim.Adam(dataset.get_adversarial_parameter(), 
                       lr=args.learning_rate)

    dataset.enable_adversarial_patch(enable_adv)
    
    model.eval()
    target_component = None
    for idx, module in enumerate(model.module_list):
        if module._get_name() == dataset.target_component:
            target_component = module
        if module._get_name() in dataset.trainging_components:
            module.train()
            
    
    for i, batch_dict in tqdm(enumerate(dataset), total=dataset.__len__()):
        if not torch.eq(batch_dict['gt_boxes'][0, :, 7], 1).any():
            continue
        
        model.zero_grad()
        pred_dicts, _ = model(batch_dict)
        
        attack_dict = None
        if dataset.detector_type == "single":
            attack_dict = pred_dicts[0]
        elif dataset.detector_type == "rcnn":
            attack_dict = target_component.forward_ret_dict

        if args.OPTIM == "rbboxloss":

            adversarial_loss = adversarial_loss_func(batch_dict['gt_boxes'],
                                                    first_dict = attack_dict,
                                                    second_dict = pred_dicts[0])
            
            regular_loss = dataset.universal_adv_patch_car.get_regularization_loss()
            total_loss = adversarial_loss + args.laplacian_weights * regular_loss

            optimizer.zero_grad()
            model.zero_grad()
            total_loss.backward()

        else:
            raise NotImplementedError
        
        if verbose_epoch > 0 and i % verbose_epoch == 0:
            logger.info("-----------------loss--------------")
            logger.info(f"total loss:{total_loss.item()}")
            logger.info(f"adversarial loss:{adversarial_loss.item()}")
            logger.info(f"regular loss:{regular_loss.item()}")
            logger.info("-----------------gradient--------------")
            logger.info(f"(1):{dataset.universal_adv_patch_car.get_parameters()[0].grad}")
            logger.info(f"(2):{dataset.universal_adv_patch_car.get_parameters()[1].grad}")
            logger.info(f"(3):{dataset.universal_adv_patch_car.get_parameters()[2].grad}")

            pdb.set_trace()
            if args.visualize:
                try:
                    import open3d
                    from visual_utils import open3d_vis_utils as V
                    OPEN3D_FLAG = True
                except:
                    import mayavi.mlab as mlab
                    from visual_utils import visualize_utils as V
                    OPEN3D_FLAG = False
                V.draw_scenes(
                    points=batch_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'].detach(),
                    ref_scores=pred_dicts[0]['pred_scores'].detach(), ref_labels=pred_dicts[0]['pred_labels'].detach(), gt_boxes=batch_dict['gt_boxes'][0]
                )
            
            
            
        if update:
            """
                set grad along z-axi to 0.
            """
            dataset.universal_adv_patch_car.constrain_grad()
            
            if args.OPTIM == "rbboxloss":
                optimizer.step()
            else:
                raise NotImplementedError
            

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
    args = argparse.ArgumentParser(description='KITTI Attack Test')
    args.add_argument('--UNI_RANDOM_SEED', type=int, default=2024, help='random seed')
    args.add_argument('--device', type=int, default=0, help='device')
    args.add_argument('--EVAL_OUTPUT_DIR', type=str, default="./eval_output/", help='evaluation output directory')
    
    args.add_argument('--cfg-file', type=str, default="configs/attack_configs/relevant_bounding_box_pointpillar.yaml", help='configuration file')
    
    args.add_argument('--BATCH_SIZE', type=int, default=1, help='batch size')
    args.add_argument('--WORKERS', type=int, default=4, help='workers')
    args.add_argument('--DIST_TEST', action='store_true', help='distributed test')
    args.add_argument('--OPTIM', type=str, default="rbboxloss", help='optimization method')
    args.add_argument('--headbox-attack', action='store_true', help='enable headbox attack')
    args.add_argument('--roihead-attack', action='store_true', help='enable roihead attack')

    args.add_argument('--eval-clean-data', action='store_true', help='evaluate clean data (only)')
    args.add_argument('--eval-init-patch', action='store_true', help='evaluate initial patch (only)')
    args.add_argument('--CHECK_GRAD_QUAD', action='store_true', help='check grad quad')
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
    
    args.add_argument('--mode', type=int, default=0)


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


def main():
    ### Setting the parameters and logger
    args, cfg = parse_config()
    
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
    ### Load the parameters of annotated rooftop 

    ### Build the dataloader
    test_set, test_loader, sampler = build_dataloader(
            dataset_cfg=dataset_cfg,
            class_names=cfg.CLASS_NAMES,
            batch_size=args.BATCH_SIZE,
            dist=args.DIST_TEST,
            workers=args.WORKERS,
            logger=logger,
            training=False
        )
    
    logger.info(f'Class names of samples: \t{test_set.class_names}')
    
    ### Build the neural network and load the checkpoint
    model = build_network(model_cfg=model_cfg,
                          num_class=len(cfg.CLASS_NAMES),
                          dataset=test_set)
    
    model.load_params_from_file(filename=model_cfg.CKPT_PATH,
                                logger=logger,
                                to_cpu=True)
    model.cuda()
    model.eval()

    components_of_model(model, logger)

    ### Prepare the adversarial dataset, which contains get_gradients methods and so forth, and optimizer oriented patch 
    lidar = LiDAR_base(origin=torch.tensor([0.0, 0.0, 0.0]).cuda(),
                   azi_range=[-90, 90],
                   polar_range= [-2.18, 2.0],
                   polar_num=10, azi_res=0.08)
    kitti_adv_dataset = adv_dataset(test_set,
                                    attack_cfg,
                                    surrogate_model=None,
                                    lidar = lidar,
                                    enable_car = True,
                                    enable_ped = False,
                                    enable_bicycle = False,
                                    car_adv_patch_scale = args.scale,
                                    car_adv_patch_level = args.level,
                                    )
    logger.info(f"parameter length:\t{kitti_adv_dataset.get_adversarial_parameter().__len__()}")

    
    if args.eval_clean_data:
        logger.info("Evaluate the clean data")
        eval_data(args = args, 
                cfg = cfg,
                adv_enabled = False, 
                model = model, 
                dataset = kitti_adv_dataset, 
                logger = logger)
        return
    
    if args.eval_init_patch:
        logger.info("Evaluate the initial patch")
        eval_data(args = args, 
                cfg = cfg,
                adv_enabled = True, 
                model = model, 
                dataset = kitti_adv_dataset, 
                logger = logger)
        return
    
    ### Evaluate patch attack with one epoch
    kitti_adv_dataset.save_adversarial_parameter(os.path.join(args.SAVE_PATH, "initial_patch_checkpoint.pt"))
    logger.info(f'Checkpoint has been saved as: \t{os.path.join(args.SAVE_PATH, "initial_patch_checkpoint.pt")}')

    run_one_epoch_attack(args,
                dataset = kitti_adv_dataset,
                model = model,
                enable_adv = True, 
                update = True, 
                visualize = True, 
                logger = logger,
                verbose_epoch = args.verbose_epoch)

    kitti_adv_dataset.save_adversarial_parameter(os.path.join(args.SAVE_PATH, "final_adversarial_patch_checkpoint.pt"))
    logger.info(f'Checkpoint has been saved as: \t{os.path.join(args.SAVE_PATH, "final_adversarial_patch_checkpoint.pt")}')
    ### Visualize the adversarial examples
    # vis_adv_examples(kitti_adv_dataset)

    ### Evaluate the adversarial examples
    logger.info("Evaluate the final patch")
    eval_data(args = args, 
            cfg = cfg,
            adv_enabled = True, 
            model = model, 
            dataset = kitti_adv_dataset, 
            logger = logger)

if __name__ == "__main__":
    main()