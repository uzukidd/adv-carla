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

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

from cudaext.ops.Rotated_IoU.oriented_iou_loss import cal_iou_3d, assign_target_3d

from raytorch.LiDAR import LiDAR_base
from pytorch3d.vis.plotly_vis import plot_scene
from pytorch3d.io import IO

from pcdet.datasets.kitti.kitti_dataset import create_kitti_infos
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import KittiDataset, build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

from data_tools import adv_dataset, kitti_carla_dataset
from eval_utils import eval_utils
from loss_utils import mesh_objectwise_loss, relevant_bounding_box_loss
from optim_utils import objectwise_deepfool

import pdb
import argparse
import os
import time


def roipooling_grad_mapping(pooled_features_grad, batch_point_features, pooled_pts_idx):
    
    batch_size = batch_point_features.size(0)
    npoint = batch_point_features.size(1)
    feature_size = batch_point_features.size(2)
    
    batch_point_features_grad = torch.zeros_like(batch_point_features)
    xyz_features_grad = batch_point_features.new_zeros((batch_size, npoint, 3))
    
    for batch_mask in range(0, batch_size):
        pts_idx_expanded = pooled_pts_idx[batch_mask].view(-1).long().unsqueeze(1).expand(-1, feature_size)
        xyz_pooled_features_grad_viewed = pooled_features_grad[:, :, :3].view(-1, 3)
        pooled_features_grad_viewed = pooled_features_grad[:, :, 3:].view(-1, feature_size)

        batch_point_features_grad[batch_mask].scatter_add_(0, pts_idx_expanded, pooled_features_grad_viewed)
        xyz_features_grad[batch_mask].scatter_add_(0, pts_idx_expanded[:, :3], xyz_pooled_features_grad_viewed)
    
    return xyz_features_grad, batch_point_features_grad

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


def evaluate_one_epoch_attack(args, enable_adv, update, visualize,
                              verbose_epoch: int = 100,
                              grad_cache = None):
    
    car_headbox_rbbox_loss_func = relevant_bounding_box_loss(frozen_iou = args.iou_frozen,
                 frozen_logit = args.logit_frozen,
                 confidence_threshold = 0.1,
                 iou_threshold = 0.1, 
                 verbose = True)
    
    car_roihead_rbbox_loss_func = relevant_bounding_box_loss(frozen_iou = args.iou_frozen,
                 frozen_logit = args.logit_frozen,
                 confidence_threshold = 0.1,
                 iou_threshold = 0.1, 
                 verbose = True)

    ped_rbbox_loss_func = relevant_bounding_box_loss(frozen_iou = args.iou_frozen,
                    frozen_logit = args.logit_frozen,
                    confidence_threshold = 0.1,
                    iou_threshold = 0.01, 
                    verbose = True)

    
    kitti_adv_dataset.enable_adversarial_patch(enable_adv)
    
    for i, batch_dict in tqdm(enumerate(kitti_adv_dataset), total=kitti_adv_dataset.__len__()):
        load_data_to_gpu(batch_dict)
        
        if not torch.eq(batch_dict['gt_boxes'][0, :, 7], 1).any():
            # logger.info(f"no vehicles found in batch \t{i}")
            continue
        
        model.eval()
        model.zero_grad()
        pred_dicts, _ = model(batch_dict)
        point_headbox_ret_dict = point_headbox.forward_ret_dict
        pointrcnn_head_ret_dict = pointrcnn_head.forward_ret_dict
        
        # if args.CHECK_GRAD_QUAD:
        #     n_size = batch_dict["gt_boxes"].size(1)
        #     grad_cache.append([])
        #     for n_idx_mask in range(n_size):
        #         mesh_proposal_loss:torch.Tensor = rbbox_loss_func(batch_dict = point_headbox_ret_dict, 
        #                                 gt_boxes = batch_dict["gt_boxes"][:, n_idx_mask:n_idx_mask+1, :], 
        #                                 target_class = 1,
        #                                 logit_normal = "sigmoid",
        #                                 ret_part_loss = False)

        #         optimizer.zero_grad()
        #         model.zero_grad()
        #         if mesh_proposal_loss.requires_grad:
        #             mesh_proposal_loss.backward(retain_graph = True)
        #             grad_cache[-1].append([grad.cpu().numpy() for grad in kitti_adv_dataset.universal_adv_patch.get_mesh_gradient()])
            
        if args.OPTIM == "rbboxloss":
            mesh_loss = torch.zeros(1).cuda()
            if args.headbox_attack:
                car_mesh_proposal_loss = car_headbox_rbbox_loss_func(batch_dict = point_headbox_ret_dict, 
                                        gt_boxes = batch_dict["gt_boxes"], 
                                        target_class = 1,
                                        logit_normal = "sigmoid",
                                        input_type = "headbox",
                                        ret_part_loss = False)
                mesh_loss = mesh_loss + car_mesh_proposal_loss
                
            if args.roihead_attack:
                car_mesh_roi_loss = car_roihead_rbbox_loss_func(batch_dict = pointrcnn_head_ret_dict, 
                                        gt_boxes = batch_dict["gt_boxes"], 
                                        target_class = 1,
                                        logit_normal = "sigmoid",
                                        input_type = "roihead",
                                        ret_part_loss = False)
                mesh_loss = mesh_loss + args.roi_head_weights * car_mesh_roi_loss
                
            regular_loss = kitti_adv_dataset.universal_adv_patch_car.get_laplacian_loss()
            total_loss = mesh_loss + args.laplacian_weights * regular_loss
            
            optimizer.zero_grad()
            model.zero_grad()
            total_loss.backward()
            
            assert not torch.isnan(total_loss).any()
            
            if pointrcnn_head.pooled_features.grad is not None:
                
                if torch.isnan(pointrcnn_head.pooled_features.grad).any():
                    pdb.set_trace()
            #     xyz_features_grad, batch_point_features_grad = roipooling_grad_mapping(pointrcnn_head.pooled_features.grad, 
            #                                                                         pointrcnn_head.batch_point_features, 
            #                                                                         pointrcnn_head.pooled_pts_idx)
            #     try:
            #         pointrcnn_head.batch_point_features.backward(batch_point_features_grad, retain_graph = True)
            #         batch_dict['points'][None, :, 1:4].backward(xyz_features_grad, retain_graph = True)
            #     except RuntimeError as e:
            #         pass

        else:
            raise NotImplementedError
        
        if verbose_epoch > 0 and i % verbose_epoch == 0:
            # logger.info(f"deformed verts of mesh: \t{kitti_adv_dataset.universal_adv_patch.get_mesh_deform_vert()}")
            # vert_grad, translate_grad, theta_grad = kitti_adv_dataset.universal_adv_patch_car.get_mesh_gradient()
            # logger.info(f"deformed vert gradients of mesh: \t{vert_grad}")

            
            if visualize:

                V.draw_scenes(
                    points=batch_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'].detach(),
                    ref_scores=pred_dicts[0]['pred_scores'].detach(), ref_labels=pred_dicts[0]['pred_labels'].detach(), gt_boxes=batch_dict['gt_boxes'][0]
                )
            
        if update:
            """
                set grad along z-axi to 0.
            """
            kitti_adv_dataset.universal_adv_patch_car.constrain_z_grad()
            
            # vert_grad, translate_grad, theta_grad = kitti_adv_dataset.universal_adv_patch_ped.get_mesh_gradient()
            # vert_grad[:, 2] = 0.
            # translate_grad[2] = 0.
            
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

    # logger.info(f"deformed verts of mesh: \t{kitti_adv_dataset.universal_adv_patch.get_mesh_deform_vert()}")
    # logger.info(f"theta of mesh: \t{kitti_adv_dataset.universal_adv_patch.theta}")
    # logger.info(f"theta of global_translation: \t{kitti_adv_dataset.universal_adv_patch.global_translation}")


def save_grad_cache(grad_cache):
    with open("./check_grad.pkl", "wb") as output:
        pkl.dump(grad_cache, output)


def whether_eval_init_patch(args, cfg, model, kitti_adv_dataset, logger):
    if args.EVAL_INIT_PATH:
        logger.info("Evaluate the initial patch")
        kitti_adv_dataset.enable_adversarial_patch(True)
        eval_utils.eval_one_epoch(
                cfg,
                args = None,
                model = model,
                dataloader = kitti_adv_dataset,
                epoch_id = 0,
                logger = logger,
                dist_test = args.DIST_TEST,
                result_dir = Path(args.EVAL_OUTPUT_DIR),
                infer_time = True
            )
        return True
    else:
        logger.info("Skip the evaluation of initial patch")


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
    
    return rooftop_approximate
    


def parse_config():
    ### Set hyperparameters, including random seed, device, dataset path, etc.
    args = argparse.ArgumentParser(description='KITTI Attack Test')
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
    args.add_argument('--OPTIM', type=str, default="rbboxloss", help='optimization method')
    args.add_argument('--headbox-attack', action='store_true', help='enable headbox attack')
    args.add_argument('--roihead-attack', action='store_true', help='enable roihead attack')

    args.add_argument('--EVAL_INIT_PATH', action='store_true', help='evaluation initial path')
    args.add_argument('--CHECK_GRAD_QUAD', action='store_true', help='check grad quad')
    args.add_argument('--roi-head-weights', type=float, default=1.0, help='roi head weights')
    args.add_argument('--laplacian_weights', type=float, default=0.001, help='laplacian weights')
    args.add_argument('--learning_rate', type=float, default=0.005, help='learning rate')
    args.add_argument('--iou-frozen', action='store_true', help='freeze iou loss while optimization')
    args.add_argument('--logit-frozen', action='store_true', help='freeze logit loss while optimization')
    args.add_argument('--exp-name', type=str, default=str(int(time.time())), help='name of saving folder')
    args.add_argument('--verbose-epoch', type=int, default=-1, help='verbose per epoch')

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
    
    args.SAVE_PATH = f"output/train/{args.exp_name}"
    os.makedirs(args.SAVE_PATH)
    logger = common_utils.create_logger(log_file = os.path.join(args.SAVE_PATH, "exp_log.log"))
    logger.info('-----------------Kitti Attack Test-------------------------')
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
                                    enable_bicycle = False,
                                    enable_double = True,
                                    cubic_level=1)
    
    optimizer = optim.Adam(kitti_adv_dataset.get_adversarial_parameter(), 
                       lr=args.learning_rate)
    
    ### Whether to evaluate the initial patch
    whether_eval_init_patch(args, cfg, model, kitti_adv_dataset, logger)
    
    ### Evaluate patch attack with one epoch
    kitti_adv_dataset.save_adversarial_parameter(os.path.join(args.SAVE_PATH, "initial_patch_checkpoint.pt"))
    logger.info(f'Checkpoint has been saved as: \t{os.path.join(args.SAVE_PATH, "initial_patch_checkpoint.pt")}')


    grad_cache = []
    grad_cache = evaluate_one_epoch_attack(args,
                            enable_adv = True, 
                            update = True, 
                            visualize = True, 
                            verbose_epoch = args.verbose_epoch,
                            grad_cache = grad_cache)
    
    ### Save the grad cache
    save_grad_cache(grad_cache)
    kitti_adv_dataset.save_adversarial_parameter(os.path.join(args.SAVE_PATH, "final_adversarial_patch_checkpoint.pt"))
    logger.info(f'Checkpoint has been saved as: \t{os.path.join(args.SAVE_PATH, "final_adversarial_patch_checkpoint.pt")}')
    ### Visualize the adversarial examples
    # vis_adv_examples(kitti_adv_dataset)

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

