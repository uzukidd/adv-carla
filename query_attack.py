import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

from attack_utils import *
from data_tools import adv_dataset
from eval_utils import eval_utils
from loss_utils import relevant_bounding_box_loss
import loss_utils.loss_reduce_func as loss_reduce_func

import argparse
import os
import time

class query_attack(nn.Module):
    
    def __init__(self, 
                 args, 
                 cfg,
                 surrogate_dataset: adv_dataset, 
                 victim_dataset: adv_dataset, 
                 surrogate_model:nn.Module,
                 victim_model:nn.Module,
                 target_component_name:str = None,):
        super().__init__()
        self.args = args
        self.cfg = cfg
        self.surrogate_dataset = surrogate_dataset
        self.victim_dataset = victim_dataset
        self.surrogate_model = surrogate_model
        self.victim_model = victim_model
        self.car_rbbox_loss_func = relevant_bounding_box_loss(frozen_iou = True,
                 frozen_logit = False,
                 confidence_threshold = 0.1,
                 iou_threshold = 0.1, 
                 verbose = False)
        self.target_component = None
        
        self.stage_1_loss_reduce_func = None
        try:
            self.stage_1_loss_reduce_func = getattr(loss_reduce_func, args.stage_1_loss_reduce_func)
        except AttributeError:
            pass
        
        self.stage_2_loss_reduce_func = None
        try:
            self.stage_2_loss_reduce_func = getattr(loss_reduce_func, args.stage_2_loss_reduce_func)
        except AttributeError:
            pass        
        
        if self.args.surrogate_stage_1:
            assert target_component_name is not None
            for idx, module in enumerate(self.surrogate_model.module_list):
                if module._get_name() == target_component_name:
                    self.target_component = module

            
        
    def black_box_forward(self, 
                        index,
                        adversarial_parameters):
        
        batch_dict = self.victim_dataset.__getitem__(index,
                                        adversarial_parameters = adversarial_parameters)
        gt_boxes = batch_dict['gt_boxes']
        
        pred_dicts, _ = self.victim_model(batch_dict)
        car_mesh_single_loss, empty_flag = self.car_rbbox_loss_func(batch_dict = pred_dicts[0], 
                                gt_boxes = gt_boxes, 
                                target_class = 1,
                                logit_normal = "sigmoid",
                                detector_type = "single",
                                loss_reduce_func = self.stage_2_loss_reduce_func)
        if empty_flag:
            return None
        return car_mesh_single_loss.item()
    
    def forward(self, 
                index):
        surrogate_batch_dict = self.surrogate_dataset.__getitem__(index)
        
        gt_boxes = surrogate_batch_dict['gt_boxes']
        if not torch.eq(gt_boxes[0, :, 7], 1).any():
            return
        
        original_parameter = [
            para.clone().detach() for para in self.surrogate_dataset.get_adversarial_parameter()
        ]
        self.surrogate_model.zero_grad()
        pred_dicts, _ = self.surrogate_model(surrogate_batch_dict)
        if self.args.surrogate_stage_1:
            car_mesh_loss, empty_flag = self.car_rbbox_loss_func(batch_dict = self.target_component.forward_ret_dict, 
                                    gt_boxes = gt_boxes, 
                                    target_class = 1,
                                    logit_normal = "sigmoid",
                                    detector_type = "rcnn",
                                    loss_reduce_func = self.stage_1_loss_reduce_func)
        else:
            car_mesh_loss, empty_flag = self.car_rbbox_loss_func(batch_dict = pred_dicts[0], 
                                    gt_boxes = gt_boxes, 
                                    target_class = 1,
                                    logit_normal = "sigmoid",
                                    detector_type = "single",
                                    loss_reduce_func = self.stage_2_loss_reduce_func)

        if empty_flag:
            return
        regular_loss = self.surrogate_dataset.universal_adv_patch_car.get_regularization_loss()
        total_loss = car_mesh_loss + self.args.laplacian_weights * regular_loss
        total_loss.backward()
        global_translation_grad, theta_grad, vert_grad = (para.grad for para in self.surrogate_dataset.universal_adv_patch_car.get_parameters())
        
        if global_translation_grad is None:
            return
        
        best_loss = self.black_box_forward(index,
                               original_parameter,)
        
        if best_loss is None:
            return
        
        global_translation_grad[2] = 0.
        vert_grad[:, 2] = 0.
        
        global_translation_grad_norm = torch.linalg.norm(global_translation_grad)
        global_translation_grad_norm[global_translation_grad_norm < 1e-5] = 1e-5
        global_translation_grad_norm = global_translation_grad / global_translation_grad_norm
        
        vert_grad_norm = torch.linalg.norm(vert_grad, dim=1)
        vert_grad_norm[vert_grad_norm < 1e-5] = 1e-5
        vert_grad_norm = vert_grad / vert_grad_norm[:, None]
        theta_grad_norm = torch.sign(theta_grad)


        for eps in [-self.args.learning_rate, self.args.learning_rate]:
        
            cur_parameter = [original_parameter[0] + eps * global_translation_grad_norm,
                            original_parameter[1] + eps * theta_grad_norm,
                            original_parameter[2] + eps * vert_grad_norm,]
            cur_loss = self.black_box_forward(index,
                                cur_parameter)
            
            if cur_loss is None:
                return

            if cur_loss < best_loss:
                best_loss = cur_loss
                self.surrogate_dataset.universal_adv_patch_car.load_parameter(cur_parameter)
                self.victim_dataset.universal_adv_patch_car.load_parameter(cur_parameter)
        

        # loss = self.CWLoss(logits, target, kappa=-999., tar=True, num_classes=self.num_class)
        # self.wb_classifier.zero_grad()
        # loss.backward()

        # grad = new_points.grad.data # g, [1, N, 3]
        # grad[:,:,2] = 0.
        # new_points.requires_grad = False
        # rankings = torch.sqrt(grad[:,:,0] ** 2 + grad[:,:,1] ** 2) # \sqrt{g_{x'}^2+g_{y'}^2}, [1, N]
        # directions = grad / (rankings.unsqueeze(-1)+1e-16) # (g_{x'}/r,g_{y'}/r,0), [1, N, 3]

        # # rank the sensitivity map in the desending order
        # point_list = []
        # for i in range(points.size(1)):
        #     point_list.append((i, directions[:,i,:], rankings[:,i].item()))
        # sorted_point_list = sorted(point_list, key=lambda c: c[2], reverse=True)

        # # query loop
        # i = 0
        # best_loss = -999.
        # while best_loss < 0 and i < len(sorted_point_list):
        #     # print(i, len(sorted_point_list))
        #     idx, direction, _ = sorted_point_list[i]
        #     for eps in {self.step_size, -self.step_size}:
        #         pert = torch.zeros_like(new_points).cuda()
        #         pert[:,idx,:] += eps * direction
        #         inputs = new_points + pert
        #         inputs = torch.matmul(spin_axis_matrix.transpose(-1, -2), inputs.unsqueeze(-1)) # U^T P', [1, N, 3, 1]
        #         inputs = inputs - translation_matrix.unsqueeze(-1) # P = U^T P' - (P \cdot N) N, [1, N, 3, 1]
        #         inputs = inputs.squeeze(-1).transpose(1, 2) # P, [1, 3, N]
        #         # inputs = torch.clamp(inputs, -1, 1)
        #         with torch.no_grad():
        #             if not self.defense_method is None:
        #                 logits = self.classifier(self.pre_head(inputs.detach()))
        #             else:
        #                 logits = self.classifier(inputs.detach()) # [1, num_class]
        #             query_costs += 1
                    
        #         loss = self.CWLoss(logits, target, kappa=-999., tar=True, num_classes=self.num_class)
        #         if loss.item() > best_loss:
        #             # print(loss.item())
        #             best_loss = loss.item()
        #             new_points = new_points + pert
        #             adv_target = logits.max(1)[1]
        #             break
        #     i += 1
    


def run_one_epoch_query_attack(args, 
                         surrogate_dataset: adv_dataset, 
                         victim_dataset: adv_dataset, 
                         surrogate_model,
                         victim_model, 
                         enable_adv, 
                         update, 
                         visualize,
                         logger,
                         verbose_epoch: int = 100):
    
    query_attack_func = query_attack(args=args,
                                     cfg=None,
                                     surrogate_dataset = surrogate_dataset,
                                     victim_dataset = victim_dataset,
                                     surrogate_model=surrogate_model,
                                     victim_model=victim_model,
                                     target_component_name=surrogate_dataset.target_component)

    surrogate_dataset.enable_adversarial_patch(enable_adv)
    victim_dataset.enable_adversarial_patch(enable_adv)
    
    surrogate_model.eval()
    victim_model.eval()

    for idx, module in enumerate(surrogate_model.module_list):
        if module._get_name() in surrogate_dataset.trainging_components:
            module.train()
            
    
    for index in tqdm(range(surrogate_dataset.__len__()), total=surrogate_dataset.__len__()):
        query_attack_func.forward(index)