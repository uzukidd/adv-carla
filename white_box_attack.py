import torch
import torch.nn as nn
import torch.optim as optim

import pdb
from tqdm import tqdm
torch.autograd.set_detect_anomaly(True)

import loss_utils.loss_reduce_func as loss_reduce_func
from data_tools import adv_dataset
from loss_utils import relevant_bounding_box_loss

import pdb


class stage_wise_full_attack(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.car_rbbox_loss_func:relevant_bounding_box_loss = relevant_bounding_box_loss(frozen_iou = args.iou_frozen,
                 frozen_logit = args.logit_frozen,
                 confidence_threshold = 0.1,
                 iou_threshold = 0.1, 
                 verbose = False)
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
        
    
    def forward(self, gt_boxes, first_dict = None, second_dict = None):
        mesh_loss = torch.zeros(1, requires_grad=True).cuda()
        if self.args.headbox_attack:
            car_mesh_proposal_loss, empty_flag = self.car_rbbox_loss_func(batch_dict = first_dict, 
                                    gt_boxes = gt_boxes, 
                                    target_class = 1,
                                    logit_normal = "sigmoid",
                                    detector_type = "rcnn",
                                    loss_reduce_func = self.stage_1_loss_reduce_func)
            if not empty_flag:
                mesh_loss = mesh_loss + car_mesh_proposal_loss
            
        if self.args.roihead_attack:
            car_mesh_single_loss, empty_flag = self.car_rbbox_loss_func(batch_dict = second_dict, 
                                gt_boxes = gt_boxes, 
                                target_class = 1,
                                logit_normal = "sigmoid",
                                detector_type = "single",
                                loss_reduce_func = self.stage_2_loss_reduce_func)
            if not empty_flag:
                mesh_loss = mesh_loss + self.args.roi_head_weights * car_mesh_single_loss
    
        return mesh_loss
    
def run_one_epoch_white_box_attack(args, 
                        dataset: adv_dataset, 
                        model, 
                        enable_adv, 
                        update, 
                        visualize,
                        logger,
                        verbose_epoch: int = 100):
    
    adversarial_loss_func = stage_wise_full_attack(args)
    
    optimizer = None
    if args.optim == "adam":
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
        # import random
        # batch_dict = dataset.__getitem__(923)
        if not torch.eq(batch_dict['gt_boxes'][0, :, 7], 1).any():
            continue
        
        model.zero_grad()
        pred_dicts, _ = model(batch_dict)
        
        attack_dict = None
        if dataset.detector_type == "single":
            attack_dict = pred_dicts[0]
        elif dataset.detector_type == "rcnn":
            attack_dict = target_component.forward_ret_dict
    
        
        adversarial_loss = adversarial_loss_func(batch_dict['gt_boxes'],
                                                first_dict = attack_dict,
                                                second_dict = pred_dicts[0])
        
        regular_loss = dataset.universal_adv_patch_car.get_regularization_loss()
        total_loss = adversarial_loss + args.laplacian_weights * regular_loss
        
        if optimizer:
            optimizer.zero_grad()
        model.zero_grad()
        total_loss.backward()

        
        if verbose_epoch > 0 and i % verbose_epoch == 0:
            logger.info("-----------------loss--------------")
            logger.info(f"total loss:{total_loss.item()}")
            logger.info(f"adversarial loss:{adversarial_loss.item()}")
            logger.info(f"regular loss:{regular_loss.item()}")
            logger.info("-----------------gradient--------------")
            logger.info(f"(1):{dataset.universal_adv_patch_car.get_parameters()[0].grad}")
            logger.info(f"(2):{dataset.universal_adv_patch_car.get_parameters()[1].grad}")
            logger.info(f"(3):{dataset.universal_adv_patch_car.get_parameters()[2].grad}")

            
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
            else:
                pdb.set_trace()
            
            
        if update:
            """
                set grad along z-axi to 0.
            """
            dataset.universal_adv_patch_car.constrain_grad()
            
            if args.optim == "adam":
                optimizer.step()
            elif args.optim == "ifgsm":
                original_parameter = [
                    para.clone().detach() for para in dataset.universal_adv_patch_car.get_parameters()
                ]
                
                global_translation_grad, theta_grad, vert_grad = (para.grad for para in dataset.universal_adv_patch_car.get_parameters())
                
                if global_translation_grad is None:
                    continue
                global_translation_grad_norm = torch.linalg.norm(global_translation_grad)
                global_translation_grad_norm[global_translation_grad_norm < 1e-5] = 1e-5
                global_translation_grad_norm = global_translation_grad / global_translation_grad_norm
                
                vert_grad_norm = torch.linalg.norm(vert_grad, dim=1)
                vert_grad_norm[vert_grad_norm < 1e-5] = 1e-5
                vert_grad_norm = vert_grad / vert_grad_norm[:, None]
                
                theta_grad_norm = torch.sign(theta_grad)
                
                cur_parameter = [original_parameter[0] + args.learning_rate * global_translation_grad_norm,
                        original_parameter[1] + args.learning_rate * theta_grad_norm,
                        original_parameter[2] + args.learning_rate * vert_grad_norm,]
                
                dataset.universal_adv_patch_car.load_parameter(cur_parameter)
            else:
                raise NotImplementedError
            
            
# class loss_wise_full_attack(nn.Module):
    
#     def __init__(self, args, ):
#         super().__init__()
#         self.args = args
#         self.car_rbbox_loss_func = relevant_bounding_box_loss(frozen_iou = args.iou_frozen,
#                  frozen_logit = args.logit_frozen,
#                  confidence_threshold = 0.1,
#                  iou_threshold = 0.1, 
#                  verbose = False)
    
#     def forward(self, gt_boxes, first_dict = None, second_dict = None):
#         mesh_loss = None
#         if self.args.mode < -2:
#             iou3d, masked_cls_preds, masked_cls_preds_extended = self.car_rbbox_loss_func(batch_dict = first_dict, 
#                                     gt_boxes = gt_boxes, 
#                                     target_class = 1,
#                                     logit_normal = "None",
#                                     detector_type = "rcnn",
#                                     ret_part_loss = True)
            
#             if self.args.mode == -3:
#                 mesh_loss = (iou3d * masked_cls_preds_extended).sum()
#         if self.args.mode < 2:
#             iou3d, masked_cls_preds, masked_cls_preds_extended = self.car_rbbox_loss_func(batch_dict = first_dict, 
#                                     gt_boxes = gt_boxes, 
#                                     target_class = 1,
#                                     logit_normal = "sigmoid",
#                                     detector_type = "rcnn",
#                                     ret_part_loss = True)
#             if self.args.mode == 0:
#                 mesh_loss = iou3d.sum()
#             elif self.args.mode == 1:
#                 mesh_loss = masked_cls_preds.sum()
#             elif self.args.mode == -1:
#                 mesh_loss = -1.0 * torch.log(1.0 - masked_cls_preds).sum()
#             elif self.args.mode == -2:
#                 mesh_loss = (iou3d * masked_cls_preds_extended).sum()

#         elif self.args.mode >= 2:
#             iou3d, masked_cls_preds, masked_cls_preds_extended = self.car_rbbox_loss_func(batch_dict = second_dict, 
#                                 gt_boxes = gt_boxes, 
#                                 target_class = 1,
#                                 logit_normal = "sigmoid",
#                                 detector_type = "single",
#                                 ret_part_loss = True)
#             if self.args.mode == 2:
#                 mesh_loss = iou3d.sum()
#             elif self.args.mode == 3:
#                 mesh_loss = masked_cls_preds.sum()
#             elif self.args.mode == 4:
#                 mesh_loss = torch.sigmoid(masked_cls_preds).sum()
#             elif self.args.mode == 5:
#                 mesh_loss = (torch.sigmoid(masked_cls_preds_extended) * iou3d).sum()
#         return mesh_loss