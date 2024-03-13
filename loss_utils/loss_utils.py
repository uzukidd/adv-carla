UNI_RANDOM_SEED = 2024

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb
from tqdm import tqdm
from pathlib import Path
import pdb

from cudaext.ops.Rotated_IoU.oriented_iou_loss import cal_iou_3d
from cudaext.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu

class mesh_objectwise_loss(nn.Module):
    def __init__(self, freezed_iou: bool = False, normalized: bool = False, verbose: bool = False):
        super().__init__()
        self.freezed_iou = freezed_iou
        self.normalized = normalized
        self.verbose = verbose
    
    def objectwise_loss(self, cls_pred: torch.Tensor, box_preds: torch.Tensor, gt_box: torch.Tensor, target_class_logit: int):
        cls_preds_softmax = F.softmax(cls_pred, dim=1) # [N, 3]
        logit, pred_classes = cls_preds_softmax.max(dim=-1)
        pred_classes = pred_classes
        target_idx_mask = (pred_classes == target_class_logit)
        
        if target_idx_mask.any():
            max_logit, max_logit_idx = cls_preds_softmax[target_idx_mask, target_class_logit].max(dim=-1)
            box_selected = box_preds[target_idx_mask][max_logit_idx]
            
            iou_3d = cal_iou_3d(box_selected.view(1, 1, -1), gt_box.view(1, 1, -1))
        else:
            max_logit = cls_pred.new_zeros(1)
            iou_3d = cls_pred.new_zeros(1)
        
        return max_logit.view(1), iou_3d.view(1)

    def forward(self, batch_dict, point_coords, gt_boxes: torch.Tensor, target_class: int, ret_part_loss: bool = False):
        """

        Args:
        - batch_cls_preds: [N, 3]
        - batch_box_preds: [N, 7]
        - gt_boxes: [1, N, 8]
        - point_coords: [N, 3]
        - ret_part_loss: bool

        Returns:
        - iou_3d: 
        - assign_idx: 
        - mesh_loss:
        """
        batch_cls_preds = batch_dict["batch_cls_preds"] # [N, 3]
        batch_box_preds = batch_dict["batch_box_preds"] # [N, 7]
        
        gt_boxes, gt_labels = torch.split(gt_boxes.squeeze(dim=0), [7, 1], dim=1)  # [N, 7], [N, 1]
        gt_boxes = gt_boxes[gt_labels[:, 0] == target_class]
        gt_labels = gt_labels[gt_labels[:, 0] == target_class]
        
        target_class_logit = target_class - 1

        assert gt_boxes.size(0), "at leaset one gt box exists."
        
        box_idxs_of_pts = points_in_boxes_gpu(
                point_coords.unsqueeze(dim=0), gt_boxes.unsqueeze(dim=0)
            ).long().squeeze(dim=0)
        
        total_loss = []
        max_logit_item = []
        iou_3d_item = []
        
        for box_idx_mask in range(gt_boxes.size(0)):
            pts_idx_mask = (box_idxs_of_pts == box_idx_mask)
            max_logit, iou_3d = self.objectwise_loss(cls_pred = batch_cls_preds[pts_idx_mask], 
                                 box_preds = batch_box_preds[pts_idx_mask],
                                 gt_box = gt_boxes[box_idx_mask],
                                 target_class_logit = target_class_logit)
            if self.freezed_iou:
                iou_3d = iou_3d.detach()
            
            object_loss = -1.0 * torch.log(1.0 - max_logit)
            max_logit_item.append(max_logit)
            iou_3d_item.append(iou_3d)
            total_loss.append(object_loss)
        
        iou_3d_item = torch.stack(iou_3d_item)
        
        if self.normalized:
            iou_3d_item = iou_3d_item / iou_3d_item.sum()
        
        total_loss = torch.stack(total_loss)
        total_loss = total_loss * iou_3d_item
        total_loss = total_loss.sum()
  
        
        if ret_part_loss:
            max_logit_item = torch.stack(max_logit_item).sum()
            iou_3d_item = iou_3d_item.sum()
            return total_loss, max_logit_item, iou_3d_item
        else:
            return total_loss