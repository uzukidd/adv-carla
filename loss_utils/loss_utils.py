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

def extended_sigmoid(input: torch.Tensor):
    return 2.0 * torch.sigmoid(input) - 1.0

def inverse_extended_sigmoid(input: torch.Tensor):
    return torch.logit((input + 1.0)/2.0)

class mesh_objectwise_loss(nn.Module):
    """
        loss from https://arxiv.org/abs/2101.10747
    """
    def __init__(self, freezed_iou: bool = False, verbose: bool = False):
        super().__init__()
        self.freezed_iou = freezed_iou
        self.verbose = verbose
    
    def objectwise_loss(self, cls_pred: torch.Tensor, box_preds: torch.Tensor, gt_box: torch.Tensor, target_class_logit: int, reduction:str = 'max'):
        cls_preds_softmax = F.softmax(cls_pred, dim=1) # [N, 3]
        
        if reduction == "max":
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
        
        elif reduction == "sum":
            logit, pred_classes = cls_preds_softmax.max(dim=-1)
            pred_classes = pred_classes
            target_idx_mask = (pred_classes == target_class_logit)
            
            if target_idx_mask.any():
                logit = cls_preds_softmax[target_idx_mask, target_class_logit]
                box_selected = box_preds[target_idx_mask]
                iou_3d = cal_iou_3d(box_selected.view(1, -1, 7), 
                                    gt_box.view(1, 1, 7).expand((-1, box_selected.size(0), -1)))
            else:
                max_logit = cls_pred.new_zeros(1)
                iou_3d = cls_pred.new_zeros(1)
            
            return logit.view(-1), iou_3d.view(-1)
        elif reduction == "mean":
            pass
            raise NotImplementedError
        else:
            raise NotImplementedError

    def forward(self, batch_dict, point_coords, gt_boxes: torch.Tensor, target_class: int, reduction:str = 'max', ret_part_loss: bool = False):
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
        for box_idx_mask in range(gt_boxes.size(0)):
            pts_idx_mask = (box_idxs_of_pts == box_idx_mask)
            logit, iou_3d = self.objectwise_loss(cls_pred = batch_cls_preds[pts_idx_mask], 
                                 box_preds = batch_box_preds[pts_idx_mask],
                                 gt_box = gt_boxes[box_idx_mask],
                                 target_class_logit = target_class_logit, reduction=reduction)
            if self.freezed_iou:
                iou_3d = iou_3d.detach()
            
            object_loss = -1.0 * torch.log(1.0 - logit) * iou_3d
            if self.verbose:
                print(logit)
                print(iou_3d)
                print(object_loss)
            
            total_loss.append(object_loss.sum())
        
        total_loss = torch.stack(total_loss)
        
        if ret_part_loss:
            return total_loss
        else:
            return total_loss
        
class relevant_bounding_box_loss(nn.Module):
    """
        loss from https://arxiv.org/abs/2004.00543
    """
    def __init__(self, frozen_iou: bool = False,
                 frozen_logit: bool = False,
                 confidence_threshold: float = 0.1,
                 iou_threshold: float = 0.1, 
                 verbose: bool = False):
        super().__init__()
        self.frozen_iou = frozen_iou
        self.frozen_logit = frozen_logit
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.verbose = verbose
        
    def iou3d(self, batch_box_preds:torch.Tensor, 
                    gt_boxes: torch.Tensor):
        """
            Args:
            - batch_box_preds: [M, 7]
            - gt_boxes: [N, 7]
            
            Return:
            - iou3d: [M, N]
        """
        M = batch_box_preds.size(0)
        N = gt_boxes.size(0)

        box3d_roi_extended = batch_box_preds.view(M, 1, 7).expand(-1, N, -1)
        box3d_gt_extended = gt_boxes.view(1, N, 7).expand(M, -1, -1)
        # iou3d [M, N]
        iou3d = cal_iou_3d(box3d_roi_extended, box3d_gt_extended)
        
        return iou3d
        
        
    def forward(self, batch_dict, 
                point_coords, 
                gt_boxes: torch.Tensor, 
                target_class: int, 
                logit_normal:str = 'sigmoid', 
                ret_part_loss: bool = False):
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
        batch_cls_preds:torch.Tensor = batch_dict["batch_cls_preds"] # [N, 3]
        batch_box_preds:torch.Tensor = batch_dict["batch_box_preds"] # [N, 7]
        
        gt_boxes, gt_labels = torch.split(gt_boxes.squeeze(dim=0), [7, 1], dim=1)  # [N, 7], [N, 1]
        gt_boxes = gt_boxes[gt_labels[:, 0] == target_class]
        gt_labels = gt_labels[gt_labels[:, 0] == target_class]
        
        target_class_logit = target_class - 1

        assert gt_boxes.size(0), "at leaset one gt box exists."
        
        # assign 
        # batch_cls_preds_normalized = F.softmax(batch_cls_preds, dim=1) # [N, 3]
        if logit_normal == "sigmoid":
            batch_cls_preds_normalized = torch.sigmoid(batch_cls_preds) # [N, 3]
        elif logit_normal == "softmax":
            batch_cls_preds_normalized = F.softmax(batch_cls_preds, dim=1) # [N, 3]
        else:
            raise NotImplementedError
        confidence_mask = (batch_cls_preds_normalized[:, target_class_logit] > self.confidence_threshold)
        
        masked_cls_preds = batch_cls_preds_normalized[confidence_mask]
        masked_box_preds = batch_box_preds[confidence_mask]
        
        iou3d = self.iou3d(masked_box_preds, gt_boxes)
        
        masked_cls_preds_extended = masked_cls_preds.unsqueeze(dim=1)
        masked_cls_preds_extended = masked_cls_preds_extended.expand(-1, iou3d.size(1), -1)
        masked_cls_preds_extended = masked_cls_preds_extended.contiguous().view(-1, masked_cls_preds.size(1))
        
        masked_box_preds_extended = masked_box_preds.unsqueeze(dim=1)
        masked_box_preds_extended = masked_box_preds_extended.expand(-1, iou3d.size(1), -1)
        masked_box_preds_extended = masked_box_preds_extended.contiguous().view(-1, masked_box_preds.size(1))
        
        iou3d = iou3d.view(-1)
        if self.frozen_iou:
            iou3d = iou3d.detach()
            
        if self.frozen_logit:
            masked_cls_preds_extended = masked_cls_preds_extended.detach()
        
        iou_mask = (iou3d > self.iou_threshold)
        iou3d = iou3d[iou_mask]
        masked_cls_preds_extended = masked_cls_preds_extended[iou_mask]
        masked_box_preds_extended = masked_box_preds_extended[iou_mask]
        
        total_loss = -1.0 * torch.log(1.0 - masked_cls_preds_extended[:, target_class_logit]) * iou3d
        total_loss = total_loss.sum()
        
        return total_loss
        
        # if self.verbose:
        #     print(total_loss.size())

        
class relevant_refine_head_loss(nn.Module):
    """
        loss from https://arxiv.org/abs/2004.00543
    """
    def __init__(self, frozen_iou: bool = False,
                 frozen_logit: bool = False,
                 confidence_threshold: float = 0.1,
                 iou_threshold: float = 0.1, 
                 verbose: bool = False):
        super().__init__()
        self.frozen_iou = frozen_iou
        self.frozen_logit = frozen_logit
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.verbose = verbose
        
    def iou3d(self, batch_box_preds:torch.Tensor, 
                    gt_boxes: torch.Tensor):
        """
            Args:
            - batch_box_preds: [M, 7]
            - gt_boxes: [N, 7]
            
            Return:
            - iou3d: [M, N]
        """
        M = batch_box_preds.size(0)
        N = gt_boxes.size(0)

        box3d_roi_extended = batch_box_preds.view(M, 1, 7).expand(-1, N, -1)
        box3d_gt_extended = gt_boxes.view(1, N, 7).expand(M, -1, -1)
        # iou3d [M, N]
        iou3d = cal_iou_3d(box3d_roi_extended, box3d_gt_extended)
        
        return iou3d
        
        
    def forward(self, batch_dict,  
                gt_boxes: torch.Tensor, 
                target_class: int, 
                logit_normal:str = 'sigmoid', 
                ret_part_loss: bool = False):
        """

        Args:
        - batch_cls_preds: [N, 3]
        - batch_box_preds: [N, 7]
        - gt_boxes: [1, N, 8]
        - ret_part_loss: bool

        Returns:
        - iou_3d: 
        - assign_idx: 
        - mesh_loss:
        """
        batch_cls_preds:torch.Tensor = batch_dict["batch_cls_preds"] # [N, 3]
        batch_box_preds:torch.Tensor = batch_dict["batch_box_preds"] # [N, 7]
        
        gt_boxes, gt_labels = torch.split(gt_boxes.squeeze(dim=0), [7, 1], dim=1)  # [N, 7], [N, 1]
        gt_boxes = gt_boxes[gt_labels[:, 0] == target_class]
        gt_labels = gt_labels[gt_labels[:, 0] == target_class]
        
        target_class_logit = target_class - 1

        assert gt_boxes.size(0), "at leaset one gt box exists."
        
        # assign 
        if logit_normal == "sigmoid":
            batch_cls_preds_normalized = torch.sigmoid(batch_cls_preds) # [N, 3]
        elif logit_normal == "softmax":
            batch_cls_preds_normalized = F.softmax(batch_cls_preds, dim=1) # [N, 3]
        else:
            raise NotImplementedError
        confidence_mask = (batch_cls_preds_normalized[:, target_class_logit] > self.confidence_threshold)
        
        masked_cls_preds = batch_cls_preds_normalized[confidence_mask]
        masked_box_preds = batch_box_preds[confidence_mask]
        
        iou3d = self.iou3d(masked_box_preds, gt_boxes)
        
        masked_cls_preds_extended = masked_cls_preds.unsqueeze(dim=1)
        masked_cls_preds_extended = masked_cls_preds_extended.expand(-1, iou3d.size(1), -1)
        masked_cls_preds_extended = masked_cls_preds_extended.contiguous().view(-1, masked_cls_preds.size(1))
        
        masked_box_preds_extended = masked_box_preds.unsqueeze(dim=1)
        masked_box_preds_extended = masked_box_preds_extended.expand(-1, iou3d.size(1), -1)
        masked_box_preds_extended = masked_box_preds_extended.contiguous().view(-1, masked_box_preds.size(1))
        
        iou3d = iou3d.view(-1)
        if self.frozen_iou:
            iou3d = iou3d.detach()
            
        if self.frozen_logit:
            masked_cls_preds_extended = masked_cls_preds_extended.detach()
        
        iou_mask = (iou3d > self.iou_threshold)
        iou3d = iou3d[iou_mask]
        masked_cls_preds_extended = masked_cls_preds_extended[iou_mask]
        masked_box_preds_extended = masked_box_preds_extended[iou_mask]
        
        total_loss = -1.0 * torch.log(1.0 - masked_cls_preds_extended[:, target_class_logit]) * iou3d
        total_loss = total_loss.sum()
        
        return total_loss
