UNI_RANDOM_SEED = 2024

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional

import pdb

from cudaext.ops.Rotated_IoU.oriented_iou_loss import cal_iou_3d
from cudaext.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu



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
                 iou_reduce: str = None,
                 verbose: bool = False):
        super().__init__()
        self.frozen_iou = frozen_iou
        self.frozen_logit = frozen_logit
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.verbose = verbose
        self.iou_reduce = iou_reduce
        
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
    
    def forward_single(self, batch_dict, 
                gt_boxes: torch.Tensor, 
                target_class: int, 
                logit_normal:str = 'sigmoid',
                loss_reduce_func: Optional[Callable[[torch.Tensor, 
                                            torch.Tensor, torch.Tensor], torch.Tensor]] = None,):
        
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
        preds_scores:torch.Tensor = batch_dict["pred_scores"] # [N, ]
        preds_boxes:torch.Tensor = batch_dict["pred_boxes"] # [N, 7]
        pred_labels:torch.Tensor = batch_dict["pred_labels"].view(-1) # [N, ]

        gt_boxes, gt_labels = torch.split(gt_boxes.squeeze(dim=0), [7, 1], dim=1)  # [N, 7], [N, 1]
        gt_boxes = gt_boxes[gt_labels[:, 0] == target_class]
        gt_labels = gt_labels[gt_labels[:, 0] == target_class]
        
        if gt_boxes.numel() == 0:
            return None, True
        # assert gt_boxes.size(0), "at leaset one gt box exists."
        
        # assign 
        # batch_cls_preds_normalized = F.softmax(batch_cls_preds, dim=1) # [N, 3]

        classes_mask = (pred_labels == target_class)
        masked_cls_preds = preds_scores[classes_mask]
        masked_box_preds = preds_boxes[classes_mask]
        
        if logit_normal == "sigmoid":
            cls_preds_normalized = torch.sigmoid(masked_cls_preds) # [N, ]
        else:
            raise NotImplementedError
        
        confidence_mask = (cls_preds_normalized > self.confidence_threshold)
        masked_cls_preds = masked_cls_preds[confidence_mask]
        masked_box_preds = masked_box_preds[confidence_mask]
        
        if masked_cls_preds.numel() == 0:
            return None, True
        
        """
        Calculate iou.
        """
        iou3d = self.iou3d(masked_box_preds, gt_boxes)
        
        if self.frozen_iou:
            iou3d = iou3d.detach()
            
        if self.frozen_logit:
            masked_cls_preds = masked_cls_preds.detach()
        
        masked_cls_preds_extended = masked_cls_preds.unsqueeze(dim=1)
        masked_cls_preds_extended = masked_cls_preds_extended.expand(-1, iou3d.size(1))
        masked_cls_preds_extended = masked_cls_preds_extended.contiguous().view(-1)
        
        iou3d = iou3d.view(-1)
        
        """
        IoU threshold filter.
        """
        iou_mask = (iou3d > self.iou_threshold)
        iou3d = iou3d[iou_mask]
        masked_cls_preds_extended = masked_cls_preds_extended[iou_mask]
        
        if masked_cls_preds_extended.numel() == 0:
            return None, True

        if loss_reduce_func is None:
            # total_loss = -1.0 * torch.log(1.0 - masked_cls_preds_extended) * iou3d
            total_loss = masked_cls_preds_extended * iou3d
            total_loss = total_loss.sum()
        else:
            total_loss = loss_reduce_func(iou3d, 
                                          masked_cls_preds, 
                                          masked_cls_preds_extended)
            
        if total_loss.numel() == 0:
            return None, True
        
        return total_loss, False
    
    def forward_headbox(self, batch_dict, 
                gt_boxes: torch.Tensor, 
                target_class: int, 
                logit_normal:str = 'sigmoid',
                loss_reduce_func: Optional[Callable[[torch.Tensor, 
                                            torch.Tensor, torch.Tensor], torch.Tensor]] = None,):
        
        """

        Args:
        - batch_cls_preds: [N, 3]
        - batch_box_preds: [N, 7]
        - gt_boxes: [1, N, 8]
        - point_coords: [N, 3]
        - ret_part_loss: bool

        Returns:
        - loss:
        - empty_flag: 
        """
        batch_cls_preds:torch.Tensor = batch_dict["batch_cls_preds"] # [N, 3]
        batch_box_preds:torch.Tensor = batch_dict["batch_box_preds"] # [N, 7]
        
        if batch_cls_preds.size().__len__() == 3:
            batch_cls_preds = batch_cls_preds.view(-1, batch_cls_preds.size(-1))
            batch_box_preds = batch_box_preds.view(-1, batch_box_preds.size(-1))
        
        gt_boxes, gt_labels = torch.split(gt_boxes.squeeze(dim=0), [7, 1], dim=1)  # [N, 7], [N, 1]
        gt_boxes = gt_boxes[gt_labels[:, 0] == target_class]
        gt_labels = gt_labels[gt_labels[:, 0] == target_class]
        
        target_class_logit = target_class - 1

        # assert gt_boxes.size(0), "at leaset one gt box exists."
        if gt_boxes.numel() == 0:
            return None, True
        
        # normalization 
        if logit_normal == "sigmoid":
            """
            Normalized scores are only used to filter out results below the threshold.
            """
            batch_cls_preds_normalized = torch.sigmoid(batch_cls_preds) # [N, 3]
            confidence_mask = (batch_cls_preds_normalized[:, target_class_logit] > self.confidence_threshold)
        elif logit_normal == "softmax":
            """
            Normalized scores are only used to filter out results below the threshold.
            """
            batch_cls_preds_normalized = F.softmax(batch_cls_preds, dim=1) # [N, 3]
            confidence_mask = (batch_cls_preds_normalized[:, target_class_logit] > self.confidence_threshold)
        else:
            raise NotImplementedError
                
        masked_cls_preds = batch_cls_preds[confidence_mask]
        masked_box_preds = batch_box_preds[confidence_mask]
        
        if masked_cls_preds.numel() == 0:
            return None, True
        
        iou3d = self.iou3d(masked_box_preds, gt_boxes)

        masked_cls_preds = masked_cls_preds[:, target_class_logit]
        masked_cls_preds_extended = masked_cls_preds.unsqueeze(dim=1)
        masked_cls_preds_extended = masked_cls_preds_extended.expand(-1, iou3d.size(1))
        masked_cls_preds_extended = masked_cls_preds_extended.contiguous().view(-1)
                
        iou3d = iou3d.view(-1)
        if self.frozen_iou:
            iou3d = iou3d.detach()
            
        if self.frozen_logit:
            masked_cls_preds_extended = masked_cls_preds_extended.detach()
        
        iou_mask = (iou3d > self.iou_threshold)
        iou3d = iou3d[iou_mask]

        masked_cls_preds_extended = masked_cls_preds_extended[iou_mask]
        
        if masked_cls_preds_extended.numel() == 0:
            return None, True
        
        if loss_reduce_func is None:
            # total_loss = -1.0 * torch.log(1.0 - masked_cls_preds_extended) * iou3d
            total_loss = masked_cls_preds_extended * iou3d
            total_loss = total_loss.sum()
        else:
            total_loss = loss_reduce_func(iou3d, 
                                          masked_cls_preds, 
                                          masked_cls_preds_extended)
            
        if total_loss.numel() == 0:
            return None, True
        
        return total_loss, False
        
    def forward(self, batch_dict, 
                gt_boxes: torch.Tensor, 
                target_class: int, 
                logit_normal:str = 'sigmoid',
                detector_type:str = 'single',
                loss_reduce_func: Optional[Callable[[torch.Tensor, 
                                            torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                ret_part_loss: bool = False):
        
        if detector_type == 'single':
            return self.forward_single(batch_dict, 
                gt_boxes, 
                target_class, 
                logit_normal,
                loss_reduce_func)
        elif detector_type == 'rcnn':
            return self.forward_headbox(batch_dict, 
                gt_boxes, 
                target_class, 
                logit_normal,
                loss_reduce_func)
            
        raise NotImplementedError
    