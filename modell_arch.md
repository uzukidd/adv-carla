# Point-RCNN

## forwarding order

1. preparing (dataset)
2. module_list
   - get_training_loss (if training)
   - post_processing (if not training)

## get training loss

- point_head.get_loss
  - get_cls_layer_loss
  - get_box_layer_loss
- roi_head.get_loss
  - get_box_cls_layer_loss
  - get_box_reg_layer_loss

## module list

1. (BACKBONE_3D) PointNet2MSG

   extends: nn.Module

2. (POINT_HEAD) PointHeadBox

   extends: PointHeadTemplate

   1. prediction

      - cls_layers (fc_layers)

      - box_layers (fc_layers)

   2. assign_targets(if training)
      1. enlarge_box3d

      2. assign_stack_targets

   3. generate_predicted_boxes (not self.training or self.predict_boxes_when_training)

3. (ROI_HEAD) PointRCNNHead

   extends: RoIHeadTemplate

   1. proposal_layer

   2. assign_targets (if training)
      1. proposal_target_layer (ProposalTargetLayer)
         1. sample_rois_for_rcnn
            1. boxes_iou3d_gpu
   
            2. subsample_rois
               1. fg_num_rois > 0 and bg_num_rois > 0
                  1. sample_bg_inds
   
               2. fg_num_rois > 0 and bg_num_rois == 0
   
               3. fg_num_rois == 0 and bg_num_rois > 0
                  1. sample_bg_inds

         2. REG_FG_THRESH

         3. CLS_SCORE
   
      2. rotate_points_along_z
   
   3. roipool3d_gpu
   
   4. xyz_up_layer
   
   5. merge_down_layer
   
   6. SA_modules
   
   7. prediction
   
      - cls_layers (fc_layers)
   
      - box_layers (fc_layers)
   
   8. generate_predicted_boxes (not self.training)

