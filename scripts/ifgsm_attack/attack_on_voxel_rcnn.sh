cd ../..
rm -rf output/train/voxel_rcnn_headbox_attack_ifgsm
rm -rf output/train/voxel_rcnn_full_attack_ifgsm
rm -rf output/train/voxel_rcnn_roihead_attack_ifgsm_iou_frozen
python attack.py --exp-name voxel_rcnn_headbox_attack_ifgsm --headbox-attack --optim ifgsm --cfg-file configs/attack_configs/kitti/relevant_bounding_box_voxel_rcnn_car.yaml
python attack.py --exp-name voxel_rcnn_full_attack_ifgsm --headbox-attack --roihead-attack --optim ifgsm --cfg-file configs/attack_configs/kitti/relevant_bounding_box_voxel_rcnn_car.yaml 
python attack.py --exp-name voxel_rcnn_roihead_attack_ifgsm_iou_frozen --roihead-attack --optim ifgsm --iou-frozen --cfg-file configs/attack_configs/kitti/relevant_bounding_box_voxel_rcnn_car.yaml 
