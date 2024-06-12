cd ..
rm -rf output/train/voxel_rcnn_car_full_attack
rm -rf output/train/voxel_rcnn_car_iou_frozen
rm -rf output/train/voxel_rcnn_car_logit_frozen
rm -rf output/train/voxel_rcnn_car_clean_data
rm -rf output/train/voxel_rcnn_car_init_patch
python attack.py --exp-name voxel_rcnn_car_full_attack --cfg-file configs/attack_configs/relevant_bounding_box_voxel_rcnn_car.yaml
python attack.py --exp-name voxel_rcnn_car_iou_frozen --iou-frozen --cfg-file configs/attack_configs/relevant_bounding_box_voxel_rcnn_car.yaml
python attack.py --exp-name voxel_rcnn_car_logit_frozen --logit-frozen --cfg-file configs/attack_configs/relevant_bounding_box_voxel_rcnn_car.yaml
python attack.py --exp-name voxel_rcnn_car_init_patch --eval-init-patch --cfg-file configs/attack_configs/relevant_bounding_box_voxel_rcnn_car.yaml
python attack.py --exp-name voxel_rcnn_car_clean_data --eval-clean-data --cfg-file configs/attack_configs/relevant_bounding_box_voxel_rcnn_car.yaml