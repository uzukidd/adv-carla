cd ..
# rm -rf output/train/voxel_rcnn_full_attack
# rm -rf output/train/voxel_rcnn_roibox_attack
python attack.py --exp-name voxel_rcnn_headbox_attack --headbox-attack --cfg-file configs/attack_configs/relevant_bounding_box_voxel_rcnn_car.yaml 
# python attack.py --exp-name voxel_rcnn_full_attack --headbox-attack --roihead-attack --cfg-file configs/attack_configs/relevant_bounding_box_voxel_rcnn_car.yaml 
# python attack.py --exp-name voxel_rcnn_roibox_attack --roihead-attack --cfg-file configs/attack_configs/relevant_bounding_box_voxel_rcnn_car.yaml 
