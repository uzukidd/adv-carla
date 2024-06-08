cd ..
rm -rf output/train/full_attack_voxel_rcnn_car
python attack.py --headbox-attack --cfg-file configs/attack_configs/relevant_bounding_box_voxel_rcnn_car.yaml --exp-name full_attack_voxel_rcnn_car --eval-clean-data