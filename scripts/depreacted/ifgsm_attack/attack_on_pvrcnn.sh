cd ../..
rm -rf output/train/pvrcnn_headbox_attack_ifgsm
rm -rf output/train/pvrcnn_full_attack_ifgsm
rm -rf output/train/pvrcnn_roihead_attack_ifgsm_iou_frozen
python attack.py --exp-name pvrcnn_headbox_attack_ifgsm --headbox-attack --optim ifgsm --cfg-file configs/attack_configs/kitti/relevant_bounding_box_pvrcnn.yaml
python attack.py --exp-name pvrcnn_full_attack_ifgsm --headbox-attack --roihead-attack --optim ifgsm --cfg-file configs/attack_configs/kitti/relevant_bounding_box_pvrcnn.yaml 
python attack.py --exp-name pvrcnn_roihead_attack_ifgsm --roihead-attack --optim ifgsm --cfg-file configs/attack_configs/kitti/relevant_bounding_box_pvrcnn.yaml 
