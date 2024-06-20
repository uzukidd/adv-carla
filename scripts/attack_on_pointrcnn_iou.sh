cd ..
rm -rf output/train/pointrcnn_iou_headbox_attack
python attack.py --exp-name pointrcnn_iou_headbox_attack --headbox-attack --cfg-file configs/attack_configs/relevant_bounding_box_pointrcnn_iou.yaml
rm -rf pointrcnn_iou_full_attack
rm -rf pointrcnn_iou_headbox_attack
python attack.py --exp-name pointrcnn_iou_full_attack --headbox-attack --roihead-attack --cfg-file configs/attack_configs/relevant_bounding_box_pointrcnn_iou.yaml 
python attack.py --exp-name pointrcnn_iou_headbox_attack --roihead-attack --cfg-file configs/attack_configs/relevant_bounding_box_pointrcnn_iou.yaml 
