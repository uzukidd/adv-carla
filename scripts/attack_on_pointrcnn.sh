cd ..
rm -rf output/train/pointrcnn_full_attack
rm -rf output/train/pointrcnn_headbox_attack
rm -rf output/train/pointrcnn_roihead_attack
python attack.py --exp-name pointrcnn_headbox_attack --headbox-attack --cfg-file configs/attack_configs/relevant_bounding_box_pointrcnn.yaml
python attack.py --exp-name pointrcnn_full_attack --headbox-attack --roihead-attack --cfg-file configs/attack_configs/relevant_bounding_box_pointrcnn.yaml 
python attack.py --exp-name pointrcnn_roihead_attack --roihead-attack --cfg-file configs/attack_configs/relevant_bounding_box_pointrcnn.yaml 
