cd ..
rm -rf output/train/pvrcnn_full_attack
rm -rf output/train/pvrcnn_roibox_attack
python attack.py --exp-name pvrcnn_full_attack --headbox-attack --roihead-attack --cfg-file configs/attack_configs/relevant_bounding_box_pvrcnn.yaml 
python attack.py --exp-name pvrcnn_roibox_attack --roihead-attack --cfg-file configs/attack_configs/relevant_bounding_box_pvrcnn.yaml 
