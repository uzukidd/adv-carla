cd ..
rm -rf output/train/full_attack_pointrcnn
python attack.py --headbox-attack --cfg-file configs/attack_configs/relevant_bounding_box_pointrcnn.yaml --exp-name full_attack_pointrcnn --eval-clean-data