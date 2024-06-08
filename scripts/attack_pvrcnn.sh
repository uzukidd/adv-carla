cd ..
rm -rf output/train/full_attack_pvrcnn
python attack.py --headbox-attack --cfg-file configs/attack_configs/relevant_bounding_box_pvrcnn.yaml --exp-name full_attack_pvrcnn --eval-clean-data