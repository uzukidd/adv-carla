cd ..
rm -rf output/train/second_full_attack
rm -rf output/train/second_iou_frozen
rm -rf output/train/second_logit_frozen
rm -rf output/train/second_clean_data
rm -rf output/train/second_init_patch
python attack.py --exp-name second_full_attack --cfg-file configs/attack_configs/relevant_bounding_box_second.yaml
python attack.py --exp-name second_iou_frozen --iou-frozen --cfg-file configs/attack_configs/relevant_bounding_box_second.yaml
python attack.py --exp-name second_logit_frozen --logit-frozen --cfg-file configs/attack_configs/relevant_bounding_box_second.yaml
python attack.py --exp-name second_init_patch --eval-init-patch --cfg-file configs/attack_configs/relevant_bounding_box_second.yaml
python attack.py --exp-name second_clean_data --eval-clean-data --cfg-file configs/attack_configs/relevant_bounding_box_second.yaml