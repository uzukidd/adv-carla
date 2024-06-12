cd ..
rm -rf output/train/pointpillar_clean_data
rm -rf output/train/pointpillar_full_attack
rm -rf output/train/pointpillar_iou_frozen
rm -rf output/train/pointpillar_logit_frozen
rm -rf output/train/pointpillar_init_patch
python attack.py --headbox-attack --exp-name pointpillar_clean_data --eval-clean-data --cfg-file configs/attack_configs/relevant_bounding_box_pointpillar.yaml
python attack.py --headbox-attack --exp-name pointpillar_full_attack --cfg-file configs/attack_configs/relevant_bounding_box_pointpillar.yaml
python attack.py --headbox-attack --exp-name pointpillar_iou_frozen --iou-frozen --cfg-file configs/attack_configs/relevant_bounding_box_pointpillar.yaml
python attack.py --headbox-attack --exp-name pointpillar_logit_frozen --logit-frozen --cfg-file configs/attack_configs/relevant_bounding_box_pointpillar.yaml
python attack.py --headbox-attack --exp-name pointpillar_init_patch --eval-init-patch --logit-frozen --cfg-file configs/attack_configs/relevant_bounding_box_pointpillar.yaml