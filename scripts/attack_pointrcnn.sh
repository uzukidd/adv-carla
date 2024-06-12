cd ..
rm -rf output/train/pointrcnn_full_attack
rm -rf output/train/pointrcnn_iou_frozen
rm -rf output/train/pointrcnn_logit_frozen
rm -rf output/train/pointrcnn_clean_data
rm -rf output/train/pointrcnn_init_patch
python attack.py --headbox-attack --exp-name pointrcnn_full_attack --cfg-file configs/attack_configs/relevant_bounding_box_pointrcnn.yaml
python attack.py --headbox-attack --exp-name pointrcnn_iou_frozen --iou-frozen --cfg-file configs/attack_configs/relevant_bounding_box_pointrcnn.yaml
python attack.py --headbox-attack --exp-name pointrcnn_logit_frozen --logit-frozen --cfg-file configs/attack_configs/relevant_bounding_box_pointrcnn.yaml
python attack.py --headbox-attack --exp-name pointrcnn_clean_data --eval-clean-data --cfg-file configs/attack_configs/relevant_bounding_box_pointrcnn.yaml
python attack.py --headbox-attack --exp-name pointrcnn_init_patch --eval-init-patch --cfg-file configs/attack_configs/relevant_bounding_box_pointrcnn.yaml
