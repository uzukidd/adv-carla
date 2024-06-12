cd ..
rm -rf output/train/pvrcnn_full_attack
rm -rf output/train/pvrcnn_iou_frozen
rm -rf output/train/pvrcnn_logit_frozen
rm -rf output/train/pvrcnn_clean_data
rm -rf output/train/pvrcnn_init_patch
python attack.py --exp-name pvrcnn_full_attack --cfg-file configs/attack_configs/relevant_bounding_box_pvrcnn.yaml
python attack.py --exp-name pvrcnn_iou_frozen --iou-frozen --cfg-file configs/attack_configs/relevant_bounding_box_pvrcnn.yaml
python attack.py --exp-name pvrcnn_logit_frozen --logit-frozen --cfg-file configs/attack_configs/relevant_bounding_box_pvrcnn.yaml
python attack.py --exp-name pvrcnn_init_patch --eval-init-patch --cfg-file configs/attack_configs/relevant_bounding_box_pvrcnn.yaml
python attack.py --exp-name pvrcnn_clean_data --eval-clean-data --cfg-file configs/attack_configs/relevant_bounding_box_pvrcnn.yaml