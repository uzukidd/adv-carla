cd ..
rm -rf output/train/pointrcnn_iou_full_attack
rm -rf output/train/pointrcnn_iou_iou_frozen
rm -rf output/train/pointrcnn_iou_logit_frozen
rm -rf output/train/pointrcnn_iou_clean_data
rm -rf output/train/pointrcnn_iou_init_patch
python attack.py --exp-name pointrcnn_iou_full_attack --cfg-file configs/attack_configs/relevant_bounding_box_pointrcnn_iou.yaml
python attack.py --exp-name pointrcnn_iou_iou_frozen --iou-frozen --cfg-file configs/attack_configs/relevant_bounding_box_pointrcnn_iou.yaml
python attack.py --exp-name pointrcnn_iou_logit_frozen --logit-frozen --cfg-file configs/attack_configs/relevant_bounding_box_pointrcnn_iou.yaml
python attack.py --exp-name pointrcnn_iou_init_patch --eval-init-patch --cfg-file configs/attack_configs/relevant_bounding_box_pointrcnn_iou.yaml
python attack.py --exp-name pointrcnn_iou_clean_data --eval-clean-data --cfg-file configs/attack_configs/relevant_bounding_box_pointrcnn_iou.yaml