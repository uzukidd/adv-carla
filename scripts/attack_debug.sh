cd ..
rm -rf output/train/pointrcnnmk1_roihead_attack_iou_frozen
rm -rf output/train/pointrcnnmk1_roihead_attack_logit_frozen
rm -rf output/train/pointrcnnmk1_full_attack_iou_frozen
rm -rf output/train/ointrcnnmk1_full_attack_logit_frozen
# python attack.py --verbose-epoch 1 --exp-name debug --cfg-file configs/attack_configs/relevant_bounding_box_pvrcnn.yaml
# python attack.py --exp-name pointrcnnmk1_clean_data --eval-clean-data --cfg-file configs/attack_configs/relevant_bounding_box_pointrcnn.yaml
# python attack.py --exp-name pointrcnnmk1_init_patch --eval-init-patch  --cfg-file configs/attack_configs/relevant_bounding_box_pointrcnn.yaml
# python attack.py --exp-name pointrcnnmk1_full_attack --headbox-attack --roihead-attack --cfg-file configs/attack_configs/relevant_bounding_box_pointrcnn.yaml 
# python attack.py --exp-name pointrcnnmk1_headbox_attack --headbox-attack --cfg-file configs/attack_configs/relevant_bounding_box_pointrcnn.yaml 
python attack.py --exp-name pointrcnnmk1_roihead_attack_iou_frozen --roihead-attack --iou-frozen --cfg-file configs/attack_configs/relevant_bounding_box_pointrcnn.yaml 
python attack.py --exp-name pointrcnnmk1_roihead_attack_logit_frozen --roihead-attack --logit-frozen --cfg-file configs/attack_configs/relevant_bounding_box_pointrcnn.yaml
python attack.py --exp-name pointrcnnmk1_full_attack_iou_frozen --roihead-attack --headbox-attack --iou-frozen --cfg-file configs/attack_configs/relevant_bounding_box_pointrcnn.yaml 
python attack.py --exp-name pointrcnnmk1_full_attack_logit_frozen --roihead-attack --headbox-attack --logit-frozen --cfg-file configs/attack_configs/relevant_bounding_box_pointrcnn.yaml 
