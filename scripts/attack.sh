cd ..
# rm -rf output/train/$1_clean_data
rm -rf output/train/$1_full_attack
# rm -rf output/train/$1_headbox_attack
# rm -rf output/train/$1_headbox_iou_frozen
# rm -rf output/train/$1_headbox_logit_frozen
# rm -rf output/train/$1_roihead_attack
# rm -rf output/train/$1_roihead_iou_frozen
# rm -rf output/train/$1_roihead_logit_frozen
# rm -rf output/train/$1_init_patch
python attack.py --exp-name $1_full_attack --headbox-attack --roihead-attack --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $2
# python attack.py --exp-name $1_headbox_attack --headbox-attack --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $2
# python attack.py --exp-name $1_headbox_iou_frozen --headbox-attack --iou-frozen --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $2
# python attack.py --exp-name $1_headbox_logit_frozen --headbox-attack --logit-frozen --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $2
# python attack.py --exp-name $1_roihead_attack --roihead-attack --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $2
# python attack.py --exp-name $1_roihead_iou_frozen --roihead-attack --iou-frozen --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $2
# python attack.py --exp-name $1_roihead_logit_frozen --roihead-attack --logit-frozen --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $2
# python attack.py --headbox-attack --exp-name $1_init_patch --eval-init-patch --cfg-file configs/attack_configs/relevant_bounding_box_$1.yaml --device $2screen