cd ..
# rm -rf output/train/$1_clean_data
rm -rf output/train/$1_full_attack
rm -rf output/train/$1_iou_frozen
rm -rf output/train/$1_logit_frozen
# rm -rf output/train/$1_init_patch
# python attack.py --headbox-attack --exp-name $1_clean_data --eval-clean-data --cfg-file configs/attack_configs/relevant_bounding_box_$1.yaml --device $2
python attack.py --headbox-attack --exp-name $1_full_attack --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $2
python attack.py --headbox-attack --exp-name $1_iou_frozen --iou-frozen --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $2
python attack.py --headbox-attack --exp-name $1_logit_frozen --logit-frozen --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $2
# python attack.py --headbox-attack --exp-name $1_init_patch --eval-init-patch --cfg-file configs/attack_configs/relevant_bounding_box_$1.yaml --device $2