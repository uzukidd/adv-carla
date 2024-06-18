cd ..
# rm -rf output/train/scale_ablation/$1_full_attack_$3_$4_$5
# rm -rf output/train/scale_ablation/$1_iou_frozen_$3_$4_$5
# rm -rf output/train/scale_ablation/$1_logit_frozen_$3_$4_$5
rm -rf output/train/level_ablation/$1_full_attack_$3
# python attack.py --headbox-attack --exp-name scale_ablation/$1_full_attack_$3_$4_$5 --cfg-file configs/attack_configs/relevant_bounding_box_$1.yaml --device $2
# python attack.py --headbox-attack --exp-name scale_ablation/$1_iou_frozen_$3_$4_$5 --iou-frozen --cfg-file configs/attack_configs/relevant_bounding_box_$1.yaml --device $2
# python attack.py --headbox-attack --exp-name scale_ablation/$1_logit_frozen_$3_$4_$5 --logit-frozen --cfg-file configs/attack_configs/relevant_bounding_box_$1.yaml --device $2
python attack.py --exp-name level_ablation/$1_full_attack_$3 --level $3 --cfg-file configs/attack_configs/relevant_bounding_box_$1.yaml --device $2