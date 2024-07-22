cd ..
# rm -rf output/train/scale_ablation/$1_roihead_attack_$3_$4_$5
# rm -rf output/train/scale_ablation/$1_iou_frozen_$3_$4_$5
# rm -rf output/train/scale_ablation/$1_logit_frozen_$3_$4_$5
rm -rf output/train/scale_ablation/$1_eval_init_patch_$3_$4_$5
# python attack.py --roihead-attack --exp-name scale_ablation/$1_roihead_attack_$3_$4_$5 --scale $3 $4 $5 --iou-frozen --optim ifgsm --learning-rate $6 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $2
# python attack.py --headbox-attack --exp-name scale_ablation/$1_iou_frozen_$3_$4_$5 --iou-frozen --cfg-file configs/attack_configs/relevant_bounding_box_$1.yaml --device $2
# python attack.py --headbox-attack --exp-name scale_ablation/$1_logit_frozen_$3_$4_$5 --logit-frozen --cfg-file configs/attack_configs/relevant_bounding_box_$1.yaml --device $2
python attack.py --exp-name scale_ablation/$1_eval_init_patch_$3_$4_$5 --eval-init-patch --scale $3 $4 $5 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $2