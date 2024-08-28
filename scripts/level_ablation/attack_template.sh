cd ../..
rm -rf output/train/level_ablation/$1_misrecognize_10_$3
python main.py --exp-name level_ablation/$1_misrecognize_10_$3 --learning-rate $4 --roihead-attack --level $3  --iou-frozen --stage-2-loss-reduce-func logit_multiply_iou3d --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $2
# python attack.py --headbox-attack --exp-name scale_ablation/$1_logit_frozen_$3_$4_$5 --logit-frozen --cfg-file configs/attack_configs/relevant_bounding_box_$1.yaml --device $2
# python attack.py --exp-name scale_ablation/$1_eval_init_patch_$3_$4_$5 --eval-init-patch --scale $3 $4 $5 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $2