cd ../..
rm -rf output/train/step_ablation/$1_ifgsm_$2
python main.py --exp-name step_ablation/$1_ifgsm_$2 --stage-2-loss-reduce-func logit_multiply_iou3d --learning-rate $2 --optim ifgsm --iou-frozen  --roihead-attack --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $3