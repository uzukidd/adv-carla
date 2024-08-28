cd ../..
rm -rf output/train/comparison_table/$1/misrecognize_10
rm -rf output/train/comparison_table/$1/misrecognize_9
python main.py --exp-name comparison_table/$1/misrecognize_10 --roihead-attack --iou-frozen --stage-2-loss-reduce-func logit_multiply_iou3d --optim ifgsm --learning-rate $3 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $2
python main.py --exp-name comparison_table/$1/misrecognize_9 --roihead-attack --iou-frozen --stage-2-loss-reduce-func score_multiply_iou3d --optim ifgsm --learning-rate $3 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $2