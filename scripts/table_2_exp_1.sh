cd ..
rm -rf output/train/second_misrecognize_10
# rm -rf output/train/second_mislocalize_4
python main.py --exp-name second_misrecognize_10 --roihead-attack --iou-frozen --stage-2-loss-reduce-func logit_multiply_iou3d --optim ifgsm --learning-rate 0.0005 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_second.yaml --device $1
# python main.py --exp-name second_mislocalize_4 --roihead-attack --logit-frozen --stage-2-loss-reduce-func physical_loss --optim ifgsm --learning-rate 0.0005 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_second.yaml --device $1