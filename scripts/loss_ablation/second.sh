cd ../..
# rm -rf output/train/loss_ablation/pvrcnn/mislocalize_4
rm -rf output/train/loss_ablation/second/misrecognize_10
rm -rf output/train/loss_ablation/second/misrecognize_9

# python main.py --exp-name loss_ablation/pvrcnn/mislocalize_4 --roihead-attack --logit-frozen --stage-2-loss-reduce-func physical_loss --optim ifgsm --learning-rate 0.005 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_pvrcnn.yaml --device $1
python main.py --exp-name loss_ablation/second/misrecognize_9 --roihead-attack --iou-frozen --stage-2-loss-reduce-func score_multiply_iou3d --optim ifgsm --learning-rate 0.0005 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_second.yaml --device $1
python main.py --exp-name loss_ablation/second/misrecognize_10 --roihead-attack --iou-frozen --stage-2-loss-reduce-func logit_multiply_iou3d --optim ifgsm --learning-rate 0.0005 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_second.yaml --device $1
