cd ../..
rm -rf output/train/loss_ablation/pointpillar/mislocalize_4
rm -rf output/train/loss_ablation/pointpillar/misrecognize_4
rm -rf output/train/loss_ablation/pointpillar/misrecognize_8
rm -rf output/train/loss_ablation/pointpillar/comprehensive_4

python main.py --exp-name loss_ablation/pointpillar/mislocalize_4 --roihead-attack --logit-frozen --stage-2-loss-reduce-func physical_loss --optim ifgsm --learning-rate 0.5 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_pointpillar.yaml --device $1
# python main.py --exp-name loss_ablation/pointpillar/mislocalize_7 --roihead-attack --logit-frozen --stage-2-loss-reduce-func monocular_iou --optim ifgsm --learning-rate 0.5 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_pointpillar.yaml --device $1
# python main.py --exp-name loss_ablation/pointpillar/mislocalize_9 --roihead-attack --logit-frozen --stage-2-loss-reduce-func score_multiply_iou3d --optim ifgsm --learning-rate 0.5 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_pointpillar.yaml --device $1
python main.py --exp-name loss_ablation/pointpillar/misrecognize_4 --roihead-attack --iou-frozen --stage-2-loss-reduce-func physical_loss --optim ifgsm --learning-rate 0.5 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_pointpillar.yaml --device $1
python main.py --exp-name loss_ablation/pointpillar/misrecognize_8 --roihead-attack --iou-frozen --stage-2-loss-reduce-func entropy_score_loss --optim ifgsm --learning-rate 0.5 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_pointpillar.yaml --device $1
# python main.py --exp-name loss_ablation/pointpillar/misrecognize_9 --roihead-attack --iou-frozen --stage-2-loss-reduce-func score_multiply_iou3d --optim ifgsm --learning-rate 0.5 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_pointpillar.yaml --device $1
# python main.py --exp-name loss_ablation/pointpillar/misrecognize_10 --roihead-attack --iou-frozen --stage-2-loss-reduce-func logit_multiply_iou3d --optim ifgsm --learning-rate 0.5 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_pointpillar.yaml --device $1
python main.py --exp-name loss_ablation/pointpillar/comprehensive_4  --roihead-attack --stage-2-loss-reduce-func physical_loss --optim ifgsm --learning-rate 0.5 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_pointpillar.yaml --device $1
# python main.py --exp-name loss_ablation/pointpillar/comprehensive_9  --roihead-attack --stage-2-loss-reduce-func score_multiply_iou3d --optim ifgsm --learning-rate 0.5 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_pointpillar.yaml --device $1
