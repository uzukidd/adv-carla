cd ../..
rm -rf output/train/loss_ablation/$1/mislocalize_4
rm -rf output/train/loss_ablation/$1/misrecognize_4
rm -rf output/train/loss_ablation/$1/misrecognize_8
rm -rf output/train/loss_ablation/$1/comprehensive_4
rm -rf output/train/loss_ablation/$1/mislocalize_7
rm -rf output/train/loss_ablation/$1/mislocalize_9
rm -rf output/train/loss_ablation/$1/misrecognize_9
rm -rf output/train/loss_ablation/$1/misrecognize_10
rm -rf output/train/loss_ablation/$1/comprehensive_9


python main.py --exp-name loss_ablation/$1/mislocalize_4 --roihead-attack --logit-frozen --stage-2-loss-reduce-func physical_loss --optim ifgsm --learning-rate $3 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $2
python main.py --exp-name loss_ablation/$1/mislocalize_7 --roihead-attack --logit-frozen --stage-2-loss-reduce-func monocular_iou --optim ifgsm --learning-rate $3 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $2
python main.py --exp-name loss_ablation/$1/mislocalize_9 --roihead-attack --logit-frozen --stage-2-loss-reduce-func score_multiply_iou3d --optim ifgsm --learning-rate $3 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $2
python main.py --exp-name loss_ablation/$1/misrecognize_4 --roihead-attack --iou-frozen --stage-2-loss-reduce-func physical_loss --optim ifgsm --learning-rate $3 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $2
python main.py --exp-name loss_ablation/$1/misrecognize_8 --roihead-attack --iou-frozen --stage-2-loss-reduce-func entropy_score_loss --optim ifgsm --learning-rate $3 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $2
python main.py --exp-name loss_ablation/$1/misrecognize_9 --roihead-attack --iou-frozen --stage-2-loss-reduce-func score_multiply_iou3d --optim ifgsm --learning-rate $3 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $2
python main.py --exp-name loss_ablation/$1/misrecognize_10 --roihead-attack --iou-frozen --stage-2-loss-reduce-func logit_multiply_iou3d --optim ifgsm --learning-rate $3 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $2
python main.py --exp-name loss_ablation/$1/comprehensive_4  --roihead-attack --stage-2-loss-reduce-func physical_loss --optim ifgsm --learning-rate $3 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $2
python main.py --exp-name loss_ablation/$1/comprehensive_9  --roihead-attack --stage-2-loss-reduce-func score_multiply_iou3d --optim ifgsm --learning-rate $3 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $2
