cd ..
# rm -rf output/train/$1_loss_ablation_1
# rm -rf output/train/$1_loss_ablation_-1
# rm -rf output/train/$1_loss_ablation_0
# rm -rf output/train/$1_loss_ablation_2
# rm -rf output/train/$1_loss_ablation_3
rm -rf output/train/$1_loss_ablation_-2
rm -rf output/train/$1_loss_ablation_-3

# (
# python attack.py --exp-name $1_loss_ablation_1 --mode 1 --optim ifgsm --learning-rate 0.05 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $2
# python attack.py --exp-name $1_loss_ablation_-1 --mode -1 --optim ifgsm --learning-rate 0.05 --iou-frozen --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $2 
# python attack.py --exp-name $1_loss_ablation_0 --mode 0 --optim ifgsm --learning-rate 0.05 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $2 
# )
# &

# python attack.py --exp-name $1_loss_ablation_2 --mode 2 --optim ifgsm --learning-rate 0.05 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $2
# &
# python attack.py --exp-name $1_loss_ablation_3 --mode 3 --optim ifgsm --learning-rate 0.05 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $3 
python attack.py --exp-name $1_loss_ablation_-2 --mode -2 --optim ifgsm --learning-rate 0.05 --iou-frozen --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $3 
python attack.py --exp-name $1_loss_ablation_-3 --mode -3 --optim ifgsm --learning-rate 0.05 --iou-frozen --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $3 

# wait