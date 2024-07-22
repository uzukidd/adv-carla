cd ..
rm -rf output/train/$1_loss_ablation/loss_4_ifgsm
rm -rf output/train/$1_loss_ablation/loss_4_adam

rm -rf output/train/$1_loss_ablation/loss_5_ifgsm
rm -rf output/train/$1_loss_ablation/loss_5_adam

rm -rf output/train/$1_loss_ablation/loss_5_ifgsm_iou_frozen
rm -rf output/train/$1_loss_ablation/loss_5_adam_iou_frozen

rm -rf output/train/$1_loss_ablation/loss_5_ifgsm_logit_frozen
rm -rf output/train/$1_loss_ablation/loss_5_adam_logit_frozen



(
python attack.py --exp-name $1_loss_ablation/loss_4_ifgsm --mode 4 --optim ifgsm --learning-rate 0.5 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $2
) & 
(
python attack.py --exp-name $1_loss_ablation/loss_4_adam --mode 4 --optim adam --learning-rate 0.5 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $3
) &
wait

(
python attack.py --exp-name $1_loss_ablation/loss_5_ifgsm --mode 5 --optim ifgsm --learning-rate 0.5 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $2
) & 
(
python attack.py --exp-name $1_loss_ablation/loss_5_adam --mode 5 --optim adam --learning-rate 0.5 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $3
) &
wait

(
python attack.py --exp-name $1_loss_ablation/loss_5_ifgsm_iou_frozen --mode 5 --optim ifgsm --iou-frozen --learning-rate 0.5 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $2
) & 
(
python attack.py --exp-name $1_loss_ablation/loss_5_adam_iou_frozen --mode 5 --optim adam --iou-frozen --learning-rate 0.5 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $3
) &
wait

(
python attack.py --exp-name $1_loss_ablation/loss_5_ifgsm_logit_frozen --mode 5 --optim ifgsm --logit-frozen --learning-rate 0.5 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $2
) & 
(
python attack.py --exp-name $1_loss_ablation/loss_5_adam_logit_frozen --mode 5 --optim adam --logit-frozen --learning-rate 0.5 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $3
) &
wait
