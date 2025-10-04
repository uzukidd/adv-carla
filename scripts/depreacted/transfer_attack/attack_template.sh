cd ../..
rm -rf output/train/transfer_eval/$1/$2_$3
python main.py --exp-name transfer_eval/$1/$2_$3 --device $4 --cfg-file configs/attack_configs/evaluation/eval_$1.yaml --patch-ckpt output/train/loss_ablation/$2/$3/final_adversarial_patch_checkpoint.pt