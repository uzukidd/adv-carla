cd ../..
rm -rf output/train/kitti_eval/$1_$2
python main.py --exp-name kitti_eval/$1_$2 --device $4 --cfg-file configs/attack_configs/kitti/eval_$2.yaml $3