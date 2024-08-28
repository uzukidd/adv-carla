cd ../..
rm -rf output/train/CarLA_eval/$1_$2
python main.py --exp-name CarLA_eval/$1_$2 --device $4 --cfg-file configs/attack_configs/kitticarla/eval_$2.yaml $3