cd ..
rm -rf output/train/weights_ablation/$1_roihead_attack_$3
python attack.py --headbox-attack --exp-name weights_ablation/$1_roihead_attack_$3 --roihead-attack --roi-head-weights $3 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $2
