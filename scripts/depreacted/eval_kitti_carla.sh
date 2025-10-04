cd ..
rm -rf output/eval/kitti-carla-eval-pointpillar
python eval.py --exp-name kitti-carla-eval-pointpillar --cfg-file configs/attack_configs/kitticarla/eval_pointpillar.yaml
