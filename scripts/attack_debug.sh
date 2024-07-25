cd ..
rm -rf output/train/debug
python main.py --exp-name debug  --cfg-file configs/attack_configs/kitticarla/inference_pointpillar.yaml
