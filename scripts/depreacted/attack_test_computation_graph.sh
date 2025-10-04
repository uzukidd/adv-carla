cd ..
rm -rf output/train/debug_computation
python attack.py --exp-name debug_computation --verbose-epoch 1 --roihead-attack --cfg-file configs/attack_configs/relevant_bounding_box_pointrcnn.yaml 