cd ..
rm -rf output/train/debug
python attack.py --verbose-epoch 1 --exp-name debug --cfg-file configs/attack_configs/relevant_bounding_box_pvrcnn.yaml