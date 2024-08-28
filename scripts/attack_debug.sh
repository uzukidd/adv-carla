cd ..
rm -rf output/train/debug
python main.py --exp-name debug --roihead-attack --logit-frozen --stage-2-loss-reduce-func physical_loss --optim ifgsm --learning-rate 0.05 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_pointrcnn.yaml --device $1
python main.py --eval-init-patch --cfg-file configs/attack_configs/kitti/relevant_bounding_box_pointrcnn.yaml --device $1
