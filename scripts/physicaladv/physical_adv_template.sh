cd ../..
rm -rf output/train/PhysicalAdv/$1
python main.py --exp-name PhysicalAdv/$1 --learning-rate 0.005 --headbox-attack --optim adam --iou-frozen --cfg-file configs/attack_configs/kitti/relevant_bounding_box_$1.yaml --device $2