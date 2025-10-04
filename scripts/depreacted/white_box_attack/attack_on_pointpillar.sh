cd ../..
# rm -rf output/train/pointrcnn_full_attack
# rm -rf output/train/pointrcnn_headbox_attack
rm -rf output/train/pointpillar_roihead_attack
# python attack.py --exp-name pointrcnn_headbox_attack --headbox-attack --cfg-file configs/attack_configs/kitti/relevant_bounding_box_pointrcnn.yaml
# python attack.py --exp-name pointrcnn_full_attack --headbox-attack --roihead-attack --roi-head-weights 0.5 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_pointrcnn.yaml 
python attack.py --exp-name pointpillar_roihead_attack --roihead-attack --roi-head-weights 0.75 --cfg-file configs/attack_configs/kitti/relevant_bounding_box_pointpillar.yaml 
