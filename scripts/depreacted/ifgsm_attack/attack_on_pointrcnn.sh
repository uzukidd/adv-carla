cd ../..
rm -rf output/train/pointrcnn_roihead_attack_ifgsm_005
rm -rf output/train/pointrcnn_roihead_attack_ifgsm_0005
rm -rf output/train/pointrcnn_roihead_attack_ifgsm_00005
python attack.py --exp-name pointrcnn_roihead_attack_ifgsm_005 --roihead-attack --learning-rate 0.05 --optim ifgsm --cfg-file configs/attack_configs/kitti/relevant_bounding_box_pointrcnn.yaml 
python attack.py --exp-name pointrcnn_roihead_attack_ifgsm_0005 --roihead-attack --learning-rate 0.005 --optim ifgsm --cfg-file configs/attack_configs/kitti/relevant_bounding_box_pointrcnn.yaml 
python attack.py --exp-name pointrcnn_roihead_attack_ifgsm_00005 --roihead-attack --learning-rate 0.0005 --optim ifgsm --cfg-file configs/attack_configs/kitti/relevant_bounding_box_pointrcnn.yaml 
