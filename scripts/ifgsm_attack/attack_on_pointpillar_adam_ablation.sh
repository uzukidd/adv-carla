cd ../..
# rm -rf output/train/pointpillar_roihead_attack_adam_005
# rm -rf output/train/pointpillar_roihead_attack_adam_0005
# rm -rf output/train/pointpillar_roihead_attack_adam_00005
# python attack.py --exp-name pointpillar_roihead_attack_adam_005 --roihead-attack --learning-rate 0.05 --iou-frozen --cfg-file configs/attack_configs/kitti/relevant_bounding_box_pointpillar.yaml 
# python attack.py --exp-name pointpillar_roihead_attack_adam_0005 --roihead-attack --learning-rate 0.005 --iou-frozen --cfg-file configs/attack_configs/kitti/relevant_bounding_box_pointpillar.yaml 
# python attack.py --exp-name pointpillar_roihead_attack_adam_00005 --roihead-attack --learning-rate 0.0005 --iou-frozen --cfg-file configs/attack_configs/kitti/relevant_bounding_box_pointpillar.yaml 
# rm -rf output/train/pointpillar_roihead_attack_ifgsm_005
# rm -rf output/train/pointpillar_roihead_attack_ifgsm_0005
# rm -rf output/train/pointpillar_roihead_attack_ifgsm_00005
# python attack.py --exp-name pointpillar_roihead_attack_ifgsm_005 --roihead-attack --learning-rate 0.05 --iou-frozen --optim ifgsm --cfg-file configs/attack_configs/kitti/relevant_bounding_box_pointpillar.yaml 
# python attack.py --exp-name pointpillar_roihead_attack_ifgsm_0005 --roihead-attack --learning-rate 0.005 --iou-frozen --optim ifgsm --cfg-file configs/attack_configs/kitti/relevant_bounding_box_pointpillar.yaml 
# python attack.py --exp-name pointpillar_roihead_attack_ifgsm_00005 --roihead-attack --learning-rate 0.0005 --iou-frozen --optim ifgsm --cfg-file configs/attack_configs/kitti/relevant_bounding_box_pointpillar.yaml 
rm -rf output/train/pointpillar_roihead_attack_ifgsm_05
rm -rf output/train/pointpillar_roihead_attack_ifgsm_000005
python attack.py --exp-name pointpillar_roihead_attack_ifgsm_05 --roihead-attack --learning-rate 0.5 --iou-frozen --optim ifgsm --cfg-file configs/attack_configs/kitti/relevant_bounding_box_pointpillar.yaml 
python attack.py --exp-name pointpillar_roihead_attack_ifgsm_000005 --roihead-attack --learning-rate 0.00005 --iou-frozen --optim ifgsm --cfg-file configs/attack_configs/kitti/relevant_bounding_box_pointpillar.yaml 
rm -rf output/train/pointpillar_roihead_attack_adam_05
rm -rf output/train/pointpillar_roihead_attack_adam_000005
python attack.py --exp-name pointpillar_roihead_attack_adam_05 --roihead-attack --learning-rate 0.5 --iou-frozen --cfg-file configs/attack_configs/kitti/relevant_bounding_box_pointpillar.yaml 
python attack.py --exp-name pointpillar_roihead_attack_adam_000005 --roihead-attack --learning-rate 0.00005 --iou-frozen --cfg-file configs/attack_configs/kitti/relevant_bounding_box_pointpillar.yaml 
