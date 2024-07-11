cd ../..
rm -rf output/train/query_attack_pointpillar_pv
# rm -rf output/train/query_attack_pointpillar
# rm -rf output/train/query_attack_pointpillar_s1
# python black_box_attack.py --exp-name query_attack_pointpillar --learning-rate 0.005 --cfg-file configs/attack_configs/kitti_black_box/pointrcnn/query_attack_pointpillar.yaml
# python black_box_attack.py --exp-name query_attack_pointpillar_s1 --surrogate-stage-1 --learning-rate 0.005 --cfg-file configs/attack_configs/kitti_black_box/pointrcnn/query_attack_pointpillar.yaml
# python black_box_attack.py --exp-name query_attack --learning-rate 0.005 --cfg-file configs/attack_configs/kitti_black_box/query_attack_pvrcnn.yaml
python black_box_attack.py --exp-name query_attack_pointpillar_pv --learning-rate 0.005 --cfg-file configs/attack_configs/kitti_black_box/pvrcnn/query_attack_pointpillar.yaml
