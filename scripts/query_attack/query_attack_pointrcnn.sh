cd ../..
# rm -rf output/train/query_attack_pointrcnn_pp
# rm -rf output/train/query_attack_pointrcnn_pr
rm -rf output/train/query_attack_pointrcnn_pv
# python black_box_attack.py --exp-name query_attack_pointrcnn_pp --learning-rate 0.005 --cfg-file configs/attack_configs/kitti_black_box/pointpillar/query_attack_pointrcnn.yaml
# python black_box_attack.py --exp-name query_attack --learning-rate 0.005 --cfg-file configs/attack_configs/kitti_black_box/query_attack_pvrcnn.yaml
python black_box_attack.py --exp-name query_attack_pointrcnn_pv --learning-rate 0.005 --cfg-file configs/attack_configs/kitti_black_box/pvrcnn/query_attack_pointrcnn.yaml
