cd ../..
# rm -rf output/train/query_attack_pvrcnn_pp
rm -rf output/train/query_attack_pvrcnn_pr
# rm -rf output/train/query_attack_pvrcnn_s1
# python black_box_attack.py --exp-name query_attack_pvrcnn_pp --learning-rate 0.005 --device 1 --cfg-file configs/attack_configs/kitti_black_box/pointpillar/query_attack_pvrcnn.yaml
python black_box_attack.py --exp-name query_attack_pvrcnn_pr --learning-rate 0.005 --cfg-file configs/attack_configs/kitti_black_box/pointrcnn/query_attack_pvrcnn.yaml
# python black_box_attack.py --exp-name query_attack_pvrcnn_s1 --surrogate-stage-1 --learning-rate 0.005 --cfg-file configs/attack_configs/kitti_black_box/pointrcnn/query_attack_pvrcnn.yaml


# python black_box_attack.py --exp-name query_attack --learning-rate 0.005 --cfg-file configs/attack_configs/kitti_black_box/query_attack_pvrcnn.yaml
