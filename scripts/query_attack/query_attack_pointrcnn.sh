cd ../..
# rm -rf output/train/query_attack_pointrcnn_pp
rm -rf output/train/query_attack_pointrcnn_pv
rm -rf output/train/query_attack_pointrcnn_pv_s1
# rm -rf output/train/query_attack_pointrcnn_vr
# rm -rf output/train/query_attack_pointrcnn_vr_s1
# python black_box_attack.py --exp-name query_attack_pointrcnn_pp --learning-rate 0.5 --cfg-file configs/attack_configs/kitti_black_box/pointpillar/query_attack_pointrcnn.yaml
python black_box_attack.py  --device $1 --exp-name query_attack_pointrcnn_pv --learning-rate 0.005 --cfg-file configs/attack_configs/kitti_black_box/pvrcnn/query_attack_pointrcnn.yaml
python black_box_attack.py  --device $1 --exp-name query_attack_pointrcnn_pv_s1 --surrogate-stage-1 --learning-rate 0.005 --cfg-file configs/attack_configs/kitti_black_box/pvrcnn/query_attack_pointrcnn.yaml
# python black_box_attack.py  --device $1 --exp-name query_attack_pointrcnn_vr_s1 --surrogate-stage-1 --learning-rate 0.0005 --cfg-file configs/attack_configs/kitti_black_box/voxel_rcnn/query_attack_pointrcnn.yaml
# python black_box_attack.py  --device $1 --exp-name query_attack_pointrcnn_vr --learning-rate 0.0005 --cfg-file configs/attack_configs/kitti_black_box/voxel_rcnn/query_attack_pointrcnn.yaml
