cd ../..
rm -rf output/train/query_attack_pvrcnn_pp
# rm -rf output/train/query_attack_pvrcnn_pr
# rm -rf output/train/query_attack_pvrcnn_pr_s1
# rm -rf output/train/query_attack_pvrcnn_vr
# rm -rf output/train/query_attack_pvrcnn_vr_s1
python main.py --device $1  --exp-name query_attack_pvrcnn_pp --learning-rate 0.005 --stage-2-loss-reduce-func logit_multiply_iou3d --iou-frozen --cfg-file configs/attack_configs/kitti_black_box/pointpillar/query_attack_pvrcnn.yaml
# python black_box_attack.py --exp-name query_attack_pvrcnn_pr --learning-rate 0.05 --cfg-file configs/attack_configs/kitti_black_box/pointrcnn/query_attack_pvrcnn.yaml
# python black_box_attack.py --device $1 --exp-name query_attack_pvrcnn_pr_s1 --surrogate-stage-1  --learning-rate 0.05 --cfg-file configs/attack_configs/kitti_black_box/pointrcnn/query_attack_pvrcnn.yaml

# python black_box_attack.py --exp-name query_attack_pvrcnn_s1 --surrogate-stage-1 --learning-rate 0.005 --cfg-file configs/attack_configs/kitti_black_box/pointrcnn/query_attack_pvrcnn.yaml


# python black_box_attack.py --exp-name query_attack --learning-rate 0.005 --cfg-file configs/attack_configs/kitti_black_box/query_attack_pvrcnn.yaml
# python black_box_attack.py --device $1 --exp-name query_attack_pvrcnn_vr_s1 --surrogate-stage-1  --learning-rate 0.0005 --cfg-file configs/attack_configs/kitti_black_box/voxel_rcnn/query_attack_pvrcnn.yaml
# python black_box_attack.py --device $1 --exp-name query_attack_pvrcnn_vr  --learning-rate 0.0005 --cfg-file configs/attack_configs/kitti_black_box/voxel_rcnn/query_attack_pvrcnn.yaml
