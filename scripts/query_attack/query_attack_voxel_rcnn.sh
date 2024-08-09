cd ../..
# rm -rf output/train/query_attack_voxel_rcnn_pp
# rm -rf output/train/query_attack_voxel_rcnn_pr
# rm -rf output/train/query_attack_voxel_rcnn_pr_s1
# rm -rf output/train/query_attack_voxel_rcnn_pv
# python black_box_attack.py --exp-name query_attack_voxel_rcnn_pp --learning-rate 0.5 --cfg-file configs/attack_configs/kitti_black_box/pointpillar/query_attack_voxel_rcnn.yaml
# python black_box_attack.py --exp-name query_attack_voxel_rcnn_pr --learning-rate 0.05 --cfg-file configs/attack_configs/kitti_black_box/pointrcnn/query_attack_voxel_rcnn.yaml
# python black_box_attack.py --device $1 --exp-name query_attack_voxel_rcnn_pr_s1 --surrogate-stage-1 --learning-rate 0.05 --cfg-file configs/attack_configs/kitti_black_box/pointrcnn/query_attack_voxel_rcnn.yaml
# python black_box_attack.py --exp-name query_attack_voxel_rcnn_pv --learning-rate 0.0005 --cfg-file configs/attack_configs/kitti_black_box/pvrcnn/query_attack_voxel_rcnn.yaml
