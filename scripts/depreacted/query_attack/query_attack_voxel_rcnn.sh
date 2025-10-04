cd ../..
rm -rf output/train/query_attack_voxel_rcnn_pp
rm -rf  output/train/query_attack_voxel_rcnn_pv_s1
rm -rf  output/train/query_attack_voxel_rcnn_pv
# rm -rf output/train/query_attack_voxel_rcnn_pr
# rm -rf output/train/query_attack_voxel_rcnn_pr_s1
# rm -rf output/train/query_attack_voxel_rcnn_pv
python main.py --device $1 --exp-name query_attack_voxel_rcnn_pp --stage-2-loss-reduce-func logit_multiply_iou3d --iou-frozen --learning-rate 0.005 --cfg-file configs/attack_configs/kitti_black_box/pointpillar/query_attack_voxel_rcnn.yaml
# python black_box_attack.py --exp-name query_attack_voxel_rcnn_pr --learning-rate 0.0005 --cfg-file configs/attack_configs/kitti_black_box/pointrcnn/query_attack_voxel_rcnn.yaml
# python black_box_attack.py --device $1 --exp-name query_attack_voxel_rcnn_pr_s1 --surrogate-stage-1 --learning-rate 0.05 --cfg-file configs/attack_configs/kitti_black_box/pointrcnn/query_attack_voxel_rcnn.yaml
python main.py --device $1 --exp-name query_attack_voxel_rcnn_pv --stage-2-loss-reduce-func logit_multiply_iou3d --iou-frozen --learning-rate 0.0005 --cfg-file configs/attack_configs/kitti_black_box/pvrcnn/query_attack_voxel_rcnn.yaml
python main.py --device $1 --exp-name query_attack_voxel_rcnn_pv_s1 --surrogate-stage-1  --iou-frozen --learning-rate 0.0005 --cfg-file configs/attack_configs/kitti_black_box/pvrcnn/query_attack_voxel_rcnn.yaml
