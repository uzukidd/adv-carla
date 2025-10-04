cd ../..
rm -rf output/train/query_attack_second_pp
# rm -rf output/train/query_attack_second_pr
# rm -rf output/train/query_attack_second_pr_s1
rm -rf output/train/query_attack_second_pv_s1
# rm -rf output/train/query_attack_second_vr
# rm -rf output/train/query_attack_second_vr_s1
rm -rf output/train/query_attack_second_pv
python main.py --device $1 --exp-name query_attack_second_pp --learning-rate 0.005 --stage-2-loss-reduce-func logit_multiply_iou3d --iou-frozen --cfg-file configs/attack_configs/kitti_black_box/pointpillar/query_attack_second.yaml
# python black_box_attack.py --exp-name query_attack_second_pr --learning-rate 0.05 --cfg-file configs/attack_configs/kitti_black_box/pointrcnn/query_attack_second.yaml
# python black_box_attack.py --device $1 --exp-name query_attack_second_pr_s1 --surrogate-stage-1 --learning-rate 0.05 --cfg-file configs/attack_configs/kitti_black_box/pointrcnn/query_attack_second.yaml
python main.py --device $1 --exp-name query_attack_second_pv_s1 --surrogate-stage-1 --learning-rate 0.0005  --iou-frozen --cfg-file configs/attack_configs/kitti_black_box/pvrcnn/query_attack_second.yaml
# python black_box_attack.py --device $1 --exp-name query_attack_second_vr --learning-rate 0.0005 --cfg-file configs/attack_configs/kitti_black_box/voxel_rcnn/query_attack_second.yaml
# python black_box_attack.py --device $1 --exp-name query_attack_second_vr_s1 --surrogate-stage-1 --learning-rate 0.0005 --cfg-file configs/attack_configs/kitti_black_box/voxel_rcnn/query_attack_second.yaml

python main.py  --device $1 --exp-name query_attack_second_pv --stage-2-loss-reduce-func logit_multiply_iou3d --iou-frozen --learning-rate 0.0005 --cfg-file configs/attack_configs/kitti_black_box/pvrcnn/query_attack_second.yaml
