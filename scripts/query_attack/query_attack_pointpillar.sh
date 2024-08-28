cd ../..
rm -rf output/train/query_attack_pointpillar_pv
rm -rf output/train/query_attack_pointpillar_pr
# rm -rf output/train/query_attack_pointpillar_pr_s1
rm -rf output/train/query_attack_pointpillar_pv_s1
# rm -rf output/train/query_attack_pointpillar_vr
# rm -rf output/train/query_attack_pointpillar_vr_s1
python main.py  --device $1 --exp-name query_attack_pointpillar_pr --stage-2-loss-reduce-func logit_multiply_iou3d --iou-frozen --learning-rate 0.05 --cfg-file configs/attack_configs/kitti_black_box/pointrcnn/query_attack_pointpillar.yaml
# python black_box_attack.py  --device $1 --exp-name query_attack_pointpillar_pr_s1 --surrogate-stage-1 --learning-rate 0.05 --cfg-file configs/attack_configs/kitti_black_box/pointrcnn/query_attack_pointpillar.yaml
python main.py  --device $1 --exp-name query_attack_pointpillar_pv --learning-rate 0.0005 --stage-2-loss-reduce-func logit_multiply_iou3d --iou-frozen --cfg-file configs/attack_configs/kitti_black_box/pvrcnn/query_attack_pointpillar.yaml
python main.py  --device $1 --exp-name query_attack_pointpillar_pv_s1 --surrogate-stage-1 --iou-frozen --learning-rate 0.0005 --cfg-file configs/attack_configs/kitti_black_box/pvrcnn/query_attack_pointpillar.yaml
# python black_box_attack.py  --device $1 --exp-name query_attack_pointpillar_vr --surrogate-stage-1 --learning-rate 0.0005 --cfg-file configs/attack_configs/kitti_black_box/voxel_rcnn/query_attack_pointpillar.yaml
# python black_box_attack.py  --device $1 --exp-name query_attack_pointpillar_vr_s1 --learning-rate 0.0005 --cfg-file configs/attack_configs/kitti_black_box/voxel_rcnn/query_attack_pointpillar.yaml
