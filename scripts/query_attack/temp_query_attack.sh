cd ../..
rm -rf output/train/query_attack_$1_vr
rm -rf output/train/query_attack_$1_vr_s1
python black_box_attack.py  --device $2 --exp-name query_attack_$1_vr --learning-rate 0.0005 --cfg-file configs/attack_configs/kitti_black_box/voxel_rcnn/query_attack_$1.yaml
python black_box_attack.py  --device $2 --exp-name query_attack_$1_vr_s1 --surrogate-stage-1 --learning-rate 0.0005 --cfg-file configs/attack_configs/kitti_black_box/voxel_rcnn/query_attack_$1.yaml
