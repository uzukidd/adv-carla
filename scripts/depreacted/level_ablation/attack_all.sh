{
bash scale_ablation.sh pointrcnn 0 0.05
bash scale_ablation.sh voxel_rcnn_car 0 0.0005
} &
{
bash scale_ablation.sh pointpillar 1 0.005
} &
wait