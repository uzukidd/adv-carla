cd ..
rm -rf output/eval/kitti-carla-eval
# rm -rf output/eval/kitti-carla-gtboxes
# python eval.py --exp-name kitti-carla --cfg-file configs/attack_configs/kitticarla/eval_pointrcnn.yaml
python eval.py --exp-name kitti-carla-eval --cfg-file configs/attack_configs/kitticarla/eval_pvrcnn.yaml
# python eval.py --exp-name kitti-carla-roihead --cfg-file configs/attack_configs/kitticarla/eval_pointrcnn.yaml
# python eval.py --exp-name kitti-carla-gtboxes --prepared-gtboxes --cfg-file configs/attack_configs/kitticarla/inference_voxel_rcnn_car.yaml