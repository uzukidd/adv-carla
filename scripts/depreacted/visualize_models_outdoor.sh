cd ..
rm -rf output/train/inference_models_outdoor_vis
# python main.py --exp-name inference_models_outdoor_demo/pointpillar  --cfg-file configs/attack_configs/outdoor_demo/inference_pointpillar.yaml
python main.py --exp-name inference_models_outdoor_vis/pointrcnn  --cfg-file configs/attack_configs/outdoor_demo/visualization_pointrcnn.yaml
# python main.py --exp-name inference_models_outdoor_demo/pvrcnn  --cfg-file configs/attack_configs/outdoor_demo/inference_pvrcnn.yaml
# python main.py --exp-name inference_models_outdoor_demo/voxel_rcnn_car  --cfg-file configs/attack_configs/outdoor_demo/inference_voxel_rcnn_car.yaml
# python main.py --exp-name inference_models_outdoor_demo/second  --cfg-file configs/attack_configs/outdoor_demo/inference_second.yaml
