cd ..
# rm -rf output/train/inference_models
python main.py --exp-name inference_models/pointpillar  --cfg-file configs/attack_configs/kitticarla/inference_pointpillar.yaml
python main.py --exp-name inference_models/pointrcnn  --cfg-file configs/attack_configs/kitticarla/inference_pointrcnn.yaml
python main.py --exp-name inference_models/pvrcnn  --cfg-file configs/attack_configs/kitticarla/inference_pvrcnn.yaml
python main.py --exp-name inference_models/voxel_rcnn_car  --cfg-file configs/attack_configs/kitticarla/inference_voxel_rcnn_car.yaml
python main.py --exp-name inference_models/second  --cfg-file configs/attack_configs/kitticarla/inference_second.yaml
