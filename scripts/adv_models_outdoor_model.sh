cd ..
# rm -rf output/train/inference_models_outdoor_demo
rm -rf output/train/inference_models_outdoor_demo/pointrcnn_adv
# python main.py --exp-name inference_models_outdoor_demo/pointpillar  --cfg-file configs/attack_configs/outdoor_demo/inference_pointpillar.yaml
python main.py --exp-name inference_models_outdoor_demo/pointrcnn_adv --adversarial-inference  --cfg-file configs/attack_configs/outdoor_demo/adv_inference_pointrcnn.yaml
# python main.py --exp-name inference_models_outdoor_demo/pvrcnn  --cfg-file configs/attack_configs/outdoor_demo/inference_pvrcnn.yaml
# python main.py --exp-name inference_models_outdoor_demo/voxel_rcnn_car  --cfg-file configs/attack_configs/outdoor_demo/inference_voxel_rcnn_car.yaml
# python main.py --exp-name inference_models_outdoor_demo/second  --cfg-file configs/attack_configs/outdoor_demo/inference_second.yaml
