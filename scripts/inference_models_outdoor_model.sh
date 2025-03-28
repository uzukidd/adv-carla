cd ..
rm -rf output/train/inference_models_outdoor_demo/pointrcnn
rm -rf output/train/inference_models_outdoor_demo/pointrcnn_adversarial
rm -rf output/train/inference_models_outdoor_demo/pointrcnn_evaluation

python main.py --exp-name inference_models_outdoor_demo/pointrcnn \
    --cfg-file configs/attack_configs/outdoor_demo/inference_pointrcnn.yaml

python rooftop_aproximate_outdoor.py  \
    --dataset-config-path configs/dataset_configs/outdoor_demo_dataset.yaml \
    --gtboxes-path output/train/inference_models_outdoor_demo/pointrcnn/gtboxes.pt \
    --output-path output/train/inference_models_outdoor_demo/pointrcnn/rooftop_appro.pkl

python main.py --exp-name inference_models_outdoor_demo/pointrcnn_adversarial \
    --cfg-file configs/attack_configs/outdoor_demo/inference_pointrcnn.yaml \
    --adversarial-inference \
    --rooftop-annotate-path output/train/inference_models_outdoor_demo/pointrcnn/rooftop_appro.pkl \
    --patch-ckpt output/train/PhysicalAdv/pointrcnn/final_adversarial_patch_checkpoint.pt \
    --gtboxes-path output/train/inference_models_outdoor_demo/pointrcnn/gtboxes.pt
