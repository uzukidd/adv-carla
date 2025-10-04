cd ..
python main.py --exp-name inference_models_outdoor_demo/pointrcnn_phyAdv \
    --cfg-file configs/attack_configs/outdoor_demo/inference_pointrcnn.yaml \
    --adversarial-inference \
    --rooftop-annotate-path output/train/inference_models_outdoor_demo/pointrcnn/rooftop_appro.pkl \
    --patch-ckpt output/train/PhysicalAdv/pointrcnn/final_adversarial_patch_checkpoint.pt \
    --gtboxes-path output/train/inference_models_outdoor_demo/pointrcnn/gtboxes.pt

python main.py --exp-name inference_models_outdoor_demo/pointrcnn_mr10 \
    --cfg-file configs/attack_configs/outdoor_demo/inference_pointrcnn.yaml \
    --adversarial-inference \
    --rooftop-annotate-path output/train/inference_models_outdoor_demo/pointrcnn/rooftop_appro.pkl \
    --patch-ckpt output/train/loss_ablation/pointrcnn/misrecognize_10/final_adversarial_patch_checkpoint.pt \
    --gtboxes-path output/train/inference_models_outdoor_demo/pointrcnn/gtboxes.pt

python rooftop_aproximate_outdoor.py \
 --dataset-config-path configs/dataset_configs/outdoor_demo_dataset.yaml \
 --gtboxes-path output/train/inference_models_outdoor_demo/pointrcnn_phyAdv/adversarial_pre_scores.pt\
 --predicted-boxes-path output/train/inference_models_outdoor_demo/pointrcnn_mr10/adversarial_boxes.pt\
 --visualize