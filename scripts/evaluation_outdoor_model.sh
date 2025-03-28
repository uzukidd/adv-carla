cd ..
rm -rf output/train/inference_models_outdoor_demo/pointrcnn_evaluation/
python main.py --exp-name inference_models_outdoor_demo/pointrcnn_evaluation/vanilla \
    --cfg-file configs/attack_configs/outdoor_demo/eval_pointrcnn.yaml \
    --rooftop-annotate-path output/train/inference_models_outdoor_demo/pointrcnn/rooftop_appro.pkl \
    --patch-ckpt output/train/PhysicalAdv/pointrcnn/initial_patch_checkpoint.pt \
    --gtboxes-path output/train/inference_models_outdoor_demo/pointrcnn/gtboxes.pt

python main.py --exp-name inference_models_outdoor_demo/pointrcnn_evaluation/phyadv \
    --cfg-file configs/attack_configs/outdoor_demo/eval_pointrcnn.yaml \
    --rooftop-annotate-path output/train/inference_models_outdoor_demo/pointrcnn/rooftop_appro.pkl \
    --patch-ckpt output/train/PhysicalAdv/pointrcnn/final_adversarial_patch_checkpoint.pt \
    --gtboxes-path output/train/inference_models_outdoor_demo/pointrcnn/gtboxes.pt

python main.py --exp-name inference_models_outdoor_demo/pointrcnn_evaluation/MR10 \
    --cfg-file configs/attack_configs/outdoor_demo/eval_pointrcnn.yaml \
    --rooftop-annotate-path output/train/inference_models_outdoor_demo/pointrcnn/rooftop_appro.pkl \
    --patch-ckpt output/train/loss_ablation/pointrcnn/misrecognize_10/final_adversarial_patch_checkpoint.pt \
    --gtboxes-path output/train/inference_models_outdoor_demo/pointrcnn/gtboxes.pt

python main.py --exp-name inference_models_outdoor_demo/pointrcnn_evaluation/MR9 \
    --cfg-file configs/attack_configs/outdoor_demo/eval_pointrcnn.yaml \
    --rooftop-annotate-path output/train/inference_models_outdoor_demo/pointrcnn/rooftop_appro.pkl \
    --patch-ckpt output/train/loss_ablation/pointrcnn/misrecognize_9/final_adversarial_patch_checkpoint.pt \
    --gtboxes-path output/train/inference_models_outdoor_demo/pointrcnn/gtboxes.pt

python rooftop_aproximate_outdoor.py \
 --dataset-config-path configs/dataset_configs/outdoor_demo_dataset.yaml \
 --gtboxes-path output/train/inference_models_outdoor_demo/pointrcnn/gtboxes.pt\
 --predicted-boxes-path output/train/inference_models_outdoor_demo/pointrcnn/gtboxes.pt\
 --visualize

python rooftop_aproximate_outdoor.py  \
 --dataset-config-path configs/dataset_configs/outdoor_demo_dataset.yaml \
 --gtboxes-path output/train/inference_models_outdoor_demo/pointrcnn/gtboxes.pt \
 --patch-ckpt output/train/loss_ablation/pointrcnn/misrecognize_9/final_adversarial_patch_checkpoint.pt \
 --predicted-boxes-path output/train/inference_models_outdoor_demo/pointrcnn_mr10/adversarial_boxes.pt \
 --visualize

python rooftop_aproximate_outdoor.py  \
 --dataset-config-path configs/dataset_configs/outdoor_demo_dataset.yaml \
 --gtboxes-path output/train/inference_models_outdoor_demo/pointrcnn/gtboxes.pt \
 --patch-ckpt output/train/PhysicalAdv/pointrcnn/final_adversarial_patch_checkpoint.pt \
 --predicted-boxes-path output/train/inference_models_outdoor_demo/pointrcnn_phyAdv/adversarial_boxes.pt \
 --visualize

--gtboxes-path output/train/inference_models_outdoor_demo/pointrcnn_phyAdv/adversarial_boxes.pt \