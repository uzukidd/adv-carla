cd ..
rm -rf output/train/full_attack_pointpillar
rm -rf output/train/iou_frozen_pointpillar
rm -rf output/train/logit_frozen_pointpillar
python attack.py --headbox-attack --exp-name full_attack_pointpillar --eval-init-patch
python attack.py --headbox-attack --exp-name iou_frozen_pointpillar --iou-frozen
python attack.py --headbox-attack --exp-name logit_frozen_pointpillar --logit-frozen