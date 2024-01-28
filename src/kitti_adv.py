### pytorch
import torch


### pytorch3d
import pytorch3d


### pcdet
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.models import build_network, load_data_to_gpu

### math
import numpy as np

### common
import argparse
from pathlib import Path

### debug
import pdb

### tools
import sys
present_dir = sys.argv[0]
root_workspace_dir = present_dir.split("src")[0]
sys.path.append(root_workspace_dir)
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="Adv with KITTI dataset demo")
    parser.add_argument("--cfg_file", type=str, default="./cfgs/kitti_models/pointrcnn.yaml", help="specify the config for demo")
    parser.add_argument("--data_path", type=str, default="/home/chw/Public/datasets/KITTI/training/velodyne", help="specify the point cloud data file or directory")
    parser.add_argument("--gt_path", type=str, default="/home/chw/Public/datasets/KITTI/training/label", help="specify the ground truth file or directory")
    parser.add_argument("--ckpt", type=str, default="/home/chw/Public/nn_models/pointrcnn_7870.pth", help="specify the pretrained model")
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    # load config
    args, cfg = parse_args()

    logger = common_utils.create_logger()
    logger.info('----------------- KITTI Adv-------------------------')


    ### load kitti dataset samples
    kitti_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=Path(args.data_path),
        gt_path=Path(args.gt_path),
        logger=logger
    )

    logger.info("Total samples for KITTI dataset: %d" % len(kitti_dataset))


    ### load models
    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=kitti_dataset
    )

    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()


if __name__ == "__main__":
    main()
    