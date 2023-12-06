import numpy as np
import torch
import torch.nn.functional as F

seed = 2024

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.set_device(0)

from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.utils import ico_sphere
from pytorch3d.transforms import Scale
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene


import matplotlib.pyplot as plt
import argparse
import glob
from pathlib import Path
import easydict

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False


from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

from utils import *
import pdb

#### path for data, meta info
CFG_FILE = "./cfgs/kitti_models/pointrcnn.yaml"
DATA_PATH = "/home/ksas/Public/datasets/carla_test_dataset/points"
GT_PATH = "/home/ksas/Public/datasets/carla_test_dataset/gt"
CKPT_PATH = "/home/ksas/Public/model_zoo/pcdet/pointrcnn_7870.pth"

cfg_from_yaml_file(CFG_FILE, cfg)
logger = common_utils.create_logger()
logger.info('-----------------Quick Demo of OpenPCDet-------------------------')

### load dataset
demo_dataset = DemoDataset(
    dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
    root_path=Path(DATA_PATH), gt_path=Path(GT_PATH), ext=".npy", logger=logger
)
print(demo_dataset.class_names)
logger.info(f'Total number of samples: \t{len(demo_dataset)}')

### load model
model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
model.load_params_from_file(filename=CKPT_PATH, logger=logger, to_cpu=True)
model.cuda()
model.eval()

PointNet2MSG = model.module_list[0]
PointHeadBox = model.module_list[1]
PointRCNNHead = model.module_list[2]

for module in model.module_list:
    print(module._get_name())

### set parameters  
epses = [0.01, 0.05, 0.1, 0.15, 0.20]
scale = Scale(0.5, 0.5, 0.5)

### loading data
for index in range(demo_dataset.__len__()):

    data_dict = demo_dataset[index]
    data_dict = demo_dataset.collate_batch([data_dict])
    load_data_to_gpu(data_dict)

    pred_dicts = predict(model, data_dict["points"], data_dict)

    gts = pred_dicts[0]['pred_boxes']
    gt_classes = torch.ones(gts.size(0)).cuda()
    gts = torch.concatenate([gts, gt_classes.view(-1, 1)], axis=1)
    gts = gts.view(1, -1, 8)

    patches, deform_verts, pos_trans = init_adv_patch(gts[0], scale)
    
    ori_points = attach_adv_patch_scene(
        data_dict["points"][:, 1:4], patches, pos_trans, deform_verts, sample_amount=50)
    ori_points = F.pad(ori_points, (1, 1), "constant", 0)

    patch_attack(model, data_dict["points"], gts, patches, deform_verts, pos_trans, epses[0], data_dict)
    
    new_points = attach_adv_patch_scene(data_dict["points"][:, 1:4], patches, pos_trans, deform_verts, sample_amount=50)
    new_points = F.pad(new_points, (1, 1), "constant", 0)
    
    pred_dicts = predict(model, new_points, data_dict)
    
    with torch.no_grad():
        l2_loss = (new_points[:, 1:4] -
                   ori_points[:, 1:4]).pow(2).sum()
        print(f"eps={epses[0]}, l2 loss={l2_loss.item()}")
    
    print(f"length of patch : {patches.__len__()}")
    pdb.set_trace()
    
    fig = plot_scene({
        "original": {
            "mesh_1": patches[0]
        },
        "adversarial": {
            "mesh_1": patches[0].offset_verts(deform_verts[0])
        },
        
    })
    fig.update_layout(height=1000, width=500)
    fig.show()

    V.draw_scenes(
        points=new_points.detach()[:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
        ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'], gt_boxes=gts[0][:, :7]
    )

    if not OPEN3D_FLAG:
        mlab.show(stop=True)
        
        data_dict = demo_dataset[0]
data_dict = demo_dataset.collate_batch([data_dict])
load_data_to_gpu(data_dict)

pred_dicts = predict(model, data_dict["points"], data_dict)

V.draw_scenes(
    points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
    ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'], gt_boxes=pred_dicts[0]['pred_boxes']
)

gts = pred_dicts[0]['pred_boxes']
gt_classes = torch.ones(gts.size(0)).cuda()
gts = torch.concatenate([gts, gt_classes.view(-1, 1)], axis=1)
gts = gts.view(1, -1, 8)

for eps in epses:

    adv_points = attack(model, data_dict["points"], gts, eps, data_dict)
    pred_dicts = predict(model, adv_points, data_dict)
    
    with torch.no_grad():
        l2_loss = (adv_points[:, 1:4] - data_dict["points"][:, 1:4]).pow(2).sum()
        print(f"eps={eps}, l2 loss={l2_loss.item()}")


    V.draw_scenes(
        points=adv_points[:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
        ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'], gt_boxes=gts[0][:, :7]
    )

if not OPEN3D_FLAG:
    mlab.show(stop=True)
