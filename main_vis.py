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
from pcdet.datasets import DatasetTemplate, KittiDataset
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

CFG_FILE = "./cfgs/kitti_models/pointrcnn.yaml"
KITTI_DATA_PATH = "/home/ksas/Public/datasets/KITTI"
DATA_PATH = "/home/ksas/Public/datasets/carla_test_dataset/points"
GT_PATH = "/home/ksas/Public/datasets/carla_test_dataset/gt"
CKPT_PATH = "/home/ksas/Public/model_zoo/pcdet/pointrcnn_7870.pth"

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, gt_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.gt_path = gt_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        if self.gt_path: 
            gts_file_list = glob.glob(str(gt_path / f'*{self.ext}')) if self.gt_path.is_dir() else [self.gt_path]
            gts_file_list.sort()


        data_file_list.sort()
        
        self.sample_file_list = data_file_list
        self.gts_file_list = gts_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        clean_data = self.__getitem_aux__(index)
        #adv_data = self.__getitem_aux__(index)

        return clean_data #, adv_data
    
    def __getitem_aux__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
            if self.gt_path:
                gts = np.fromfile(self.gts_file_list[index], dtype=np.float32).reshape(-1, 4)
                
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
            if self.gt_path:
                gts = np.load(self.gts_file_list[index])
        else:
            raise NotImplementedError
        
        gts_name = np.array(["Car"] * gts.shape[0])

        input_dict = {
            'points': points,
            'frame_id': index,
            "gt_names": gts_name,
            "gt_boxes": gts,

        }

        data_dict = self.prepare_data(data_dict=input_dict)
        
        return data_dict

def plot_pointcloud(mesh, title=""):
    # Sample points uniformly from the surface of the mesh.
    points = sample_points_from_meshes(mesh, 50)
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)    
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(190, 30)
    plt.show()
    
def generate_mono_adv_patch(level:int = 0):
    mSphere = ico_sphere(level)
    scale = Scale(0.5, 0.5, 0.5)
    new_verts = scale.transform_points(mSphere.verts_padded())
    mSphere = mSphere.update_padded(new_verts) 
    return mSphere

def init_adv_patch(gt_boxes):
    n = gt_boxes.size(0)
    patches = []
    pos_trans = []
    deform_verts = []
    for i in range(n): 
        patch = generate_mono_adv_patch().cuda()
        deform_vert = torch.full(patch.verts_packed().shape, 0.0).cuda().contiguous()
        deform_vert.requires_grad_()
        bottom = patch.verts_packed()[:, 2].min()
        pos = torch.tensor([gt_boxes[i][0], gt_boxes[i][1], gt_boxes[i][2] + gt_boxes[i][5] / 2 - bottom]).cuda()
        
        patches.append(patch)
        deform_verts.append(deform_vert)
        pos_trans.append(pos)
        
    return patches, deform_verts, pos_trans
        
def attach_adv_patch_scene(points, patches, pos_trans, deform_verts, sample_amount = 50):
    n = patches.__len__()
    pts_set = [points]
    
    for i in range(n):
        trans_deform_vert = deform_verts[i] + pos_trans[i][None, :]
        deformed_patch = patches[i].offset_verts(trans_deform_vert)
        patch_sampled = sample_points_from_meshes(deformed_patch, sample_amount)
        pts_set.append(patch_sampled.view(-1, 3))
        
    return torch.concatenate(pts_set)

def predict(model, points, data_template):
    data_dict = {
        
    }
    data_dict["frame_id"] = data_template["frame_id"]
    data_dict["use_lead_xyz"] = data_template["use_lead_xyz"].clone().detach()
    data_dict["batch_size"] = data_template["batch_size"]
    data_dict["points"] = points
    
    pred_dicts = None
    
    with torch.no_grad() :
        model.eval()
        pred_dicts, _ = model.forward(data_dict) 

    return pred_dicts

def attack(model, points, gt_boxes, eps, data_template):
    data_dict = {
        
    }
    data_dict["gt_boxes"] = gt_boxes
    data_dict["frame_id"] = data_template["frame_id"]
    data_dict["use_lead_xyz"] = data_template["use_lead_xyz"].clone().detach()
    data_dict["batch_size"] = data_template["batch_size"]
    data_dict["points"] = points.clone().detach()
    data_dict["points"].requires_grad = True

    model.train()
    model.zero_grad()
    ret_dict, tb_dict, _ = model.forward(data_dict)
    
    print(ret_dict["loss"])
    ret_dict["loss"].backward()

    grad = data_dict["points"].grad.data
    grad[:, [0, 4]] = 0
    pert = eps * torch.sign(grad)
        
    return (data_dict["points"] + pert).clone().detach()

def patch_attack(model, points, gt_boxes, patches, deform_verts, pos_trans, eps, data_template):
    data_dict = {
        
    }
    data_dict["gt_boxes"] = gt_boxes
    data_dict["frame_id"] = data_template["frame_id"]
    data_dict["use_lead_xyz"] = data_template["use_lead_xyz"].clone().detach()
    data_dict["batch_size"] = data_template["batch_size"]
    
    new_points = attach_adv_patch_scene(points[:, 1:4], patches, pos_trans, deform_verts, sample_amount=50)
    new_points = F.pad(new_points, (1, 1), "constant", 0)
    data_dict["points"] = new_points

    model.train()
    model.zero_grad()
    ret_dict, tb_dict, _ = model.forward(data_dict)
    
    print(ret_dict["loss"])
    ret_dict["loss"].backward()

    for idx, deform_verts_elem in enumerate(deform_verts):
        grad = deform_verts_elem.grad.data
        pert = eps * torch.sign(grad)
        
        deform_verts[idx] = (deform_verts_elem + pert).clone().detach()
        deform_verts[idx].requires_grad_()

    
def main():

    cfg_from_yaml_file(CFG_FILE, cfg)
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(DATA_PATH), gt_path=Path(GT_PATH), ext=".npy", logger=logger
    )
    kitti_carla_dataset = KittiDataset(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False, root_path=Path(KITTI_DATA_PATH), logger=logger)
    print(kitti_carla_dataset.class_names)
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=kitti_carla_dataset)
    model.load_params_from_file(filename=CKPT_PATH, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    PointNet2MSG = model.module_list[0]
    PointHeadBox = model.module_list[1]
    PointRCNNHead = model.module_list[2]

    for module in model.module_list:
        print(module._get_name())
        
    # epses = [0.01, 0.05, 0.1, 0.15, 0.20]
    
    eps = 0.1
    for index in range(kitti_carla_dataset.__len__()):

        data_dict = kitti_carla_dataset[index]
        data_dict = kitti_carla_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)

        pred_dicts = predict(model, data_dict["points"], data_dict)

        gts = pred_dicts[0]['pred_boxes']
        gt_classes = torch.ones(gts.size(0)).cuda()
        gts = torch.concatenate([gts, gt_classes.view(-1, 1)], axis=1)
        gts = gts.view(1, -1, 8)

        patches, deform_verts, pos_trans = init_adv_patch(gts[0])
        
        ori_points = attach_adv_patch_scene(
            data_dict["points"][:, 1:4], patches, pos_trans, deform_verts, sample_amount=50)
        ori_points = F.pad(ori_points, (1, 1), "constant", 0)

        patch_attack(model, data_dict["points"], gts, patches, deform_verts, pos_trans, eps, data_dict)
        
        new_points = attach_adv_patch_scene(data_dict["points"][:, 1:4], patches, pos_trans, deform_verts, sample_amount=50)
        new_points = F.pad(new_points, (1, 1), "constant", 0)
        
        pred_dicts = predict(model, new_points, data_dict)
        
        with torch.no_grad():
            l2_loss = (new_points[:, 1:4] -
                    ori_points[:, 1:4]).pow(2).sum()
            print(f"eps={eps}, l2 loss={l2_loss.item()}")
        
        print(f"length of patch : {patches.__len__()}")
        
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
    
    

if __name__ == "__main__":
    main()