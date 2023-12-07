import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

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

import pdb


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
    
def generate_mono_adv_patch(scale, level: int=0):
    mSphere = ico_sphere(level)
    new_verts = scale.transform_points(mSphere.verts_padded())
    mSphere = mSphere.update_padded(new_verts) 
    return mSphere

def init_adv_patch(gt_boxes, scale):
    n = gt_boxes.size(0)
    patches = []
    pos_trans = []
    deform_verts = []
    for i in range(n): 
        patch = generate_mono_adv_patch(scale).cuda()
        deform_vert = torch.full(patch.verts_packed().shape, 0.0).cuda().contiguous()
        deform_vert.requires_grad_()
        bottom = patch.verts_packed()[:, 2].min()
        pos = torch.tensor([gt_boxes[i][0], gt_boxes[i][1], gt_boxes[i][2] + gt_boxes[i][5] / 2 - bottom]).cuda()
        
        patches.append(patch)
        deform_verts.append(deform_vert)
        pos_trans.append(pos)
        
    return patches, deform_verts, pos_trans

def init_adv_patch_uni(gt_boxes, scale):
    n = gt_boxes.size(0)
    patch = generate_mono_adv_patch(scale).cuda()
    deform_vert = torch.full(patch.verts_packed().shape, 0.0).cuda().contiguous()
    deform_vert.requires_grad_()
    bottom = patch.verts_packed()[:, 2].min()

    pos_trans = []
    for i in range(n):
        pos = torch.tensor([gt_boxes[i][0], gt_boxes[i][1], gt_boxes[i][2] + gt_boxes[i][5] / 2 - bottom]).cuda()
        pos_trans.append(pos)
    return patch, deform_vert, pos_trans
        
def attach_adv_patch_scene(points, patches, pos_trans, deform_verts, sample_amount = 50):
    n = patches.__len__()
    pts_set = [points]
    
    for i in range(n):
        trans_deform_vert = deform_verts[i] + pos_trans[i][None, :]
        deformed_patch = patches[i].offset_verts(trans_deform_vert)
        patch_sampled = sample_points_from_meshes(deformed_patch, sample_amount)
        pts_set.append(patch_sampled.view(-1, 3))
        
    return torch.concatenate(pts_set)
    
def attach_adv_patch_scene_uni(points, patch, pos_trans, deform_vert, sample_amount = 50):
    n = pos_trans.__len__()
    pts_set = [points]
    
    for i in range(n):
        trans_deform_vert = deform_vert + pos_trans[i][None, :]
        deformed_patch = patch.offset_verts(trans_deform_vert)
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

    model.train() # set loss easily but have potential problem
    model.zero_grad()
    ret_dict, tb_dict, _ = model.forward(data_dict)
    
    print(ret_dict["loss"])
    ret_dict["loss"].backward()

    for idx, deform_verts_elem in enumerate(deform_verts):
        grad = deform_verts_elem.grad.data
        pert = eps * torch.sign(grad)
        
        deform_verts[idx] = (deform_verts_elem + pert).clone().detach()
        deform_verts[idx].requires_grad_()


def patch_obj_attack(model, points, gt_boxes, patch, deform_ori, pos_trans, eps, data_template, max_iter):
    data_dict = {
        
    }
    data_dict["gt_boxes"] = gt_boxes
    data_dict["frame_id"] = data_template["frame_id"]
    data_dict["use_lead_xyz"] = data_template["use_lead_xyz"].clone().detach()
    data_dict["batch_size"] = data_template["batch_size"]
    

    deform_vert = torch.full(deform_ori.shape, 0.000001, device='cuda', requires_grad=True)
    opt = optim.Adam([deform_vert], lr=1e-2, weight_decay=0.)
    model.train()

    for iteration in range(max_iter):
        opt.zero_grad()

        new_points = attach_adv_patch_scene_uni(points[:, 1:4], patch, pos_trans, deform_vert, sample_amount=50)
        new_points = F.pad(new_points, (1,1), "constant", 0)
        data_dict["points"] = new_points

        model.zero_grad()
        ret_dict, tb_dict, _ = model.forward(data_dict)
        ret_dict["loss"].backward()
        opt.step()
    return deform_vert

