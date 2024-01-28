import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.utils import ico_sphere
from pytorch3d.transforms import Scale
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene

from pytorch3d.io import load_obj

import matplotlib.pyplot as plt
import argparse
import glob
from pathlib import Path
import easydict

from scipy.spatial.transform import Rotation as R

try:
    # import open3d
    # from visual_utils import open3d_vis_utils as V
    # OPEN3D_FLAG = True
    ...
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.utils import loss_utils

import pdb
import os

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

        self.only_car = True

        if self.gt_path:
            
            self.dataset_labeling_format = None
            gt_file_list = os.listdir(self.gt_path)
            first_gt_file = os.path.join(self.gt_path, gt_file_list[0])

            if first_gt_file.endswith(".txt"):
                logger.info("----------------- Using KITTI Original Labeling Format (*.txt) -----------------")
                gts_file_list = glob.glob(str(gt_path / '*txt')) if self.gt_path.is_dir() else [self.gt_path]
                gts_file_list.sort()
                self.dataset_labeling_format = "txt"

            # elif first_gt_file.endswith(".npy"):
            #     # TODO: Add support for other format, such as kitii-carla
            #     pass

            else:
                gts_file_list = glob.glob(str(gt_path / f'*{self.ext}')) if self.gt_path.is_dir() else [self.gt_path]
                gts_file_list.sort()
                self.dataset_labeling_format = "bin"


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
            if self.gt_path and self.dataset_labeling_format == "bin":
                gts = np.load(self.gts_file_list[index])
            elif self.gt_path and self.dataset_labeling_format == "txt":
                gts = self.kitti_txt_label_process()
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

    def kitti_txt_label_process(self):
        
        # [x, y, z, dx, dy, dz, heading]
        gts = None

        for txt_file in self.gts_file_list:
            with open(txt_file, "r") as file:
                car_lines = [line.strip() for line in file.readlines() if line.startswith("Car")]
        
        return gts

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

def generate_custom_adv_patch(obj_path):
    verts, faces_all, file_property = load_obj(obj_path)
    verts, faces = verts, faces_all[0]
    obj_path = Meshes(verts=[verts], faces=[faces])
    return obj_path

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
    # patch = generate_mono_adv_patch(scale).cuda()

    # custom obj path
    custom_obj_path = "/home/ksas/chw_space/adv-carla/image.obj"
    patch = generate_custom_adv_patch(custom_obj_path).cuda()
    
    deform_vert = torch.full(patch.verts_packed().shape, 0.0).cuda().contiguous()
    deform_vert.requires_grad_()
    
    bottom = patch.verts_packed()[:, 2].min()

    # Set the center position of the patch in 3-d spaces
    pos_trans = []
    for i in range(n):
        # dz / 2?
        pos = torch.tensor([gt_boxes[i][0], gt_boxes[i][1], gt_boxes[i][2] + gt_boxes[i][5] / 2 - bottom]).cuda()
        pos_trans.append(pos)
    return patch, deform_vert, pos_trans

### TODO: Test Align for each adv patch

def align_heading_for_adv_patch_uni(gt_boxes, patch, deform_vert, pos_trans):
    n = gt_boxes.size(0)
    aligned_patches = []

    deform_vert_ = deform_vert

    # copy from func init_adv_patch_uni()
    
    pos_trans_ = []

    for i in range(n):
        # Extract heading from gt_boxes
        heading = gt_boxes[i, 6]

        # Rotate the patch and deform_vert based on the heading
        rotated_patch = rotate_patch(patch, heading)
        bottom = rotated_patch.verts_packed()[:, 2].min()
        rotated_deform_vert = rotate_deform_vert(deform_vert, heading)

        # Translate the rotated patch to the position of the gt_boxes
        translated_patch = translate_patch(rotated_patch, pos_trans[i])

        pos = torch.tensor([gt_boxes[i][0], gt_boxes[i][1], gt_boxes[i][2] + gt_boxes[i][5] / 2 - bottom]).cuda()
        pos_trans_.append(pos)

        aligned_patches.append(rotated_patch)

    return aligned_patches, deform_vert_, pos_trans_

def rotate_patch(patch, heading):
    # Rotate the patch based on the heading (in radians) around z-axis
    rotated_patch = patch.clone()
    rotated_patch.verts_packed()[:, :3] = rotate_points(rotated_patch.verts_packed()[:, :3], heading)
    return rotated_patch

def rotate_deform_vert(deform_vert, heading):
    # Assuming deform_vert contains vertex offsets in the local coordinate system
    # Rotate the deform_vert based on the heading (in radians) around z-axis
    rotated_deform_vert = deform_vert.clone()  # need to implement the rotation logic
    return rotated_deform_vert

def translate_patch(patch, translation):
    # Translate the patch to the specified position
    translated_patch = patch.clone()
    translated_patch.offset_verts_(translation)
    return translated_patch

def rotate_points(points, angle):
    # Rotate 3D points around the z-axis
    cos_theta = torch.cos(angle)
    sin_theta = torch.sin(angle)

    rotation_matrix = torch.tensor([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ], dtype=points.dtype, device=points.device)

    rotated_points = torch.matmul(points, rotation_matrix)
    return rotated_points

# def align_heading_for_adv_patch_uni(gt_boxes, init_patch):
#     # Extract heading from gt_boxes
#     heading = gt_boxes[:, 6].unsqueeze(1)  # Extract heading from gt_boxes

#     # Get the rotation matrix
#     rotation_matrix = get_rotation_matrix(heading)

#     # Apply the rotation to the patch vertices
#     rotated_patch_verts = rotate_patch(init_patch, rotation_matrix)

#     # Update the patch with the rotated vertices
#     # Check if rotated_patch_verts has the same batch dimension as init_patch. If not, expand it.
#     if rotated_patch_verts.shape[0] != init_patch.verts_packed().shape[0]:
#         rotated_patch_verts = rotated_patch_verts.expand(init_patch.verts_packed().shape[0], -1, -1)

#     init_patch = init_patch.update_padded(rotated_patch_verts)

#     return init_patch

# def get_rotation_matrix(heading):
#     # Convert heading from radians to degrees
#     heading_degrees = heading * (180.0 / 3.14159265358979323846)

#     # Create a 2D rotation matrix for each heading
#     rotation_matrix = torch.stack([torch.cos(heading_degrees), -torch.sin(heading_degrees),
#                                    torch.sin(heading_degrees), torch.cos(heading_degrees)], dim=1)

#     # Reshape to 2x2 matrix
#     rotation_matrix = rotation_matrix.view(-1, 2, 2)

#     # Pad to 3x3 matrix
#     rotation_matrix = F.pad(rotation_matrix, (0, 1, 0, 1), "constant", 0)

#     return rotation_matrix

# def rotate_patch(patch, rotation_matrix):
#     # Extract patch vertices
#     verts = patch.verts_packed()

#     # Get the original coordinates
#     coords = verts[:, :3]

#     # Apply the rotation to the coordinates using matrix multiplication
#     rotated_coords = torch.matmul(coords, rotation_matrix.transpose(1, 2))

#     # If verts[:, 3:] is not empty, combine rotated coordinates with original color and alpha
#     if verts[:, 3:].numel() > 0:
#         rotated_patch_verts = torch.cat([rotated_coords, verts[:, 3:].unsqueeze(2).expand(-1, -1, rotated_coords.size(2))], dim=1)
#     else:
#         # If verts[:, 3:] is empty, use rotated_coords alone
#         rotated_patch_verts = rotated_coords.unsqueeze(1)

#     return rotated_patch_verts

### TODO: Test End

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
        deformed_patch = patch[i].offset_verts(trans_deform_vert)
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

    deform_vert = torch.full(deform_ori.shape, 0.000001, device='cuda', requires_grad=True)
    opt = optim.Adam([deform_vert], lr=1e-2, weight_decay=0.)

    ClassificationLoss = loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)

    best_vert = deform_vert.clone()
    best_loss = 9999999
    for iteration in range(max_iter):
        opt.zero_grad()
        data_dict = {
            
        }
        data_dict["gt_boxes"] = gt_boxes
        data_dict["frame_id"] = data_template["frame_id"]
        data_dict["use_lead_xyz"] = data_template["use_lead_xyz"].clone().detach()
        data_dict["batch_size"] = data_template["batch_size"]

        new_points = attach_adv_patch_scene_uni(points[:, 1:4], patch, pos_trans, deform_vert, sample_amount=50)
        new_points = F.pad(new_points, (1,1), "constant", 0)
        data_dict["points"] = new_points

        #### training loss
        #model.train() # set loss easily but have potential problem
        #model.zero_grad()
        #ret_dict, tb_dict, _ = model.forward(data_dict)
        #
        #print(ret_dict["loss"])
        #loss = ret_dict["loss"]
        #ret_dict["loss"].backward()

        ## testing loss
        model.eval()
        #model.zero_grad()
        ret_dict, tb_dict = model.forward(data_dict)
        ### point_head.get_loss
        point_cls_preds = model.module_list[1].forward_ret_dict['point_cls_preds']
        #point_box_preds = model.module_list[1].forward_ret_dict['point_box_preds']
        algin_data = model.module_list[1].assign_targets(data_dict)
        point_cls_labels = algin_data['point_cls_labels']
        point_cls_labels = point_cls_labels.view(-1)
        point_cls_preds = point_cls_preds.view(-1, model.module_list[1].num_class)

        positives = (point_cls_labels > 0)
        negative_cls_weights = (point_cls_labels == 0) * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        pos_normalizer = positives.sum(dim=0).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), model.module_list[1].num_class + 1)
        one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = ClassificationLoss(point_cls_preds, one_hot_targets, weights=cls_weights)
        point_loss_cls = cls_loss_src.sum()

        #### roi_head.get_loss
        #forward_ret_dict = model.module_list[2].assign_targets(data_dict)
        #code_size = model.module_list[2].box_coder.code_size
        #rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels']
        #reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        #gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
        #roi_boxes3d = forward_ret_dict['rois']
        #gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size)

        ##### change OpenPCD/pcdet/models/roi_heads/pointrcnn_head.py adding rcnn_reg rcnn_cls to dict
        ##rcnn_reg = data_dict['rcnn_cls'] 
        ##rcnn_cls = data_dict['rcnn_reg']
        #rcnn_reg = data_dict['batch_cls_preds'] 
        #rcnn_cls = data_dict['batch_box_preds']

        #rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]
        #fg_mask = (reg_valid_mask > 0)
        #fg_sum = fg_mask.long().sum().item()
        #### get_box_cls_layer_loss
        #pdb.set_trace()
        ##batch_loss_cls = F.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
        ##cls_valid_mask = (rcnn_cls_labels >= 0).float()
        ##rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        #### get_box+reg_layer_loss
        ##loss_cfgs = model.module_list[2].model_cfg.LOSS_CONFIG
        ##WeightedSmoothL1Loss = loss_utils.WeightedSmoothL1Loss(code_weights = loss_cfgs['code_weights'])
        ##rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
        ##rois_anchor[:, 0:3] = 0
        ##rois_anchor[:, 6] = 0
        ##reg_targets = model.module_list[2].box_coder.encode_torch(gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor)
        ##rcnn_loss_reg = model.module_list[2].reg_loss_func(rcnn_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),reg_targets.unsqueeze(dim=0),)
        ##rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
        ##rcnn_loss_reg = rcnn_loss_reg * loss_cfgs['rcnn_reg_weight']
        ##rcnn_loss_reg = rcnn_loss_reg.item()



        #print(ret_dict[0]['pred_labels'].shape)
        #if iteration >9:
        #    pdb.set_trace()
        #deform_vert.retain_grad()
        #loss_cla = ClassificationLoss(ret_dict[0]['pred_labels'],gt_boxes[0,:,-1],ret_dict[0]['pred_scores'])
        #loss_det = loss_utils.get_corner_loss_lidar(ret_dict[0]['pred_boxes'], gt_boxes[0,:,:-1])
        #
        #loss = loss_cla.sum() + loss_det.sum()
        #loss.backward()
        point_loss_cls.backward()
        opt.step()


        ### save best
        if point_loss_cls < best_loss:
            best_vert = deform_vert.clone()
            best_loss = point_loss_cls

    return best_vert

