import numpy as np
import torch
import torch.nn.functional as F

from easydict import EasyDict

from pcdet.datasets.processor.data_processor import DataProcessor
from pcdet.utils import common_utils
from collections import defaultdict
from functools import partial
from typing import Union

def resgister_data_processor(name:str, module:callable):
    setattr(DataProcessor, name, module)

def voxel_collate_batch(batch_list, _unused=False):
    data_dict = defaultdict(list)
    for cur_sample in batch_list:
        for key, val in cur_sample.items():
            data_dict[key].append(val)
    batch_size = len(batch_list)
    ret = {}
    batch_size_ratio = 1

    for key, val in data_dict.items():
        try:
            if key in ['voxels', 'voxel_num_points']:
                if isinstance(val[0], list):
                    batch_size_ratio = len(val[0])
                    val = [i for item in val for i in item]

                # Retain gradient graph
                if isinstance(val[0], torch.Tensor):
                    ret[key] = torch.concatenate(val, axis=0)
                elif isinstance(val[0], np.ndarray):
                    ret[key] = np.concatenate(val, axis=0)

            elif key in ['points', 'voxel_coords']:
                coors = []
                if isinstance(val[0], list):
                    val =  [i for item in val for i in item]

                # Retain gradient graph
                if isinstance(val[0], torch.Tensor):
                    for i, coor in enumerate(val):
                        coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
                        coors.append(coor_pad)
                    ret[key] = torch.concatenate(coors, axis=0)

                elif isinstance(val[0], np.ndarray):
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
            elif key in ['gt_boxes']:
                max_gt = max([len(x) for x in val])
                if isinstance(val[0], np.ndarray):
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]

                elif isinstance(val[0], torch.Tensor):
                    batch_gt_boxes3d = val[0].new_zeros((batch_size, max_gt, val[0].shape[-1]), dtype=torch.float)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                ret[key] = batch_gt_boxes3d

            elif key in ['roi_boxes']:
                max_gt = max([x.shape[1] for x in val])
                batch_gt_boxes3d = np.zeros((batch_size, val[0].shape[0], max_gt, val[0].shape[-1]), dtype=np.float32)
                for k in range(batch_size):
                    batch_gt_boxes3d[k,:, :val[k].shape[1], :] = val[k]
                ret[key] = batch_gt_boxes3d

            elif key in ['roi_scores', 'roi_labels']:
                max_gt = max([x.shape[1] for x in val])
                batch_gt_boxes3d = np.zeros((batch_size, val[0].shape[0], max_gt), dtype=np.float32)
                for k in range(batch_size):
                    batch_gt_boxes3d[k,:, :val[k].shape[1]] = val[k]
                ret[key] = batch_gt_boxes3d

            elif key in ['gt_boxes2d']:
                max_boxes = 0
                max_boxes = max([len(x) for x in val])
                batch_boxes2d = np.zeros((batch_size, max_boxes, val[0].shape[-1]), dtype=np.float32)
                for k in range(batch_size):
                    if val[k].size > 0:
                        batch_boxes2d[k, :val[k].__len__(), :] = val[k]
                ret[key] = batch_boxes2d
            elif key in ["images", "depth_maps"]:
                # Get largest image size (H, W)
                max_h = 0
                max_w = 0
                for image in val:
                    max_h = max(max_h, image.shape[0])
                    max_w = max(max_w, image.shape[1])

                # Change size of images
                images = []
                for image in val:
                    pad_h = common_utils.get_pad_params(desired_size=max_h, cur_size=image.shape[0])
                    pad_w = common_utils.get_pad_params(desired_size=max_w, cur_size=image.shape[1])
                    pad_width = (pad_h, pad_w)
                    pad_value = 0

                    if key == "images":
                        pad_width = (pad_h, pad_w, (0, 0))
                    elif key == "depth_maps":
                        pad_width = (pad_h, pad_w)

                    image_pad = np.pad(image,
                                        pad_width=pad_width,
                                        mode='constant',
                                        constant_values=pad_value)

                    images.append(image_pad)
                ret[key] = np.stack(images, axis=0)
            elif key in ['calib']:
                ret[key] = val
            elif key in ["points_2d"]:
                max_len = max([len(_val) for _val in val])
                pad_value = 0
                points = []
                for _points in val:
                    pad_width = ((0, max_len-len(_points)), (0,0))
                    points_pad = np.pad(_points,
                            pad_width=pad_width,
                            mode='constant',
                            constant_values=pad_value)
                    points.append(points_pad)
                ret[key] = np.stack(points, axis=0)
            elif key in ['camera_imgs']:
                ret[key] = torch.stack([torch.stack(imgs,dim=0) for imgs in val],dim=0)
            else:
                ret[key] = np.stack(val, axis=0)
        except:
            print('Error in collate_batch: key=%s' % key)
            raise TypeError

    ret['batch_size'] = batch_size * batch_size_ratio
    return ret

def constant_reflectness(data_processor:DataProcessor, data_dict:dict=None, config:EasyDict=None):
    if data_dict is None:
        return partial(data_processor.constant_reflectness, config=config)
    
    def _check_tensor(input:Union[np.ndarray, torch.Tensor]):
        if isinstance(input, np.ndarray):
            input = (
                torch.from_numpy(input)
                .cuda()
            )
        
        return input
    
    data_dict["points"]:torch.Tensor = _check_tensor(data_dict["points"]) # type: ignore

    cosntant_value:float = config["constant"]

    reflectness = data_dict["points"].new_full(data_dict["points"].size(), cosntant_value)
    mask = data_dict["points"].new_zeros(data_dict["points"].size())
    mask[:, 3] = 1.0

    data_dict["points"] = data_dict["points"] * (1 - mask) + reflectness * mask 

    return data_dict
resgister_data_processor("constant_reflectness", constant_reflectness)

def sample_points_aux(data_processor:DataProcessor, data_dict:dict=None, config:EasyDict=None):
    if data_dict is None:
        return partial(sample_points_aux, data_processor=data_processor, config=config)
    
    def _check_tensor(input:Union[np.ndarray, torch.Tensor]):
        if isinstance(input, np.ndarray):
            input = (
                torch.from_numpy(input)
                .cuda()
            )
        
        return input
    
    num_points = config.NUM_POINTS[data_processor.mode]
    if num_points == -1:
        return data_dict
    
    data_dict["points"]:torch.Tensor = _check_tensor(data_dict["points"]) # type: ignore

    points = data_dict['points']
    if num_points < points.size(0):
        pts_depth = torch.linalg.norm(points[:, 0:3], dim=1)
        pts_near_flag = pts_depth < 40.0
        far_idxs_choice = torch.where(~pts_near_flag)[0]
        near_idxs = torch.where(pts_near_flag)[0]
        choice = []
        if num_points > far_idxs_choice.size(0):
            near_idxs_choice = torch.randperm(near_idxs.size(0))[:num_points - far_idxs_choice.size(0)]
            near_idxs_choice = near_idxs[near_idxs_choice]
            
            # near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
            choice = torch.concatenate((near_idxs_choice, far_idxs_choice), dim=0) \
                if far_idxs_choice.size(0) > 0 else near_idxs_choice
            choice = choice[torch.randperm(choice.size(0))]
        else: 
            # choice = np.arange(0, len(points), dtype=np.int32)
            # choice = np.random.choice(choice, num_points, replace=False)
            choice = torch.randperm(points.size(0))[:num_points]
        
    else:
        choice = torch.arange(0, points.size(0))
        if num_points > points.size(0):
            # extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
            extra_choice = torch.randperm(points.size(0))[:num_points - points.size(0)]
            choice = torch.concatenate((choice, extra_choice), dim=0)
        # np.random.shuffle(choice)
        choice = choice[torch.randperm(choice.size(0))]
        
    data_dict['points'] = points[choice]
    return data_dict
resgister_data_processor("sample_points_aux", sample_points_aux)