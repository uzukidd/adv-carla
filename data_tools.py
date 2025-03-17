import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import pickle as pkl
import pdb
import glob


from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from functools import partial
from typing import Optional

from raytorch.LiDAR import LiDAR_base

import pytorch3d
from pytorch3d.ops import sample_points_from_meshes, laplacian
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.utils import ico_sphere
from pytorch3d.transforms import Scale, Rotate, Translate, euler_angles_to_matrix
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene

from cudaext.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader, DatasetTemplate, KittiDataset
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


from preprocessing import Data_preprocessor
from pcdet.datasets.processor.data_processor import DataProcessor

from cudaext.ops.Rotated_IoU.oriented_iou_loss import cal_iou_3d, cal_iou
from cudaext.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
from attack_utils import *

ply_dtypes = dict([
    (b'char', 'i1'),
    (b'int8', 'i1'),
    (b'uchar', 'b1'),
    (b'uchar', 'u1'),
    (b'uint8', 'u1'),
    (b'short', 'i2'),
    (b'int16', 'i2'),
    (b'ushort', 'u2'),
    (b'uint16', 'u2'),
    (b'int', 'i4'),
    (b'int32', 'i4'),
    (b'uint', 'u4'),
    (b'uint32', 'u4'),
    (b'float', 'f4'),
    (b'float32', 'f4'),
    (b'double', 'f8'),
    (b'float64', 'f8')
])

# Numpy reader format
valid_formats = {'ascii': '', 'binary_big_endian': '>', 'binary_little_endian': '<'}

def register_data_processing(processing_name:str):
    def _real_decorator(processing_func):
        setattr(DataProcessor, 
                processing_name, 
                processing_func)
        
        return processing_func
    return _real_decorator



def parse_header(plyfile, ext):
    # Variables
    line = []
    properties = []
    num_points = None

    while b'end_header' not in line and line != b'':
        line = plyfile.readline()
    
        if b'element' in line:
            line = line.split()
            num_points = int(line[2])

        elif b'property' in line:
            line = line.split()
            properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))

    return num_points, properties

def read_ply(filename):
    """
    Read ".ply" files

    Parameters
    ----------
    filename : string
        the name of the file to read.

    Returns
    -------
    result : array
        data stored in the file

    Examples
    --------
    Store data in file

    >>> points = np.random.rand(5, 3)
    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example.ply', [points, values], ['x', 'y', 'z', 'values'])

    Read the file

    >>> data = read_ply('example.ply')
    >>> values = data['values']
    array([0, 0, 1, 1, 0])
    
    >>> points = np.vstack((data['x'], data['y'], data['z'])).T
    array([[ 0.466    0.595    0.324]
             [ 0.538    0.407    0.654]
             [ 0.850    0.018    0.988]
             [ 0.395    0.394    0.363]
             [ 0.873    0.996    0.092]])

    """

    with open(filename, 'rb') as plyfile:
        # Check if the file start with ply
        if b'ply' not in plyfile.readline():
            raise ValueError('The file does not start whith the word ply')

        # get binary_little/big or ascii
        fmt = plyfile.readline().split()[1].decode()
        if fmt == "ascii":
            raise ValueError('The file is not binary')

        # get extension for building the numpy dtypes
        ext = valid_formats[fmt]

        # Parse header
        num_points, properties = parse_header(plyfile, ext)

        # Get data
        data = np.fromfile(plyfile, dtype=properties, count=num_points)

    return data

@register_data_processing("test_preprocesssing")
def test_preprocesssing(self, data_dict=None, config=None):
    if data_dict is None:
        return partial(self.test_preprocesssing, config=config)
    print(data_dict)
    return data_dict


class kitti_carla_dataset(DatasetTemplate):
    def __init__(self, dataset_cfg, 
                 class_names, 
                 training=True, 
                 map_name="", 
                 logger=None, 
                 ext='.ply'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, logger=logger
        )
        self.ext = ext
        self.frames_path = Path(dataset_cfg.DATA_PATH) / Path("generated/frames")
        self.calib_path = Path(dataset_cfg.DATA_PATH) / Path("generated/lidar_to_cam0.txt")
        self.frames = sorted(glob.glob(str(self.frames_path) + f"/frame*{self.ext}"))
        
        self.gtboxes = None
        self.class_names = class_names

        if dataset_cfg.GTBOXES is not None:
            self.gtboxes = torch.load(dataset_cfg.GTBOXES)
        
        if self.logger is not None:
            self.logger.info('Total samples for KITTI-CARLA dataset: %d' % (len(self)))

        self.load_calib()
            
    def load_calib(self):
        with open(self.calib_path, 'r') as pos_file:
            line = pos_file.readlines()[-1].split()
            tf = np.vstack((np.array(line, float).reshape(3,4), [0,0,0,1]))
            self.tf_lidar_to_cam0 = tf

    def prepare_predicted_gtboxes(self, 
                                  surrogate_model:nn.Module,
                                  gt_boxes_path:str = None,
                                  pts_scene_path:str = None):
        self.gtboxes = []
        points_scene = []
        target_amount = 0
        with tqdm(self, 
                  desc="Inferencing...", 
                  postfix={"Target detected": 0},) as pbar:
            
            for batch_dict in pbar:
                pdb.set_trace()
                surrogate_model.eval()
                pred_dicts, _ = surrogate_model(batch_dict)
                
                preds_scores:torch.Tensor = pred_dicts[0]["pred_scores"] # [N, ]
                preds_boxes:torch.Tensor = pred_dicts[0]["pred_boxes"] # [N, 7]
                pred_labels:torch.Tensor = pred_dicts[0]["pred_labels"].view(-1) # [N, ]
                
                gtbox = torch.cat((preds_boxes, pred_labels), dim=1)
                
                self.gtboxes.append(gtbox)
                points_scene.append(pred_dicts[0]["points"])
                
                target_amount += gtbox.size(0)
                pbar.set_postfix({"Target detected": target_amount})
        
        self.logger.info(f"{target_amount} targets have been detected")
        
        if gt_boxes_path is not None:
            self.logger.info(f"Saving target as: {gt_boxes_path}")
            torch.save(self.gtboxes, gt_boxes_path)
            
        if pts_scene_path is not None:
            self.logger.info(f"Saving pointt scenes as: {pts_scene_path}")
            torch.save(points_scene, pts_scene_path)
            
    def evaluation(self, det_annos, class_names, **kwargs):
        assert self.gtboxes is not None
        recall_BEV = 0
        recall_3D = 0
        
        gt_boxes_count = 0
        for i, det_anno in enumerate(det_annos):
            gt_boxes = self.gtboxes[i]
            gt_boxes, gt_labels = torch.split(gt_boxes, [7, 1], dim=1)
            gt_boxes = gt_boxes[gt_labels.view(-1) == 1]
            
            pred_labels = torch.from_numpy(det_anno["pred_labels"]).to(gt_boxes.device)
            score = torch.from_numpy(det_anno["score"]).to(gt_boxes.device)
            boxes_lidar = torch.from_numpy(det_anno["boxes_lidar"]).to(gt_boxes.device)
            boxes_lidar = boxes_lidar[pred_labels == 1]
            
            M = gt_boxes.size(0)
            N = boxes_lidar.size(0)
            gt_boxes_count += M
            
            if M == 0 or N == 0:
                continue

            box3d_gt_extended = gt_boxes.view(M, 1, 7).expand(-1, N, -1)
            box3d_bl_extended = boxes_lidar.view(1, N, 7).expand(M, -1, -1)
            # iou3d [M, N]
            iou2d, _, _, _ = cal_iou(box3d_gt_extended[..., [0, 1, 3, 4, 6]], box3d_bl_extended[..., [0, 1, 3, 4, 6]])
            iou3d = cal_iou_3d(box3d_gt_extended, box3d_bl_extended)

            recall_BEV += (iou2d >= 0.7).sum().item()
            recall_3D += (iou3d >= 0.7).sum().item()
            
        
        recall_BEV = recall_BEV/gt_boxes_count
        recall_3D = recall_3D/gt_boxes_count
        
        result_str = f"""
        Car BEV@0.7\t{recall_BEV}
        Car 3D@0.7\t{recall_3D}
        """
        return result_str, {"recall_BEV":recall_BEV, "recall_3D":recall_3D}
    
    def prepare_kitti_det_annos(self, batch_index, box_dict):
        
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.tile(np.array([0., 0., 0., 50.]), (num_samples, 1)), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict
        
        pred_scores = box_dict['score']
        pred_boxes = box_dict['boxes_lidar']
        pred_labels = box_dict['pred_labels']
        pred_dict = get_template_prediction(pred_scores.shape[0])
        if pred_scores.shape[0] == 0:
            return pred_dict

        # calib = batch_dict['calib'][batch_index]
        # image_shape = batch_dict['image_shape'][batch_index].cpu().numpy()
        # pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
        # pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
        #     pred_boxes_camera, calib, image_shape=image_shape
        # )
        loc_camera = np.pad(pred_boxes[:, 0:3], ((0, 0), (0, 1)), constant_values=1).dot(self.tf_lidar_to_cam0.T)
        pred_dict['name'] = np.array(self.class_names)[pred_labels - 1]
        pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + 0.
        pred_dict['dimensions'] = pred_boxes[:, 3:6]
        pred_dict['location'] = loc_camera[:, 0:3]
        pred_dict['rotation_y'] = pred_boxes[:, 6]
        pred_dict['score'] = pred_scores
        pred_dict['boxes_lidar'] = pred_boxes
        
        return pred_dict
    
    def prepare_kitti_gt_annos(self, batch_index, gt_boxes, gt_labels):
        loc_camera = np.pad(gt_boxes[:, 0:3].cpu().numpy(), ((0, 0), (0, 1)), constant_values=1).dot(self.tf_lidar_to_cam0.T)
        
        annotations = {}
        annotations['name'] = np.array(self.class_names)[gt_labels.cpu().numpy().reshape(-1).astype(int) - 1]
        annotations['truncated'] = np.ones(gt_boxes.size(0)) * 0.
        annotations['occluded'] = np.ones(gt_boxes.size(0)) * 0.
        annotations['alpha'] = np.ones(gt_boxes.size(0)) * 0.
        annotations['bbox'] = np.tile(np.array([0., 0., 0., 50.]), (gt_boxes.size(0), 1))
        annotations['dimensions'] = gt_boxes[:, 3:6].cpu().numpy()
        annotations['location'] = loc_camera[:, 0:3]
        annotations['rotation_y'] = gt_boxes[:, 6].cpu().numpy()
        annotations['score'] = np.ones(gt_boxes.size(0)) * 0.
        annotations['difficulty'] = np.ones(gt_boxes.size(0)) * 1.

        annotations['index'] = np.array(batch_index, dtype=np.int32)
        annotations['gt_boxes_lidar'] = gt_boxes.cpu().numpy()
        
        return annotations

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        clean_data = self.__getitem_aux__(index)

        return clean_data
    
    def __getitem_aux__(self, index):
        if self.ext == '.ply':
            raw_data = read_ply(self.frames[index])
            points = np.stack([raw_data["x"], 
                               raw_data["y"], 
                               raw_data["z"], 
                               raw_data["cos_angle_lidar_surface"], 
                               raw_data['semantic'],
                               ], axis=1)
        else:
            raise NotImplementedError

        input_dict = {
            'points': points[:, :4],
            'frame_id': index,
            'points_with_semantic':  points,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        if self.gtboxes is not None:
            input_dict["gt_boxes"] = self.gtboxes[index].to(torch.device('cuda'))

        return data_dict
    
class outdoor_demo_dataset(DatasetTemplate):
    HEIGHT_OFFSET = [0.0, 0.0, -0.4, 0.0]
    def __init__(self, dataset_cfg, 
                 class_names, 
                 training=True, 
                 map_name="", 
                 gtboxes_path=None,
                 logger=None, 
                 ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, 
            class_names=class_names, 
            training=training, 
            logger=logger,
        )
        self.ext = ext
        self.frames_path = Path(dataset_cfg.DATA_PATH) / Path("point_cloud_outdoor")
        self.class_names = class_names
        self.frames = sorted(glob.glob(str(self.frames_path) + f"/*{self.ext}"))
        self.gtboxes = None
        
        if gtboxes_path is not None:
            self.gtboxes = torch.load(gtboxes_path)
        elif dataset_cfg.GTBOXES is not None:
            self.gtboxes = torch.load(dataset_cfg.GTBOXES)
        
        if self.logger is not None:
            self.logger.info('Total samples for OUTDOOR-DEMO dataset: %d' % (len(self)))

    def __getitem__(self, index):
        clean_data = self.__getitem_aux__(index)
        return clean_data
    
    def __len__(self):
        return self.frames.__len__()

    
    def __getitem_aux__(self, index):
        if self.ext == '.bin':
            raw_data = np.fromfile(self.frames[index], dtype=np.float32)
            points = deepcopy(raw_data).reshape(-1, 4)
            points += np.array(self.HEIGHT_OFFSET)
        else:
            raise NotImplementedError

        input_dict = {
            'points': points[:, :4],
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        if self.gtboxes is not None:
            data_dict["gt_boxes"] = self.gtboxes[index].to(torch.device('cuda'))
            
        return data_dict
    
    def prepare_kitti_det_annos(self, batch_index, box_dict):
            
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.tile(np.array([0., 0., 0., 50.]), (num_samples, 1)), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict
        
        pred_scores = box_dict['score']
        pred_boxes = box_dict['boxes_lidar']
        pred_labels = box_dict['pred_labels']
        pred_dict = get_template_prediction(pred_scores.shape[0])
        if pred_scores.shape[0] == 0:
            return pred_dict

        # calib = batch_dict['calib'][batch_index]
        # image_shape = batch_dict['image_shape'][batch_index].cpu().numpy()
        # pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
        # pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
        #     pred_boxes_camera, calib, image_shape=image_shape
        # )
        loc_camera = np.pad(pred_boxes[:, 0:3], ((0, 0), (0, 1)), constant_values=1)
        pred_dict['name'] = np.array(self.class_names)[pred_labels - 1]
        pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + 0.
        pred_dict['dimensions'] = pred_boxes[:, 3:6]
        pred_dict['location'] = loc_camera[:, 0:3]
        pred_dict['rotation_y'] = pred_boxes[:, 6]
        pred_dict['score'] = pred_scores
        pred_dict['boxes_lidar'] = pred_boxes
        
        return pred_dict
    
    def prepare_kitti_gt_annos(self, batch_index, gt_boxes, gt_labels):
        loc_camera = np.pad(gt_boxes[:, 0:3].cpu().numpy(), ((0, 0), (0, 1)), constant_values=1)
        
        annotations = {}
        annotations['name'] = np.array(self.class_names)[gt_labels.cpu().numpy().reshape(-1).astype(int) - 1]
        annotations['truncated'] = np.ones(gt_boxes.size(0)) * 0.
        annotations['occluded'] = np.ones(gt_boxes.size(0)) * 0.
        annotations['alpha'] = np.ones(gt_boxes.size(0)) * 0.
        annotations['bbox'] = np.tile(np.array([0., 0., 0., 50.]), (gt_boxes.size(0), 1))
        annotations['dimensions'] = gt_boxes[:, 3:6].cpu().numpy()
        annotations['location'] = loc_camera[:, 0:3]
        annotations['rotation_y'] = gt_boxes[:, 6].cpu().numpy()
        annotations['score'] = np.ones(gt_boxes.size(0)) * 0.
        annotations['difficulty'] = np.ones(gt_boxes.size(0)) * 1.

        annotations['index'] = np.array(batch_index, dtype=np.int32)
        annotations['gt_boxes_lidar'] = gt_boxes.cpu().numpy()
        
        return annotations

        
    
class adv_dataset(DatasetTemplate):
    
    CAR_ADV_PATCH_SCALE = [0.7, 0.7, 0.5]
    PED_ADV_PATCH_SCALE = [0.3, 0.3, 0.3]
    
    def __init__(self, parent_dataset, 
                 attack_config,
                 rooftop_annotate:Optional[str] = None,
                 surrogate_model = None, 
                 enable_car:bool = True,
                 enable_ped:bool = False,
                 enable_bicycle:bool = False,
                 enable_double:bool = False,
                 lidar:Optional[LiDAR_base] = None,
                 car_adv_patch_scale:Optional[list[float]] = None,
                 car_adv_patch_level:int = 2,
                 sample_amount = [50, 25],
                 target_class = 1,):
        """
        Args:
            parent_dataset:
        """
        super().__init__(
            dataset_cfg=parent_dataset.dataset_cfg, 
            class_names=parent_dataset.class_names, 
            training=parent_dataset.training, 
            root_path=parent_dataset.root_path, 
            logger=parent_dataset.logger
        )
        
        
        self.dataset = parent_dataset
        self.attack_config = attack_config
        self.parent_dataset = parent_dataset
        self.evaluating = False
        self.enabled_adversarial_patch = True
        self.lidar = lidar
        self.sample_amount = sample_amount
        self.gtboxes = None
        
        self.enable_car = enable_car
        self.enable_ped = enable_ped
        self.enable_bicycle = enable_bicycle
        self.enable_double = enable_double
        self.car_adv_patch_level = car_adv_patch_level
        
        self.target_class = target_class # 0 for background
        self.surrogate_model = surrogate_model
        
        if rooftop_annotate is not None:
            self.rooftop_approximate = self.load_annotated_rooftop(rooftop_annotate)
        else:
            self.rooftop_approximate = self.load_annotated_rooftop(self.attack_config.ROOFTOP_ANNOTATE)
        
        self.car_adv_patch_scale = car_adv_patch_scale
        if car_adv_patch_scale is None:
            self.car_adv_patch_scale = self.CAR_ADV_PATCH_SCALE
        assert self.car_adv_patch_scale.__len__() == 3
        
        
        if self.logger is not None:
            self.logger.info('Total samples for dataset: %d' % (len(self)))
            
        if self.rooftop_approximate is not None:
            rooftop_size = 0
            for i in self.rooftop_approximate:
                rooftop_size += i.__len__()
                
            self.logger.info('Successfully loaded rooftop appromximation: %d' % (rooftop_size))
        
        
        self.universal_adv_patch_car = single_sphere(scale=self.car_adv_patch_scale,
                                                     level=self.car_adv_patch_level)
        
        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.data_processor = Data_preprocessor(
            self.dataset_cfg.ADVANCED_DATA_PROCESSOR, point_cloud_range=self.point_cloud_range,
            training=self.training, num_point_features=4
        )
        
        self.detector_type = self.attack_config.DETECTOR
        self.trainging_components = self.attack_config.TRAINING_COMPONENTS
        self.target_component = self.attack_config.TARGET_COMPONENT

    def __len__(self):
        return self.parent_dataset.__len__()

    def load_annotated_rooftop(self, 
                               ROOFTOP_ANNOTATE):
        rooftop_approximate = None
        try:
            with open(ROOFTOP_ANNOTATE, "rb") as input:
                rooftop_approximate = pkl.load(input)
        except FileNotFoundError as error:
            self.logger.info(error.__str__())
        except TypeError as error:
            self.logger.info(error.__str__())
        
        return rooftop_approximate
    
    def set_adv_patches_training(self, enable:bool):
        self.universal_adv_patch_car.set_training(enable)
        # self.universal_adv_patch_ped.set_training(enable)
        
    def load_gtboxes(self,
                     path):
        self.gtboxes = torch.load(path)
    
    def prepare_predicted_gtboxes(self, 
                                  gt_boxes_path:str = None,
                                  predicted_score_path:str = None,
                                  pts_scenes_path:str = None,
                                  adversarial_enabled:bool = False):
        self.gtboxes = []
        self.predicted_score = []
        pts_scenes = []
        target_amount = 0
        assert self.surrogate_model is not None
        
        self.set_adv_patches_training(False)
        self.enable_adversarial_patch(adversarial_enabled)
        with tqdm(self, 
                  desc="Inferencing...", 
                  postfix={"Target detected": 0},) as pbar:
            
            for batch_dict in pbar:
                self.surrogate_model.eval()
                pred_dicts, _ = self.surrogate_model(batch_dict)
                
                preds_scores:torch.Tensor = pred_dicts[0]["pred_scores"] # [N, ]
                preds_boxes:torch.Tensor = pred_dicts[0]["pred_boxes"] # [N, 7]
                pred_labels:torch.Tensor = pred_dicts[0]["pred_labels"].view(-1) # [N, ]


                label_mask = (pred_labels == 1)
                pred_labels = pred_labels[label_mask].detach().view(-1, 1)
                preds_boxes = preds_boxes[label_mask].detach()
                preds_scores = preds_scores[label_mask].detach()
                
                
                if batch_dict.get('points_with_semantic') is not None and batch_dict.get('points_with_semantic')[0].shape[1] == 5:
                    points_with_semantic = batch_dict['points_with_semantic'][0]
                    points_assign = points_in_boxes_gpu(points_with_semantic[:, :3].unsqueeze(0), preds_boxes.unsqueeze(0))
                    points_assign = points_assign.squeeze(0)
                    
                    pred_boxes_mask = preds_boxes.new_zeros(preds_boxes.size(0)).bool()
                    for i in range(preds_boxes.size(0)):
                        points_idx_mask = (points_assign == i)
                        points_pred_box = points_with_semantic[points_idx_mask]
                        if points_pred_box.numel() == 0:
                            continue
                        
                        total_points_pred_box = points_pred_box.size(0)
                        confidence_pred_boxes = (points_pred_box == 10).sum()/total_points_pred_box
                        pred_boxes_mask[i] = (confidence_pred_boxes.item() > 0.7)
                    
                    pred_labels = pred_labels[pred_boxes_mask].detach()
                    preds_boxes = preds_boxes[pred_boxes_mask].detach()
                    preds_scores = preds_scores[pred_boxes_mask].detach()
                    
                gtbox = torch.cat((preds_boxes, pred_labels), dim=1)
                
                self.predicted_score.append(preds_scores)
                self.gtboxes.append(gtbox)
                pts_scenes.append(batch_dict["points"])
                
                target_amount += gtbox.size(0)
                pbar.set_postfix({"Target detected": target_amount})
        
        self.logger.info(f"{target_amount} targets have been detected")
        
        if predicted_score_path is not None:
            self.logger.info(f"Saving score as: {predicted_score_path}")
            torch.save(self.predicted_score, predicted_score_path)
        
        if gt_boxes_path is not None:
            self.logger.info(f"Saving target as: {gt_boxes_path}")
            torch.save(self.gtboxes, gt_boxes_path)
            
        if pts_scenes_path is not None:
            self.logger.info(f"Saving pointts scenes as: {pts_scenes_path}")
            torch.save(pts_scenes, pts_scenes_path)
            

    
    def load_adversarial_parameter(self, path:str):
        input_dict = torch.load(path)
        self.universal_adv_patch_car.load_parameter(input_dict["universal_adv_patch_car"])
        # self.universal_adv_patch_ped.load_parameter(input_dict["universal_adv_patch_ped"])
        
    def save_adversarial_parameter(self, path:str):
        output_dict = {
            "universal_adv_patch_car" : self.universal_adv_patch_car.get_parameters(),
            # "universal_adv_patch_ped" : self.universal_adv_patch_ped.get_parameters(),
        }
        
        torch.save(output_dict, path)
    
    def get_adversarial_parameter(self):
        return self.universal_adv_patch_car.get_parameters()

    def __getitem__(self, index,
                    adversarial_parameters:list[torch.Tensor] = None):
        batch_dict = self.parent_dataset.__getitem__(index)
        batch_dict["idx"] = index
        load_data_to_gpu(batch_dict)
        
        # pdb.set_trace()
        rooftop_approximate = None
        if self.rooftop_approximate is not None:
            rooftop_approximate = self.rooftop_approximate[index]
        if self.enabled_adversarial_patch:
            batch_dict = self.prepare_car_gtbox(batch_dict,
                rooftop_approximate)
            batch_dict = self.prepare_pedestrain_gtbox(batch_dict)
            batch_dict = self.prepare_bicycle_gtbox(batch_dict)
                    
            
            batch_dict=self.prepare_adversarial_data(batch_dict,
                                                     adversarial_parameters)
            
            # batch_dict = self.collect_all_class_gtbox(batch_dict)
        batch_dict = self.data_processor.forward(
            data_dict=batch_dict
        )
        load_data_to_gpu(batch_dict)
         
        batch_dict = self.gpu_collate_batch([batch_dict])
        
        
        return batch_dict
    
    def get_gt_boxes_statistic_info(self):
        res = np.array((0, 0, 0, 0))
        for batch_dict in self.parent_dataset:
            classes, counts = np.unique(batch_dict['gt_boxes'][:, 7], return_counts=True)
            for idx, count in zip(classes.astype(np.int64), counts):
                res[idx] += count
        
        return res
    
    def enable_adversarial_patch(self, enable):
        self.enabled_adversarial_patch = enable
        
    def prepare_adversarial_data(self, 
                                 batch_dict,
                                 adversarial_parameters = None,):
        gt_boxes_car = batch_dict.get("gt_boxes_car", None) #  [N, 8]
        gt_boxes_ped = batch_dict.get("gt_boxes_ped", None) #  [N, 8]
        gt_boxes_bicycle = batch_dict.get("gt_boxes_bicycle", None) #  [N, 8]
        
        if gt_boxes_car is not None and self.enable_car:
            _, theta, _ = torch.split(gt_boxes_car, [6, 1, 1], dim=1)
            
            batch_dict["points"] = self.attach_adv_patch_scene_car_aux(batch_dict["points"], 
                                                            self.universal_adv_patch_car,
                                                            theta, 
                                                            batch_dict["rooftop_approximate"],
                                                            self.sample_amount[0],
                                                            adversarial_parameters)

            
        # if gt_boxes_ped is not None and self.enable_ped:
        #     pos_trans, theta, _ = torch.split(gt_boxes_ped, [6, 1, 1], dim=1)
            
        #     batch_dict["points"] = self.attach_adv_patch_scene_ped_aux(batch_dict["points"], 
        #                                                     self.universal_adv_patch_ped,
        #                                                     pos_trans,
        #                                                     theta, 
        #                                                     self.sample_amount[1])

        return batch_dict
    
    def collect_all_class_gtbox(self, batch_dict):
        gt_boxes_car = batch_dict.get("gt_boxes_car", None) #  [N, 8]
        gt_boxes_ped = batch_dict.get("gt_boxes_ped", None) #  [N, 8]
        gt_boxes_bicycle = batch_dict.get("gt_boxes_bicycle", None) #  [N, 8]
        
        gt_boxes = []
        if gt_boxes_car is not None: 
            gt_boxes.append(gt_boxes_car)
            
        if gt_boxes_ped is not None: 
            gt_boxes.append(gt_boxes_ped)
            
        if gt_boxes_bicycle is not None: 
            gt_boxes.append(gt_boxes_bicycle)
            
        # batch_dict['gt_boxes'] = torch.concat(gt_boxes, 
        #                                           dim = 0) # [N1 + N2 + N3, 8]
        return batch_dict
    
    def prepare_bicycle_gtbox(self, batch_dict):
        """
            gt_boxes: [N, 8]
        """
        gt_boxes: torch.Tensor = batch_dict['gt_boxes']
        
        classes_mask = (gt_boxes[:, 7].view(-1) == 3)
        gt_boxes = gt_boxes[classes_mask, :]

        batch_dict['gt_boxes_bicycle'] = gt_boxes

        return batch_dict
    
    def prepare_pedestrain_gtbox(self, batch_dict):
        """
            gt_boxes: [N, 8]
        """
        gt_boxes: torch.Tensor = batch_dict['gt_boxes']
        
        classes_mask = (gt_boxes[:, 7].view(-1) == 2)
        gt_boxes = gt_boxes[classes_mask, :]

        batch_dict['gt_boxes_ped'] = gt_boxes

        return batch_dict
    
    def prepare_car_gtbox(self, batch_dict,
                      rooftop_approximate: list[np.ndarray]):
        """
            gt_boxes: [N, 8]
        """
        gt_boxes: torch.Tensor = batch_dict['gt_boxes']
        
        classes_mask = (gt_boxes[:, 7].view(-1) == 1)
        gt_boxes = gt_boxes[classes_mask, :]
        
        vaild_mask = [item is not None for item in rooftop_approximate]
        gt_boxes = gt_boxes[vaild_mask, :]
        if gt_boxes.size(0) != 0:
            temp = []
            for i in range(rooftop_approximate.__len__()):
                if rooftop_approximate[i] is not None:
                    temp.append(rooftop_approximate[i])
            rooftop_approximate = np.stack(temp)
        else:
            rooftop_approximate = None
            
        batch_dict['gt_boxes_car'] = gt_boxes
        batch_dict['rooftop_approximate'] = rooftop_approximate

        return batch_dict
    
    @staticmethod
    def generate_basic_mesh(level:int = 0,
                            scale:list=[0.7, 0.7, 0.5],
                            eps = 0.0):
        mSphere = ico_sphere(level)

        new_vert = mSphere.verts_padded()
        new_vert[:, :, 0] = new_vert[:, :, 0] * scale[0] * (1 - eps)
        new_vert[:, :, 1] = new_vert[:, :, 1] * scale[1] * (1 - eps)
        new_vert[:, :, 2] = new_vert[:, :, 2] * scale[2] * (1 - eps)

        mSphere = mSphere.update_padded(new_vert)
        return mSphere
    
    def attach_adv_patch_scene_car_aux(self, points, adv_patch:adversarial_patch_3d, 
                               theta, 
                               rooftop_approximate: np.ndarray = None,
                               sample_amount = 50,
                               adversarial_parameters = None,):
        pts_set = [points]
        meshes_batch = []
        if rooftop_approximate is not None:
            n = rooftop_approximate.shape[0]
            for i in range(n):
                extend_pts = None
                if self.lidar is not None:
                    transformed_mesh = adv_patch.get_transformed_meshes(torch.from_numpy(rooftop_approximate[i]).float().cuda(),
                                                                         theta[i],
                                                                         adversarial_parameters)
                    meshes_batch.append(transformed_mesh)
                    
                else:
                    pts = adv_patch.sample_points(sample_amount=sample_amount).view(-1, 3) - adv_patch.get_base_coord(False)
                    extend_pts = torch.from_numpy(rooftop_approximate[i])[None, :3].float().cuda() + adv_dataset.rotate_points(
                                                            pts, theta[i])
                    pts_set.append(extend_pts)
                    
            if meshes_batch.__len__() != 0:
                meshes_batch = join_meshes_as_batch(meshes_batch)
                extend_pts = self.lidar.scan_triangles(meshes_batch)
                extend_pts = F.pad(extend_pts,  (0, 1), "constant", 0)
                pts_set.append(extend_pts)
                
        return torch.concatenate(pts_set)
    
    def attach_adv_patch_scene_ped_aux(self, points, adv_patch:adversarial_patch_3d, 
                               pos_trans:torch.Tensor,
                               theta:torch.Tensor,  
                               sample_amount = 50):
        pts_set = [points]
        
        if pos_trans is None:
            return torch.concatenate(pts_set)
        
        n = pos_trans.__len__()
        
        for i in range(n):
            pts = adv_patch.sample_points(sample_amount=sample_amount).view(-1, 3) - adv_patch.base_coord
            extend_pts = pos_trans[i][None, :3] + adv_dataset.rotate_points(
                                                    pts, theta[i])
            extend_pts[:, 2] += pos_trans[i][None, 5] / 2.0
            
            pts_set.append(extend_pts)
        
       
        return torch.concatenate(pts_set)
    
    @staticmethod
    def rotate_points(points: torch.Tensor, angle: torch.Tensor):
        """
        Rotate a set of points around the origin by a given angle.

        Args:
            points (torch.Tensor): Tensor of shape (N, 3) representing N points in 3D space.
            angle (torch.Tensor): The angle of rotation in radians.

        Returns:
            torch.Tensor: Tensor of shape (N, 3) containing the rotated points.
        """
        cos_theta = torch.cos(angle)
        sin_theta = torch.sin(angle)

        rotation_matrix = points.new_tensor([
            [cos_theta, -sin_theta, 0.0],
            [sin_theta, cos_theta, 0.0],
            [0.0, 0.0, 1.0]
        ])

        rotated_points = torch.matmul(points, rotation_matrix.T)
        return rotated_points
    
    @staticmethod
    def gpu_collate_batch(batch_list, _unused=False):
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
                    ret[key] = torch.concat(val, dim=0)
                elif key in ['points', 'voxel_coords']:
                    coors = []
                    if isinstance(val[0], list):
                        val =  [i for item in val for i in item]
                    for i, coor in enumerate(val):
                        coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
                        coors.append(coor_pad)
                    ret[key] = torch.concat(coors, dim=0)
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = val[0].new_zeros((batch_size, max_gt, val[0].size(-1)))
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ['frame_id']:
                    ret[key] = val
                elif key in ['calib']:
                    ret[key] = val
                else:
                    ret[key] = val
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size * batch_size_ratio
        return ret
    

    
if __name__ == "__main__":
    # CFG_FILE = "configs/attack_configs/kitticarla/inference_pointrcnn.yaml"
    # DATA_PATH = "/home/ksas/Public/datasets/KITTI-CARLA/dataset/Town01"
    # cfg_from_yaml_file(CFG_FILE, cfg)
    # logger = common_utils.create_logger()
    # logger.info('-----------------Quick Demo of Kitti-carla-------------------------')
    # dataset = kitti_carla_dataset(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False, ext=".ply", logger=logger)
    
    # sample = dataset[0]
    # points = sample["points"]
    # points = points[points[:, 4] == 10]
    # print(points)
    
    CFG_FILE = "configs/dataset_configs/outdoor_demo_dataset.yaml"
    cfg_from_yaml_file(CFG_FILE, cfg)
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of Kitti-carla-------------------------')
    dataset = outdoor_demo_dataset(dataset_cfg=cfg, 
                                   class_names= ['Car', 'Pedestrian', 'Cyclist'], 
                                   training=False, 
                                   ext=".bin", 
                                   logger=logger)
    
    sample = dataset[0]
    points = sample['points']
    print(points)
    print(points.shape)

    # pdb.set_trace()
    
    # V.draw_scenes(
    #     points=points, ref_boxes=None,
    #     ref_scores=None, ref_labels=None, gt_boxes=None
    # )
    
    # BATCH_SIZE = 1
    # WORKERS = 4
    # DIST_TEST = False
    # CFG_FILE = "./cfgs/kitti_models/pointrcnn.yaml"
    # cfg_from_yaml_file(CFG_FILE, cfg)
    
    # kitti_test_set, kitti_test_loader, sampler = build_dataloader(
    #     dataset_cfg=cfg.DATA_CONFIG,
    #     class_names=cfg.CLASS_NAMES,
    #     batch_size=BATCH_SIZE,
    #     dist=DIST_TEST, workers=WORKERS, logger=logger, training=False
    # )
    # logger.info(f'Class names of samples: \t{kitti_test_set.class_names}')

    # test_adv_dataset = adv_dataset(kitti_test_set)
    # data_dict = test_adv_dataset[0]
    # logger.info(f"The keys of data_dict:\t{data_dict.keys()}") # [1, N, 3]
    # logger.info(f"The size of sampled points:\t{test_adv_dataset.universal_adv_patch.sample_points(50).size()}") # [1, N, 3]
    # logger.info(f"data_dict['points']:\t{data_dict['points'].size()}")
    # logger.info(f"data_dict['gt_boxes']:\t{data_dict['gt_boxes'].size()}")
        
    # V.draw_scenes(
    #     points=data_dict['points'][:, 1:], gt_boxes=data_dict['gt_boxes'][0]
    # )
