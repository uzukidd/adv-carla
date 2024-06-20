import numpy as np
import torch
import voxel_ops
import pdb

from pcdet.datasets.processor.data_processor import DataProcessor

from functools import partial

class Data_preprocessor(DataProcessor):
    
    def __init__(self, advanced_processor_configs, point_cloud_range, training, num_point_features):
        super().__init__([], point_cloud_range, training, num_point_features)
        
        self.advanced_processor_configs = advanced_processor_configs
        self.differentiable_voxel_generator = None
        
        if advanced_processor_configs is not None:
            for cur_cfg in advanced_processor_configs:
                cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
                self.data_processor_queue.append(cur_processor)
                
                
    def remove_reflective(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.remove_reflective, config=config)
        
        points = data_dict['points'][:, :3]
        zeros = points.new_zeros((points.size(0), 1))
        data_dict['points'] = torch.cat((points, zeros), dim=1)
        return data_dict
    
    def sample_points_gpu(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points_gpu, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

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
    
    def differentiable_voxelize(self, data_dict=None, config=None):
        if data_dict is None:
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.differentiable_voxelize, config=config)
        
        if self.differentiable_voxel_generator is None:
            self.differentiable_voxel_generator = voxel_ops.Voxelization(
                voxel_size = self.voxel_size,
                # [x, y, z]
                point_cloud_range = self.point_cloud_range,
                # [x_min, y_min, z_min, x_max, y_max, z_max]
                max_num_points = config.MAX_POINTS_PER_VOXEL,
                # int
                max_voxels = config.MAX_NUMBER_OF_VOXELS[self.mode],
                # int
                deterministic = True,
                # boolean
            )

        points = data_dict['points']
        voxel_output = self.differentiable_voxel_generator(points)
        voxels, coordinates, num_points = voxel_output
        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)
        
        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates[:, [2, 1, 0]]
        data_dict['voxel_num_points'] = num_points
        
        return data_dict