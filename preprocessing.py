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