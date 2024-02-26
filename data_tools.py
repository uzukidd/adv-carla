import numpy as np
import torch
import torch.nn.functional as F

import pdb
import glob

from pathlib import Path

from pytorch3d.ops import sample_points_from_meshes, laplacian
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.utils import ico_sphere
from pytorch3d.transforms import Scale
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader, DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

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

class kitti_carla_dataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, map_name="", logger=None, ext='.ply'):
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
        self.ext = ext
        self.frames_path = self.root_path / Path("generated/frames")
        self.frames = sorted(glob.glob(str(self.frames_path) + f"/frame*{self.ext}"))
        
        if self.logger is not None:
            self.logger.info('Total samples for KITTI-CARLA dataset: %d' % (len(self)))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        clean_data = self.__getitem_aux__(index)

        return clean_data
    
    def __getitem_aux__(self, index):
        if self.ext == '.ply':
            raw_data = read_ply(self.frames[index])
            points = np.stack([raw_data["x"], raw_data["y"], raw_data["z"], raw_data["cos_angle_lidar_surface"]], axis=1)
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        
        return data_dict
    
    
class adversarial_patch_3d:
    def __init__(self, basic_mesh):
        self.basic_mesh: Meshes = basic_mesh
        self.deform_vert: torch.Tensor = torch.zeros_like(basic_mesh.verts_packed(), requires_grad=True).cuda().contiguous()
        self.base_coord: torch.Tensor = self.get_base_coord()
        
    def get_basic_mesh(self):
        return self.basic_mesh
        
    def get_deformed_mesh(self):
        return self.basic_mesh.offset_verts(self.deform_vert)
    
    def sample_points(self, sample_amount=50):
        deformed_mesh = self.get_deformed_mesh()
        pts_sampled = sample_points_from_meshes(deformed_mesh, sample_amount)
        return pts_sampled
    
    def update_mesh(self, dst_vert):
        self.deform_vert = dst_vert.detach().clone()
        self.deform_vert.requires_grad_(True)

    def get_laplacian_loss(self):
        deformed_mesh = self.get_deformed_mesh()
        return mesh_laplacian_smoothing(deformed_mesh)

    def get_mesh_deform_vert(self):
        return self.deform_vert
    
    def get_mesh_gradient(self):
        return self.deform_vert.grad
    
    def clear_mesh_gradient(self):
        self.deform_vert.grad.zero_()
    
    def get_base_coord(self):
        base_z = self.basic_mesh.verts_packed()[:, 2].min()
        return self.deform_vert.new_tensor([0.0, 0.0, base_z], requires_grad=False)
    
class adv_dataset(DatasetTemplate):
    def __init__(self, parent_dataset, target_class = 1):
        """
        Args:
            parent_dataset:
        """
        super().__init__(
            dataset_cfg=parent_dataset.dataset_cfg, class_names=parent_dataset.class_names, training=parent_dataset.training, root_path=parent_dataset.root_path, logger=parent_dataset.logger
        )
        self.dataset = parent_dataset
        self.parent_dataset = parent_dataset
        self.evaluating = False
        self.enabled_adversarial_patch = True
        self.target_class = target_class
        
        if self.logger is not None:
            self.logger.info('Total samples for dataset: %d' % (len(self)))
            
        self.universal_adv_patch = adversarial_patch_3d(basic_mesh=self.generate_basic_mesh().cuda())
        

    def __len__(self):
        return len(self.parent_dataset)

    def __getitem__(self, index):
        batch_dict = self.parent_dataset.__getitem__(index)
        batch_dict = kitti_carla_dataset.collate_batch([batch_dict])
        load_data_to_gpu(batch_dict)
    
        
        if self.enabled_adversarial_patch:
            batch_dict=self.prepare_adversarial_data(batch_dict)
        
        return batch_dict
    
    def enable_adversarial_patch(self, enable):
        self.enabled_adversarial_patch = enable
    
    def prepare_adversarial_data(self, data_dict):
        pos_trans = None
        theta = None
        if data_dict.get('gt_boxes', None) is not None:
            
            gt_boxes_selected_idx = (data_dict['gt_boxes'][0, :, 7] == self.target_class)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][:, gt_boxes_selected_idx]
            
            pos_trans, theta, _ = torch.split(data_dict['gt_boxes'].squeeze(0), [6, 1, 1], dim=1)
            data_dict["points"] = self.attach_adv_patch_scene(data_dict["points"][:, 1:4], 
                                                         self.universal_adv_patch,
                                                         pos_trans, theta, None)
            data_dict["points"] = F.pad(data_dict["points"], (1, 1), "constant", 0)
        return data_dict
    
    @staticmethod
    def generate_basic_mesh(level:int = 0, scale:float = 0.5):
        mSphere = ico_sphere(level)
        # new_verts = mSphere.verts_padded() * scale
        mSphere = mSphere.update_padded(mSphere.verts_padded() * scale)
        return mSphere
    
    @staticmethod
    def attach_adv_patch_scene(points, adv_patch:adversarial_patch_3d, pos_trans, theta, deform_vert, sample_amount = 50):
        pts_set = [points]
        
        if pos_trans is None:
            return torch.concatenate(pts_set)
        
        n = pos_trans.__len__()
        
        for i in range(n):
            pts = adv_patch.sample_points(sample_amount=sample_amount).view(-1, 3) - adv_patch.base_coord
            extend_pts = pos_trans[i][None, :3] + adv_dataset.rotate_points(
                                                    pts, theta[i]) 
            extend_pts[:, 2] += pos_trans[i][None, 5] / 2.0
            
            # print(f"extend_pts:\t{extend_pts}")
            
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
    def generate_adv_sample(batch_dict):
        pass
    
if __name__ == "__main__":
    CFG_FILE = "./cfgs/kitti_models/pointrcnn.yaml"
    DATA_PATH = "/home/ksas/Public/datasets/KITTI-CARLA/dataset/Town01"
    cfg_from_yaml_file(CFG_FILE, cfg)
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of Kitti-carla-------------------------')
    dataset = kitti_carla_dataset(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(DATA_PATH), ext=".ply", logger=logger)
    
    
    BATCH_SIZE = 1
    WORKERS = 4
    DIST_TEST = False
    CFG_FILE = "./cfgs/kitti_models/pointrcnn.yaml"
    cfg_from_yaml_file(CFG_FILE, cfg)
    
    kitti_test_set, kitti_test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=BATCH_SIZE,
        dist=DIST_TEST, workers=WORKERS, logger=logger, training=False
    )
    logger.info(f'Class names of samples: \t{kitti_test_set.class_names}')

    test_adv_dataset = adv_dataset(kitti_test_set)
    data_dict = test_adv_dataset[0]
    logger.info(f"The keys of data_dict:\t{data_dict.keys()}") # [1, N, 3]
    logger.info(f"The size of sampled points:\t{test_adv_dataset.universal_adv_patch.sample_points(50).size()}") # [1, N, 3]
    logger.info(f"data_dict['points']:\t{data_dict['points'].size()}")
    logger.info(f"data_dict['gt_boxes']:\t{data_dict['gt_boxes'].size()}")
        
    V.draw_scenes(
        points=data_dict['points'][:, 1:], gt_boxes=data_dict['gt_boxes'][0]
    )
