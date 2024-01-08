import numpy as np
import torch
import torch.nn.functional as F

import pdb
import glob

from pathlib import Path

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
    
if __name__ == "__main__":
    pass
    CFG_FILE = "./cfgs/kitti_models/pointrcnn.yaml"
    DATA_PATH = "/home/ksas/Public/datasets/KITTI-CARLA/dataset/Town01"
    cfg_from_yaml_file(CFG_FILE, cfg)
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of Kitti-carla-------------------------')
    dataset = kitti_carla_dataset(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(DATA_PATH), ext=".ply", logger=logger)
    
    data_0 = dataset[0]
    
    V.draw_scenes(
        points=data_0["points"]
    )
