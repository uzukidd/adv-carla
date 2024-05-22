import numpy as np
import torch
import torch.nn.functional as F

import pdb
import glob

from copy import deepcopy
from pathlib import Path

from raytorch.LiDAR import LiDAR_base

import pytorch3d
from pytorch3d.ops import sample_points_from_meshes, laplacian
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.utils import ico_sphere
from pytorch3d.transforms import Scale, Rotate, Translate, euler_angles_to_matrix
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene

from cudaext.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu

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

from loss_utils import extended_sigmoid, inverse_extended_sigmoid

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
    def __init__(self, 
                 basic_mesh, 
                 scale:list):
        self.basic_mesh: Meshes = basic_mesh
        self.deform_vert: torch.Tensor = torch.zeros_like(basic_mesh.verts_packed(), requires_grad=True).cuda().contiguous()
        self.deform_vert.requires_grad_(True)
    
        self.base_coord: torch.Tensor = self.get_base_coord()
        
        self.offset_limit: torch.Tensor = torch.tensor([0.1]).cuda()
        self.global_translation: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]).cuda()
        self.global_translation.requires_grad_(True)
        
        self.scale: torch.Tensor = torch.tensor(scale).cuda()
        self.theta: torch.Tensor = torch.tensor([0.0]).cuda()
        self.theta.requires_grad_(True)
        
        self.init_vert_quadrant: torch.Tensor = torch.sign(basic_mesh.verts_packed()).detach().clone()
        self.init_vert_logit: torch.Tensor = torch.logit(torch.abs(basic_mesh.verts_packed()/self.scale[None, :])).detach().clone()
        print(f"mesh vertex count : {self.basic_mesh.verts_packed().size()}")
        
        
    def generate_rotate_matrix(self, theta:torch.Tensor) -> torch.Tensor:
        tensor_0 = torch.zeros(1).cuda()
        RZ = euler_angles_to_matrix(torch.concatenate([tensor_0, tensor_0, theta]), ["X", "Y", "Z"])

        return RZ
        
    def get_basic_mesh(self):
        return self.basic_mesh
    
    def get_transformed_mesh(self, local_offset:torch.Tensor, pos:torch.Tensor,
                                theta:torch.Tensor):
        """
            Args:
                local_offset: [3]
                pos: [3]
                theta: [1]
        """
        deformed_mesh = self.get_deformed_mesh()
        verts = deformed_mesh.verts_padded()
        verts = verts + local_offset
        R = self.generate_rotate_matrix(theta)
        verts = torch.matmul(verts, R.T)
        verts = verts + (pos - self.get_base_coord(need_naive_rooftop_approxiamte=False))
        deformed_mesh = deformed_mesh.update_padded(verts)
        
        return deformed_mesh
        
        
    def get_deformed_mesh(self):
        offset = self.scale[None, :] \
                    * self.init_vert_quadrant \
                    * torch.sigmoid(self.init_vert_logit + self.deform_vert) \
                    + (self.offset_limit * torch.tanh(self.global_translation/self.offset_limit))[None, :]
        R = self.generate_rotate_matrix(self.theta)
        rotated_vert = torch.matmul(offset, R.T)
        return self.basic_mesh.update_padded(rotated_vert.unsqueeze(0))
    
    def get_deformed_mesh_aux(self):
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
        return self.deform_vert.grad, self.global_translation.grad, self.theta.grad
    
    def clear_mesh_gradient(self):
        self.deform_vert.grad.zero_()
        
    def load_parameter(self, input_dict:dict):
        self.deform_vert = input_dict["deform_vert"]
        self.theta = input_dict["theta"]
        self.global_translation = input_dict["global_translation"]
    
    def get_base_coord(self, need_naive_rooftop_approxiamte:bool=True):
        base_z = self.basic_mesh.verts_packed()[:, 2].min()
        if need_naive_rooftop_approxiamte:
            return self.deform_vert.new_tensor([0.2, 0.0, base_z], requires_grad=False)
        else:
            return self.deform_vert.new_tensor([0.0, 0.0, base_z], requires_grad=False)
        
class learnable_sphere:
    
    def __init__(self, level:int = 2,
                    scale:list=[0.35, 0.35, 0.25],
                    eps = 0.0):
        self.scale: torch.Tensor = torch.tensor(scale).float().cuda()
        self.basic_mesh = self.generate_basic_mesh(level=level, 
                                                   scale=scale, 
                                                   eps=eps)
        
        self.deform_vert_logit: torch.Tensor = torch.zeros_like(self.basic_mesh.verts_packed(), requires_grad=True).cuda().contiguous()
        self.deform_vert_logit.requires_grad_(True)
        
        self.init_vert_quadrant: torch.Tensor = torch.sign(self.basic_mesh.verts_packed()).detach().clone()
        self.init_vert_logit: torch.Tensor = torch.logit(torch.abs(self.basic_mesh.verts_packed() / 
                                                                   self.scale[None, :])).detach().clone()
        print(f"mesh vertex count : {self.basic_mesh.verts_packed().size()}")

    
    def get_parameters(self) -> list[torch.Tensor]:
        return [self.deform_vert_logit]
    
    def deformed_meshes(self) -> Meshes:
        deformed_vert = self.scale[None, :] \
            * self.init_vert_quadrant \
            * torch.sigmoid(self.init_vert_logit + self.deform_vert_logit)

        return self.basic_mesh.update_padded(deformed_vert.unsqueeze(0))
    
    def get_transformed_meshes(self, translate:torch.Tensor):
        deformed_meshes = self.deformed_meshes()
        verts = deformed_meshes.verts_padded()
        verts = verts + translate[None, :]
        
        return deformed_meshes.update_padded(verts)
    
    def get_base_coord(self):
        base_z = self.basic_mesh.verts_packed()[:, 2].min()
        
        return self.deform_vert.new_tensor([0.0, 0.0, base_z], requires_grad=False)
    
    @staticmethod
    def generate_basic_mesh(level:int,
                            scale:list,
                            eps):
        mSphere = ico_sphere(level).cuda()

        new_vert = mSphere.verts_padded()
        new_vert[:, :, 0] = new_vert[:, :, 0] * scale[0] * (1 - eps)
        new_vert[:, :, 1] = new_vert[:, :, 1] * scale[1] * (1 - eps)
        new_vert[:, :, 2] = new_vert[:, :, 2] * scale[2] * (1 - eps)

        mSphere = mSphere.update_padded(new_vert)
        return mSphere

class simple_cubic_meshes:
    
    def __init__(self, cubic_level:int = 2,
                    scale:list=[0.7, 0.7, 0.5],
                    eps = 0.0):
        self.internal_atom: list[learnable_sphere] = None
        self.scale = np.array(scale)
        self.cubic_level = cubic_level
        
        self.offset_limit: torch.Tensor = torch.tensor([0.1]).cuda()
        self.global_translation: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]).cuda()
        self.global_translation.requires_grad_(True)
        
        self.base_coord: torch.Tensor = torch.tensor([0.0, 0.0, -scale[2]]).float().cuda()
        
        self.theta: torch.Tensor = torch.tensor([0.0]).cuda()
        self.theta.requires_grad_(True)
        
        self._init_internal_atoms()
        self.lattice_grid = torch.stack([grid_pos.contiguous().view(-1) for grid_pos in self.generate_mesh_grid()], dim=1)
        
    def _init_internal_atoms(self):
        dscale = self.scale / self.cubic_level
        self.internal_atoms = []
        for _ in range(self.cubic_level ** 3):
            self.internal_atoms.append(learnable_sphere(level = 2,
                                                        scale = dscale))
            
    def get_laplacian_loss(self):
        deformed_mesh = self.get_deformed_lattice()
        return mesh_laplacian_smoothing(deformed_mesh)
            
    def get_parameters(self):
        parameters = [self.global_translation, self.theta]
        for internal_atom in self.internal_atoms:
            parameters.append(internal_atom.deform_vert_logit)
        
        return parameters
    
    def load_parameter(self, parameters:list):
        self.global_translation = parameters[0]
        self.theta = parameters[1]
        for i, internal_atom in enumerate(self.internal_atoms):
            internal_atom.deform_vert_logit = parameters[2 + i]
            
    def generate_mesh_grid(self):
        dscale = self.scale/self.cubic_level
        x = torch.arange(-self.scale[0] + dscale[0], 
                         self.scale[0], dscale[0] * 2.0).cuda()
        y = torch.arange(-self.scale[1] + dscale[1], 
                         self.scale[1], dscale[1] * 2.0).cuda()
        z = torch.arange(-self.scale[2] + dscale[2], 
                         self.scale[2], dscale[2] * 2.0).cuda()
        
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
        return grid_x, grid_y, grid_z

    def get_deformed_lattice(self) -> Meshes:
        atoms_meshes = []
        for i in range(self.lattice_grid.size(0)):
            atoms_meshes.append(self.internal_atoms[i].get_transformed_meshes(self.lattice_grid[i]))
        
        atoms_meshes = join_meshes_as_batch(atoms_meshes)
        return atoms_meshes
    
    def get_transformed_lattice(self, 
                                pos:torch.Tensor,
                                theta:torch.Tensor) -> Meshes:
        deformed_mesh = self.get_deformed_lattice()
        verts = deformed_mesh.verts_padded()
        verts = verts + (self.offset_limit * 
                         torch.tanh(self.global_translation / self.offset_limit))[None, :]
        global_R = self.generate_rotate_matrix(self.theta)
        verts = torch.matmul(verts, global_R.T)
        
        # translate to the rooftop of the vehicle
        local_R = self.generate_rotate_matrix(theta)
        verts = torch.matmul(verts, local_R.T)
        verts = verts + (pos - self.base_coord)
        
        transformed_mesh = deformed_mesh.update_padded(verts)
        
        return transformed_mesh
    
    def constrain_z_grad(self):
        self.global_translation.grad[2] = 0.
        for internal_atom in self.internal_atoms:
            internal_atom.deform_vert_logit.grad[:, 2] = 0.
        
    def generate_rotate_matrix(self, theta:torch.Tensor) -> torch.Tensor:
        tensor_0 = torch.zeros(1).cuda()
        RZ = euler_angles_to_matrix(torch.concatenate([tensor_0, tensor_0, theta]), ["X", "Y", "Z"])

        return RZ
    
class adv_dataset(DatasetTemplate):
    
    CAR_ADV_PATCH_SCALE = [0.7, 0.7, 0.5]
    PED_ADV_PATCH_SCALE = [0.3, 0.3, 0.3]
    
    def __init__(self, parent_dataset, 
                 sample_amount = [50, 25], 
                 surrogate_model = None, 
                 enable_car:bool = True,
                 enable_ped:bool = False,
                 enable_bicycle:bool = False,
                 enable_double:bool = False,
                 lidar:LiDAR_base = None,
                 rooftop_approximate: list[np.ndarray] = None,
                 target_class = 1,
                 cubic_level = 2,
                 car_basic_patch:Meshes = None,
                 ped_basic_patch:Meshes = None,):
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
        self.parent_dataset = parent_dataset
        self.evaluating = False
        self.enabled_adversarial_patch = True
        self.lidar = lidar
        
        self.enable_car = enable_car
        self.enable_ped = enable_ped
        self.enable_bicycle = enable_bicycle
        self.enable_double = enable_double
        
        self.target_class = target_class # 0 for background
        self.surrogate_model = surrogate_model
        self.sample_amount = sample_amount
        self.rooftop_approximate = rooftop_approximate
        
        if self.logger is not None:
            self.logger.info('Total samples for dataset: %d' % (len(self)))
            
        if self.rooftop_approximate is not None:
            rooftop_size = 0
            for i in self.rooftop_approximate:
                rooftop_size += i.__len__()
                
            self.logger.info('Successfully loaded rooftop appromximation: %d' % (rooftop_size))
        
        
        self.universal_adv_patch_car = simple_cubic_meshes(cubic_level=cubic_level)

        
        # if car_basic_patch is None:
        #     car_basic_patch = self.generate_basic_mesh(
        #     scale=self.CAR_ADV_PATCH_SCALE,
        #     level=2).cuda()
        # self.universal_adv_patch_car = adversarial_patch_3d(basic_mesh=car_basic_patch,
        #     scale=self.CAR_ADV_PATCH_SCALE)
        
        if ped_basic_patch is None:
            ped_basic_patch = self.generate_basic_mesh(
            scale=self.PED_ADV_PATCH_SCALE,
            level=2).cuda()
        self.universal_adv_patch_ped = adversarial_patch_3d(basic_mesh=ped_basic_patch,
            scale=self.PED_ADV_PATCH_SCALE)
        
        # self.logger.info(f"ground truth boxes statistic: {self.get_gt_boxes_statistic_info()}")

    def __len__(self):
        return self.parent_dataset.__len__()
    
    def load_adversarial_parameter(self, path:str):
        input_dict = torch.load(path)
        self.universal_adv_patch_car.load_parameter(input_dict["universal_adv_patch_car"])
        # self.universal_adv_patch_ped.load_parameter(input_dict["universal_adv_patch_ped"])
        
    def save_adversarial_parameter(self, path:str):
        output_dict = {
            "universal_adv_patch_car" : self.universal_adv_patch_car.get_parameters(),
        }
        
        torch.save(output_dict, path)
    
    def get_adversarial_parameter(self):
        return self.universal_adv_patch_car.get_parameters()

    def __getitem__(self, index):
        batch_dict = self.parent_dataset.__getitem__(index)
        batch_dict = kitti_carla_dataset.collate_batch([batch_dict])
        batch_dict["idx"] = index
        load_data_to_gpu(batch_dict)
        
        rooftop_approximate = None
        if self.rooftop_approximate is not None:
            rooftop_approximate = self.rooftop_approximate[index]
        
        batch_dict = self.prepare_car_gtbox(batch_dict,
            rooftop_approximate)
        
        batch_dict = self.prepare_pedestrain_gtbox(batch_dict)
        
        batch_dict = self.prepare_bicycle_gtbox(batch_dict)
        
        if self.enabled_adversarial_patch:
            batch_dict=self.prepare_adversarial_data(batch_dict)
            
        batch_dict = self.collect_all_class_gtbox(batch_dict)
        
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
        
    def prepare_adversarial_data(self, batch_dict):
        gt_boxes_car = batch_dict.get("gt_boxes_car", None) #  [1, N, 8]
        gt_boxes_ped = batch_dict.get("gt_boxes_ped", None) #  [1, N, 8]
        gt_boxes_bicycle = batch_dict.get("gt_boxes_bicycle", None) #  [1, N, 8]
        
        if gt_boxes_car is not None and self.enable_car:
            _, theta, _ = torch.split(gt_boxes_car.squeeze(0), [6, 1, 1], dim=1)
            
            batch_dict["points"] = self.attach_adv_patch_scene_car_aux(batch_dict["points"][:, 1:4], 
                                                            self.universal_adv_patch_car,
                                                            theta, 
                                                            batch_dict["rooftop_approximate"],
                                                            self.sample_amount[0])
            batch_dict["points"] = F.pad(batch_dict["points"], (1, 1), "constant", 0)
            
        if gt_boxes_ped is not None and self.enable_ped:
            pos_trans, theta, _ = torch.split(gt_boxes_ped.squeeze(0), [6, 1, 1], dim=1)
            
            batch_dict["points"] = self.attach_adv_patch_scene_ped_aux(batch_dict["points"][:, 1:4], 
                                                            self.universal_adv_patch_ped,
                                                            pos_trans,
                                                            theta, 
                                                            self.sample_amount[1])
            batch_dict["points"] = F.pad(batch_dict["points"], (1, 1), "constant", 0)
        return batch_dict
    
    def collect_all_class_gtbox(self, batch_dict):
        gt_boxes_car = batch_dict.get("gt_boxes_car", None) #  [1, N, 8]
        gt_boxes_ped = batch_dict.get("gt_boxes_ped", None) #  [1, N, 8]
        gt_boxes_bicycle = batch_dict.get("gt_boxes_bicycle", None) #  [1, N, 8]
        
        gt_boxes = []
        if gt_boxes_car is not None: 
            gt_boxes.append(gt_boxes_car)
            
        if gt_boxes_ped is not None: 
            gt_boxes.append(gt_boxes_ped)
            
        if gt_boxes_bicycle is not None: 
            gt_boxes.append(gt_boxes_bicycle)
            
        batch_dict['gt_boxes_car'] = torch.concat(gt_boxes, 
                                                  dim = 1) # [1, N1 + N2 + N3, 8]
        return batch_dict
    
    def prepare_bicycle_gtbox(self, batch_dict):
        """
            gt_boxes: [1, N, 8]
        """
        gt_boxes: torch.Tensor = batch_dict['gt_boxes']
        
        classes_mask = (gt_boxes[:, :, 7].view(-1) == 3)
        gt_boxes = gt_boxes[:, classes_mask]

        batch_dict['gt_boxes_bicycle'] = gt_boxes

        return batch_dict
    
    def prepare_pedestrain_gtbox(self, batch_dict):
        """
            gt_boxes: [1, N, 8]
        """
        gt_boxes: torch.Tensor = batch_dict['gt_boxes']
        
        classes_mask = (gt_boxes[:, :, 7].view(-1) == 2)
        gt_boxes = gt_boxes[:, classes_mask]

        batch_dict['gt_boxes_ped'] = gt_boxes

        return batch_dict
    
    def prepare_car_gtbox(self, batch_dict,
                      rooftop_approximate: list[np.ndarray]):
        """
            gt_boxes: [1, N, 8]
        """
        gt_boxes: torch.Tensor = batch_dict['gt_boxes']
        
        classes_mask = (gt_boxes[:, :, 7].view(-1) == 1)
        gt_boxes = gt_boxes[:, classes_mask]
        
        vaild_mask = [item is not None for item in rooftop_approximate]
        gt_boxes = gt_boxes[:, vaild_mask]
        if gt_boxes.size(1) != 0:
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
        # new_vert[:, :, 0] = new_vert[:, :, 0] - 0.2
        mSphere = mSphere.update_padded(new_vert)
        return mSphere
    

    def attach_adv_patch_scene_car_aux(self, points, adv_patch:adversarial_patch_3d, 
                               theta, 
                               rooftop_approximate: np.ndarray = None,
                               sample_amount = 50):
        pts_set = [points]
        meshes_batch = []
        if rooftop_approximate is not None:
            n = rooftop_approximate.shape[0]
            for i in range(n):
                extend_pts = None
                if self.lidar is not None:
                    # deformed_mesh = adv_patch.get_transformed_mesh(torch.tensor([10, 0.0, 0.0]).cuda(), theta[i])
                    if self.enable_double:
                        transformed_mesh = adv_patch.get_transformed_lattice(torch.from_numpy(rooftop_approximate[i]).float().cuda(),
                                                                         theta[i])

                        meshes_batch.append(transformed_mesh)
                    else:
                        deformed_mesh = adv_patch.get_transformed_mesh(
                            points.new_tensor([0.0, 0.0, 0.0], requires_grad=False),
                            torch.from_numpy(rooftop_approximate[i]).float().cuda(), 
                            theta[i])
                        meshes_batch.append(deformed_mesh)
                    
                else:
                    pts = adv_patch.sample_points(sample_amount=sample_amount).view(-1, 3) - adv_patch.get_base_coord(False)
                    extend_pts = torch.from_numpy(rooftop_approximate[i])[None, :3].float().cuda() + adv_dataset.rotate_points(
                                                            pts, theta[i])
                    pts_set.append(extend_pts)
            if meshes_batch.__len__() != 0:
                meshes_batch = join_meshes_as_batch(meshes_batch)
                extend_pts = self.lidar.scan_triangles(meshes_batch)
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
