UNI_RANDOM_SEED = 2024
DEVICE = 1

import numpy as np
import torch
import torch.nn.functional as F

np.random.seed(UNI_RANDOM_SEED) 
torch.manual_seed(UNI_RANDOM_SEED)

torch.cuda.manual_seed(UNI_RANDOM_SEED)
torch.cuda.manual_seed_all(UNI_RANDOM_SEED)

torch.cuda.set_device(DEVICE)


import argparse

import open3d as o3d
import pickle as pkl


from tqdm import tqdm
from pathlib import Path
from typing import Optional
from data_tools import *
from vehicle_reconstruction import *
from vehicle_reconstruction.scene_segmentation import *


from pytorch3d.structures import Meshes

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils

from attack_utils import *

def draw_reconstructed_scene(pts:torch.Tensor, 
                             lidar:Optional[LiDAR_base] = None,
                             gt_boxes:Optional[torch.Tensor] = None, 
                             ref_boxes:Optional[torch.Tensor] = None, 
                             vehicles:Optional[list[vehicle_object]] = None,
                             adv_patch:Optional[adversarial_patch_3d] = None):
    from open3d_vis_utils import draw_scenes

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="view")
    meshes_batch = []
    pts_set = [pts]
    if vehicles is not None:
        for vehi in vehicles:
            if vehi is None:
                continue
            
            mesh, rooftop_idx = vehi.reconstruct_at_scene(
                vehi_reconstructor, vehicle_reconstructor.STANDARD_BBOX, show_rooftop=True)
            
            rooftop_center = mesh.vertices[rooftop_idx].mean(0)
            rooftop_center[2] = mesh.vertices[rooftop_idx][:, 2].max()
            
            axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.5, origin=rooftop_center)
            
            if lidar is None:
                vis.add_geometry(axis_pcd)
                
                mesh = o3d.geometry.TriangleMesh(
                    o3d.utility.Vector3dVector(mesh.vertices),
                    o3d.utility.Vector3iVector(mesh.faces))

                vertex_colors = np.ones_like(mesh.vertices) * np.array([0.5, 1.0, 0.5])
                vertex_colors[rooftop_idx] = np.array([1.0, 0.5, 0.5])
                vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
                mesh.vertex_colors = vertex_colors
                # mesh = mesh.filter_smooth_simple(number_of_iterations=5)
                mesh.compute_vertex_normals()

                vis.add_geometry(mesh)
                
            if adv_patch is not None:
                transformed_mesh:Meshes = adv_patch.get_transformed_meshes(torch.from_numpy(rooftop_center).float().cuda(),
                                                                            vehi.translation[3].view((1)))
                if lidar is not None:
                    meshes_batch.append(transformed_mesh)
                else:
                    mesh = o3d.geometry.TriangleMesh(
                        o3d.utility.Vector3dVector(transformed_mesh.verts_packed().detach().cpu().numpy()),
                        o3d.utility.Vector3iVector(transformed_mesh.faces_packed().detach().cpu().numpy()))

                    mesh.compute_vertex_normals()
                
                    vis.add_geometry(mesh)
    
    point_color = None
    if meshes_batch.__len__() != 0:
        meshes_batch = join_meshes_as_batch(meshes_batch)
        extend_pts = lidar.scan_triangles(meshes_batch)
        extend_pts = F.pad(extend_pts,  (0, 1), "constant", 0)
        point_color = torch.concatenate([torch.ones_like(pts_set[0])[:, :3].detach().cpu(), torch.ones_like(extend_pts)[:, :3].detach().cpu() * torch.tensor([1.0, 0.0, 0.0])])
        pts_set.append(extend_pts)
    
    pts_set = torch.concatenate(pts_set)
    ref_labels = None
    if ref_boxes is not None:
        ref_boxes, ref_labels = torch.split(ref_boxes, [7, 1], dim=1)
        ref_labels = ref_labels.view(-1).cpu().long().numpy()
        pts_assign = points_in_boxes_gpu(
        pts_set[:, :3].cuda().view(1, -1, 3), ref_boxes.cuda().view(1, -1, 7)).squeeze(dim=0).cpu()
        for bbox_idx in range(ref_boxes.size(0)):
            bbox_mask = (pts_assign == bbox_idx)
            point_color[bbox_mask] = torch.tensor([0.0, 0.0, 1.0])
        
    draw_scenes(vis, points=pts_set, gt_boxes=gt_boxes,
                point_colors = point_color,
                ref_labels=ref_labels,
                ref_boxes=ref_boxes,
            draw_origin=True)
    vis.run()
    vis.destroy_window()

def parse_config():
    args = argparse.ArgumentParser(description='vehicles reconstruction test')
    args.add_argument('--dataset-config-path', type=str, default="configs/dataset_configs/outdoor_demo_dataset.yaml")
    args.add_argument('--gtboxes-path', type=str, default=None)
    args.add_argument('--predicted-boxes-path', type=str, default=None)
    args.add_argument('--patch-ckpt', type=str, default=None)
    args.add_argument('--reconstructor-ckpt-path', type=str, default="data/reconstructor_ckpts/vehi_reconstructor_baidu_apollo.pt")
    args.add_argument('--output-path', type=str, default="./rooftop_appro.pkl")
    args.add_argument('--visualize', action='store_true')

    args = args.parse_args()
    cfg_from_yaml_file(args.dataset_config_path, cfg)

    return args, cfg

if __name__ == "__main__":
    
    args, cfg = parse_config()

    logger = common_utils.create_logger()
    logger.info('-----------------vehicles reconstruction test-------------------------')

    vehi_reconstructor = vehicle_reconstructor(vehicles=None,
                                            sampling_space=vehicle_reconstructor.STANDARD_BBOX,
                                            grid_res=vehicle_reconstructor.GRID_RES,
                                            global_res=vehicle_reconstructor.GLOBAL_RES)
    vehi_reconstructor.load_parameters(args.reconstructor_ckpt_path)

    rooftop_array = []
    
    dataset = outdoor_demo_dataset(cfg, 
                                class_names=['Car', 'Pedestrian', 'Cyclist'], 
                                training=False, 
                                ext=".bin", 
                                gtboxes_path = args.gtboxes_path,
                                logger=logger)
    predicted_boxes = None
    if args.predicted_boxes_path is not None:
        predicted_boxes = torch.load(args.predicted_boxes_path)
        
    adv_patches = None
    if args.patch_ckpt is not None:
        adv_patches_params = torch.load(args.patch_ckpt)["universal_adv_patch_car"]
        adv_patches = single_sphere()
        adv_patches.load_parameter(adv_patches_params)
        
    for index, batch_dict in tqdm(enumerate(dataset), total=dataset.__len__()):
        if args.visualize:
            index = 11
        #     index = np.random.randint(0, dataset.__len__())
        #     batch_dict = dataset.__getitem__(index)
            print(f"getting sample (index=: {index})")
        predicted_box = None
        if predicted_boxes is not None:
            predicted_box = predicted_boxes[index]
        load_data_to_gpu(batch_dict)

        print(batch_dict['gt_boxes'])
        pts = batch_dict['points']
        gt_boxes, gt_labels = torch.split(batch_dict['gt_boxes'], [7, 1], dim=1)  # [N, 7], [N, 1]
        gt_boxes = gt_boxes[gt_labels[:, 0] == 1]
        gt_labels = gt_labels[gt_labels[:, 0] == 1]
        

        scene = point_cloud_scene(pts=pts,
                            gt_boxes=gt_boxes,
                            vehi_reconstructor=vehi_reconstructor)
        vehicles: list[vehicle_object] = scene.get_vehicles()
        lidar = LiDAR_base(origin=torch.tensor([0.0, 0.0, 0.0]).cuda(),
                    azi_range=[-90, 90],
                    polar_range= [-24.8, 2.0],
                    polar_num=10, azi_res=0.08)


        scene.pose_estimate(iter=30)
        
        vehi_rooftop = []
        
        for vehi in vehicles:
            if vehi is None:
                vehi_rooftop.append(None)
                continue
            mesh, rooftop_idx = vehi.reconstruct_at_scene(
                vehi_reconstructor, vehicle_reconstructor.STANDARD_BBOX, show_rooftop=True)
            rooftop_center = mesh.vertices[rooftop_idx].mean(0)
            rooftop_center[2] = mesh.vertices[rooftop_idx][:, 2].max()
            rooftop_center = np.array(rooftop_center)
            vehi_rooftop.append(rooftop_center)
        
        rooftop_array.append(vehi_rooftop)
        if not args.visualize:
            with open(args.output_path, "wb") as output:
                pkl.dump(rooftop_array, output)

        if args.visualize:
            draw_reconstructed_scene(pts=pts,
                                     
                                    gt_boxes=gt_boxes,
                                    # ref_boxes=predicted_box,
                                    vehicles=vehicles,
                                    adv_patch=adv_patches)
            # draw_reconstructed_scene(pts=pts,
            #             lidar=lidar,
            #             # gt_boxes=gt_boxes,
            #             ref_boxes=predicted_box,
            #             vehicles=vehicles,
            #             adv_patch=adv_patches)
            
"""


{
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 48.698535919189453, 39.966701507568359, 13.827677726745605 ],
			"boundingbox_min" : [ -0.059999999999999998, -39.935523986816406, -2.0649197101593018 ],
			"field_of_view" : 60.0,
			"front" : [ -0.5593333727783002, -0.16299348072470099, 0.81276029881982903 ],
			"lookat" : [ 8.4395653629290468, 1.8159517486762229, -3.8732592605485952 ],
			"up" : [ 0.82858424796324448, -0.081097819330042661, 0.55395964449325763 ],
			"zoom" : 0.16
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}
"""