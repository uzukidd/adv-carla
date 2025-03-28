import argparse
import open3d_vis_utils, visualize_utils
import torch

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 添加必需参数
    parser.add_argument(
        '-p', '--point_clouds',
        type=str,
        required=True,
    )
    
    parser.add_argument(
        '-g', '--gt_boxes',
        type=str,
        required=True,
    )
    
    # 解析参数
    args = parser.parse_args()
    point_clouds:torch.Tensor = torch.load(args.point_clouds)
    gt_boxes:torch.Tensor = torch.load(args.gt_boxes)
    print(point_clouds.size())
    print(gt_boxes.size())
    batch_idx = 2
    open3d_vis_utils.draw_scenes(point_clouds[point_clouds[:, 0]==batch_idx][:, 1:], gt_boxes[batch_idx])

if __name__ == "__main__":
    main()