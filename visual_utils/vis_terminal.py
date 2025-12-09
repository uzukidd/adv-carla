import argparse
import open3d_vis_utils, visualize_utils
import torch
import numpy as np
import os

from typing import Union, Optional
from pathlib import Path

def load_point_cloud(path: Union[str, Path],
                     *,
                     bin_dtype: np.dtype = np.float32,
                     num_fields: Optional[int] = None) -> torch.Tensor:
    """
    Load point cloud from .pt or .bin and return a CPU torch.Tensor.

    - .pt: loaded with torch.load(..., map_location="cpu")
    - .bin: loaded with numpy.fromfile with dtype `bin_dtype`.
             The binary layout is assumed to be contiguous floats.
             If num_fields is provided, reshape to (-1, num_fields).
             If num_fields is None, this function will try to infer
             common layouts (4 then 3) and raise if ambiguous.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pt":
        # .pt file should contain a tensor or array-like object
        tensor: torch.Tensor = torch.load(path, map_location="cpu")
        # ensure it's a torch tensor on CPU
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
        return tensor.to("cpu")

    if suffix == ".bin":
        # read raw binary floats using numpy
        raw = np.fromfile(str(path), dtype=bin_dtype)

        if num_fields is not None:
            if raw.size % num_fields != 0:
                raise ValueError(f"File size not divisible by num_fields={num_fields}.")
            points = raw.reshape(-1, num_fields)
        else:
            # try common cases: 4 (e.g., x,y,z,intensity) then 3 (x,y,z)
            print(raw.shape)
            if raw.size % 5 == 0:
                points = raw.reshape(-1, 5)
            elif raw.size % 4 == 0:
                points = raw.reshape(-1, 4)
            elif raw.size % 3 == 0:
                points = raw.reshape(-1, 3)
            else:
                # fallback: return a 1D tensor but warn the user
                raise ValueError(
                    "Cannot infer point width from .bin size. "
                    "Please provide num_fields (e.g., 4 for XYZ+I or 3 for XYZ)."
                )

        # convert to torch tensor on CPU
        tensor = torch.from_numpy(points).to("cpu")
        return tensor

    raise ValueError(f"Unsupported file extension: {suffix}")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '-p', '--point_clouds',
        type=str,
        required=True,
    )

    
    args = parser.parse_args()
    point_clouds:torch.Tensor = load_point_cloud(args.point_clouds)

    gt_boxes: Optional[torch.Tensor] = None
    pc_path = args.point_clouds
    gt_boxes_path = pc_path + ".gt_boxes.bin"
    if os.path.exists(gt_boxes_path):
        gt_boxes = torch.from_numpy(np.fromfile(gt_boxes_path, dtype=np.float32).reshape(-1, 8))
        print(gt_boxes.size())
        if gt_boxes.numel() > 0:
            labels: torch.Tensor = gt_boxes[:, -1].to(torch.long)
            classes, counts = torch.unique(labels, return_counts=True)
            print("GT class counts:")
            for cls, cnt in zip(classes.tolist(), counts.tolist()):
                print(f"  class {cls}: {cnt}")
    
    
    batch_idx = 0
    open3d_vis_utils.draw_scenes(point_clouds[point_clouds[:, 0]==batch_idx][:, 1:], None if gt_boxes is None else gt_boxes)

if __name__ == "__main__":
    main()