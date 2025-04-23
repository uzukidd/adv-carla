# fmt: off
import sys
from pathlib import Path

parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir) 

import argparse
import os

import torch
import torch.nn as nn
from pytorch3d.io import save_obj

from attack_utils import single_sphere

# fmt: on


def parse_arguments():
    args = argparse.ArgumentParser()

    args.add_argument(
        "-c",
        "--adversary-ckpt",
        default="output/train/PhysicalAdv/pointpillar/final_adversarial_patch_checkpoint.pt",
        help="adversary checkpoint filename",
    )

    args.add_argument(
        "-o",
        "--output-filename",
        default="assets/output_meshes/PhysicalAdv_PP.obj",
        help="meshes saving filename",
    )
    args = args.parse_args()

    return args


def load_adversarial_meshes(path: str):
    ckpt = torch.load(path)

    # legacy parameter format
    if isinstance(ckpt, dict) and "universal_adv_patch_car" in ckpt:
        return {
            "_T": nn.Parameter(ckpt["universal_adv_patch_car"][0]),
            "_theta": nn.Parameter(ckpt["universal_adv_patch_car"][1]),
            "deform_vert_logit": nn.Parameter(ckpt["universal_adv_patch_car"][2]),
        }

    return ckpt


def main(args):
    device = torch.device("cuda")
    adversarial_meshes = single_sphere(device=device)

    ckpt = load_adversarial_meshes(args.adversary_ckpt)
    adversarial_meshes.load_parameter(ckpt)

    adversarial_meshes_obj = adversarial_meshes.get_transformed_meshes().detach()

    # verts = adversarial_meshes_obj.verts_packed().cpu().numpy()
    # faces = adversarial_meshes_obj.faces_packed().cpu().numpy()

    # import trimesh

    # tri_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    # tri_mesh.show()

    save_obj(
        args.output_filename,
        adversarial_meshes_obj.verts_packed(),
        adversarial_meshes_obj.faces_packed(),
    )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
