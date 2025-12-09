import os

import matplotlib.pyplot as plt
import mayavi.mlab as mlab
import numpy as np
import torch
from matplotlib.collections import LineCollection

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = (
        torch.stack((cosa, sina, zeros, -sina, cosa, zeros, zeros, zeros, ones), dim=1)
        .view(-1, 3, 3)
        .float()
    )
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = (
        boxes3d.new_tensor(
            (
                [1, 1, -1],
                [1, -1, -1],
                [-1, -1, -1],
                [-1, 1, -1],
                [1, 1, 1],
                [1, -1, 1],
                [-1, -1, 1],
                [-1, 1, 1],
            )
        )
        / 2
    )

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(
        -1, 8, 3
    )
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


def visualize_pts(
    pts,
    fig=None,
    bgcolor=(0, 0, 0),
    fgcolor=(1.0, 1.0, 1.0),
    show_intensity=False,
    size=(600, 600),
    draw_origin=True,
):
    if not isinstance(pts, np.ndarray):
        pts = pts.cpu().numpy()
    if fig is None:
        fig = mlab.figure(
            figure=None, bgcolor=bgcolor, fgcolor=fgcolor, engine=None, size=size
        )

    if show_intensity:
        G = mlab.points3d(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            pts[:, 3],
            mode="point",
            colormap="gnuplot",
            scale_factor=1,
            figure=fig,
        )
    else:
        G = mlab.points3d(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            mode="point",
            colormap="gnuplot",
            scale_factor=1,
            figure=fig,
        )
    if draw_origin:
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode="cube", scale_factor=0.2)
        mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), tube_radius=0.1)

    return fig


def draw_sphere_pts(
    pts, color=(0, 1, 0), fig=None, bgcolor=(0, 0, 0), scale_factor=0.2
):
    if not isinstance(pts, np.ndarray):
        pts = pts.cpu().numpy()

    if fig is None:
        fig = mlab.figure(
            figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(600, 600)
        )

    if isinstance(color, np.ndarray) and color.shape[0] == 1:
        color = color[0]
        color = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)

    if isinstance(color, np.ndarray):
        pts_color = np.zeros((pts.__len__(), 4), dtype=np.uint8)
        pts_color[:, 0:3] = color
        pts_color[:, 3] = 255
        G = mlab.points3d(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            np.arange(0, pts_color.__len__()),
            mode="sphere",
            scale_factor=scale_factor,
            figure=fig,
        )
        G.glyph.color_mode = "color_by_scalar"
        G.glyph.scale_mode = "scale_by_vector"
        G.module_manager.scalar_lut_manager.lut.table = pts_color
    else:
        mlab.points3d(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            mode="sphere",
            color=color,
            colormap="gnuplot",
            scale_factor=scale_factor,
            figure=fig,
        )

    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode="cube", scale_factor=0.2)
    mlab.plot3d(
        [0, 3],
        [0, 0],
        [0, 0],
        color=(0, 0, 1),
        line_width=3,
        tube_radius=None,
        figure=fig,
    )
    mlab.plot3d(
        [0, 0],
        [0, 3],
        [0, 0],
        color=(0, 1, 0),
        line_width=3,
        tube_radius=None,
        figure=fig,
    )
    mlab.plot3d(
        [0, 0],
        [0, 0],
        [0, 3],
        color=(1, 0, 0),
        line_width=3,
        tube_radius=None,
        figure=fig,
    )

    return fig


def draw_grid(x1, y1, x2, y2, fig, tube_radius=None, color=(0.5, 0.5, 0.5)):
    mlab.plot3d(
        [x1, x1],
        [y1, y2],
        [0, 0],
        color=color,
        tube_radius=tube_radius,
        line_width=1,
        figure=fig,
    )
    mlab.plot3d(
        [x2, x2],
        [y1, y2],
        [0, 0],
        color=color,
        tube_radius=tube_radius,
        line_width=1,
        figure=fig,
    )
    mlab.plot3d(
        [x1, x2],
        [y1, y1],
        [0, 0],
        color=color,
        tube_radius=tube_radius,
        line_width=1,
        figure=fig,
    )
    mlab.plot3d(
        [x1, x2],
        [y2, y2],
        [0, 0],
        color=color,
        tube_radius=tube_radius,
        line_width=1,
        figure=fig,
    )
    return fig


def draw_multi_grid_range(fig, grid_size=20, bv_range=(-60, -60, 60, 60)):
    for x in range(bv_range[0], bv_range[2], grid_size):
        for y in range(bv_range[1], bv_range[3], grid_size):
            fig = draw_grid(x, y, x + grid_size, y + grid_size, fig)

    return fig


def draw_scenes(
    points, gt_boxes=None, ref_boxes=None, ref_scores=None, ref_labels=None
):
    if not isinstance(points, np.ndarray):
        points = points.detach().cpu().numpy()
    if ref_boxes is not None and not isinstance(ref_boxes, np.ndarray):
        ref_boxes = ref_boxes.detach().cpu().numpy()
    if gt_boxes is not None and not isinstance(gt_boxes, np.ndarray):
        gt_boxes = gt_boxes.detach().cpu().numpy()
    if ref_scores is not None and not isinstance(ref_scores, np.ndarray):
        ref_scores = ref_scores.detach().cpu().numpy()
    if ref_labels is not None and not isinstance(ref_labels, np.ndarray):
        ref_labels = ref_labels.detach().cpu().numpy()

    fig = visualize_pts(points)
    fig = draw_multi_grid_range(fig, bv_range=(0, -40, 80, 40))
    if gt_boxes is not None:
        corners3d = boxes_to_corners_3d(gt_boxes)
        fig = draw_corners3d(corners3d, fig=fig, color=(0, 0, 1), max_num=100)

    if ref_boxes is not None and len(ref_boxes) > 0:
        ref_corners3d = boxes_to_corners_3d(ref_boxes)
        if ref_labels is None:
            fig = draw_corners3d(
                ref_corners3d, fig=fig, color=(0, 1, 0), cls=ref_scores, max_num=100
            )
        else:
            for k in range(ref_labels.min(), ref_labels.max() + 1):
                cur_color = tuple(box_colormap[k % len(box_colormap)])
                mask = ref_labels == k
                fig = draw_corners3d(
                    ref_corners3d[mask],
                    fig=fig,
                    color=cur_color,
                    cls=ref_scores[mask],
                    max_num=100,
                )
    mlab.view(azimuth=-179, elevation=54.0, distance=104.0, roll=90.0)
    return fig


def draw_corners3d(
    corners3d,
    fig,
    color=(1, 1, 1),
    line_width=2,
    cls=None,
    tag="",
    max_num=500,
    tube_radius=None,
):
    """
    :param corners3d: (N, 8, 3)
    :param fig:
    :param color:
    :param line_width:
    :param cls:
    :param tag:
    :param max_num:
    :return:
    """
    import mayavi.mlab as mlab

    num = min(max_num, len(corners3d))
    for n in range(num):
        b = corners3d[n]  # (8, 3)

        if cls is not None:
            if isinstance(cls, np.ndarray):
                mlab.text3d(
                    b[6, 0],
                    b[6, 1],
                    b[6, 2],
                    "%.2f" % cls[n],
                    scale=(0.3, 0.3, 0.3),
                    color=color,
                    figure=fig,
                )
            else:
                mlab.text3d(
                    b[6, 0],
                    b[6, 1],
                    b[6, 2],
                    "%s" % cls[n],
                    scale=(0.3, 0.3, 0.3),
                    color=color,
                    figure=fig,
                )

        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            mlab.plot3d(
                [b[i, 0], b[j, 0]],
                [b[i, 1], b[j, 1]],
                [b[i, 2], b[j, 2]],
                color=color,
                tube_radius=tube_radius,
                line_width=line_width,
                figure=fig,
            )

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d(
                [b[i, 0], b[j, 0]],
                [b[i, 1], b[j, 1]],
                [b[i, 2], b[j, 2]],
                color=color,
                tube_radius=tube_radius,
                line_width=line_width,
                figure=fig,
            )

            i, j = k, k + 4
            mlab.plot3d(
                [b[i, 0], b[j, 0]],
                [b[i, 1], b[j, 1]],
                [b[i, 2], b[j, 2]],
                color=color,
                tube_radius=tube_radius,
                line_width=line_width,
                figure=fig,
            )

        i, j = 0, 5
        mlab.plot3d(
            [b[i, 0], b[j, 0]],
            [b[i, 1], b[j, 1]],
            [b[i, 2], b[j, 2]],
            color=color,
            tube_radius=tube_radius,
            line_width=line_width,
            figure=fig,
        )
        i, j = 1, 4
        mlab.plot3d(
            [b[i, 0], b[j, 0]],
            [b[i, 1], b[j, 1]],
            [b[i, 2], b[j, 2]],
            color=color,
            tube_radius=tube_radius,
            line_width=line_width,
            figure=fig,
        )

    return fig


def save_adversarial_mesh_views(
    adversarial_patch,
    output_dir: str,
    file_prefix: str,
    dpi: int = 300,
):
    """
    Save three orthogonal projections (top/side/front) comparing base vs transformed mesh.

    Parameters
    ----------
    adversarial_patch : object
        Patch instance providing get_transformed_meshes(), get_basic_meshes(), and optional base_tranlate.
    output_dir : str
        Directory to place the rendered image (will be created).
    file_prefix : str
        Filename prefix; the function appends `_three_views.png`.
    dpi : int, optional
        Matplotlib DPI for the saved image.

    Returns
    -------
    str | None
        Full path of the saved image, or None if rendering skipped due to empty meshes.
    """
    if adversarial_patch is None:
        return None

    transformed_mesh = adversarial_patch.get_transformed_meshes()
    base_mesh = adversarial_patch.get_basic_meshes()

    verts = transformed_mesh.verts_packed().detach().cpu().numpy()
    faces = transformed_mesh.faces_packed().detach().cpu().numpy()

    base_verts_tensor = base_mesh.verts_padded().detach()
    base_translate = getattr(adversarial_patch, "base_tranlate", None)
    if base_translate is not None:
        base_verts_tensor = base_translate.transform_points(base_verts_tensor)
    base_verts = base_verts_tensor.squeeze(0).cpu().numpy()
    base_faces = base_mesh.faces_packed().detach().cpu().numpy()

    if (
        verts.size == 0
        or faces.size == 0
        or base_verts.size == 0
        or base_faces.size == 0
    ):
        return None

    def build_edge_pairs(face_array: np.ndarray):
        edges = set()
        for face in face_array:
            i, j, k = face.tolist()
            edges.add(tuple(sorted((i, j))))
            edges.add(tuple(sorted((j, k))))
            edges.add(tuple(sorted((k, i))))
        return list(edges)

    os.makedirs(output_dir, exist_ok=True)
    axis_labels = ["X", "Y", "Z"]
    view_configs = [
        ((0, 1), "Top (XY)"),
        ((0, 2), "Side (XZ)"),
        ((1, 2), "Front (YZ)"),
    ]

    edge_pairs = build_edge_pairs(faces)
    base_edge_pairs = build_edge_pairs(base_faces)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, ((axis_a, axis_b), title) in zip(axes, view_configs):
        ax.scatter(
            base_verts[:, axis_a],
            base_verts[:, axis_b],
            s=0.8,
            alpha=0.6,
            color="tab:red",
        )
        base_segments = [
            [
                (base_verts[i, axis_a], base_verts[i, axis_b]),
                (base_verts[j, axis_a], base_verts[j, axis_b]),
            ]
            for i, j in base_edge_pairs
        ]
        if base_segments:
            base_lines = LineCollection(
                base_segments, colors="tab:red", linewidths=0.35, alpha=0.7
            )
            ax.add_collection(base_lines)

        ax.scatter(verts[:, axis_a], verts[:, axis_b], s=1.2, alpha=0.85)
        segments = [
            [
                (verts[i, axis_a], verts[i, axis_b]),
                (verts[j, axis_a], verts[j, axis_b]),
            ]
            for i, j in edge_pairs
        ]
        if segments:
            line_collection = LineCollection(
                segments, colors="tab:blue", linewidths=0.45, alpha=0.9
            )
            ax.add_collection(line_collection)

        ax.set_xlabel(axis_labels[axis_a])
        ax.set_ylabel(axis_labels[axis_b])
        ax.set_title(title)
        ax.axis("equal")
        ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.4)

    fig.tight_layout()
    view_path = os.path.join(output_dir, f"{file_prefix}_three_views.png")
    fig.savefig(view_path, dpi=dpi)
    plt.close(fig)
    return view_path
