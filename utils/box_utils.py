import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple, Union, Optional, Any
import torch


def get_box_corners(box: Union[np.ndarray, List[float]]) -> np.ndarray:
    """
    Computes 8 corners of an axis-aligned 3D bounding box.
    Box format: [center_x, center_y, center_z, size_x, size_y, size_z]
    Returns:
        corners: (8, 3) numpy array
    """
    cx, cy, cz, dx, dy, dz = box
    x_c = [cx - dx / 2.0, cx + dx / 2.0]
    y_c = [cy - dy / 2.0, cy + dy / 2.0]
    z_c = [cz - dz / 2.0, cz + dz / 2.0]
    return np.array([[x, y, z] for x in x_c for y in y_c for z in z_c])


def calculate_3d_iou(box1: Union[np.ndarray, List[float], torch.Tensor],
                     box2: Union[np.ndarray, List[float], torch.Tensor]) -> float:
    """
    Computes 3D Intersection over Union (IoU) for two axis-aligned 3D boxes.
    Box format: [cx, cy, cz, dx, dy, dz]
    """
    if isinstance(box1, torch.Tensor):
        box1 = box1.detach().cpu().numpy()
    if isinstance(box2, torch.Tensor):
        box2 = box2.detach().cpu().numpy()

    box1 = np.asarray(box1, dtype=np.float32)
    box2 = np.asarray(box2, dtype=np.float32)

    # Box 1 bounds
    min1 = box1[:3] - box1[3:6] / 2.0
    max1 = box1[:3] + box1[3:6] / 2.0
    vol1 = np.prod(np.maximum(box1[3:6], 0))

    # Box 2 bounds
    min2 = box2[:3] - box2[3:6] / 2.0
    max2 = box2[:3] + box2[3:6] / 2.0
    vol2 = np.prod(np.maximum(box2[3:6], 0))

    if vol1 <= 0 or vol2 <= 0:
        return 0.0

    # Intersection bounds
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    inter_dims = np.maximum(inter_max - inter_min, 0.0)
    inter_vol = np.prod(inter_dims)

    union_vol = vol1 + vol2 - inter_vol
    if union_vol <= 0:
        return 0.0

    return float(inter_vol / union_vol)


def batch_calculate_3d_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Computes pairwise 3D IoU for boxes1 of shape (N, 6) and boxes2 of shape (M, 6).
    Returns:
        iou_matrix: (N, M)
    """
    min1 = boxes1[:, None, :3] - boxes1[:, None, 3:6] / 2.0
    max1 = boxes1[:, None, :3] + boxes1[:, None, 3:6] / 2.0
    vol1 = np.prod(np.maximum(boxes1[:, 3:6], 0), axis=1)[:, None]

    min2 = boxes2[None, :, :3] - boxes2[None, :, 3:6] / 2.0
    max2 = boxes2[None, :, :3] + boxes2[None, :, 3:6] / 2.0
    vol2 = np.prod(np.maximum(boxes2[:, 3:6], 0), axis=1)[None, :]

    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    inter_dims = np.maximum(inter_max - inter_min, 0.0)
    inter_vol = np.prod(inter_dims, axis=-1)

    union_vol = vol1 + vol2 - inter_vol
    union_vol = np.maximum(union_vol, 1e-8)
    return inter_vol / union_vol


def make_bbox_lines(box: Union[np.ndarray, List[float]],
                    color: str = "red",
                    name: str = "Box",
                    width: int = 5) -> go.Scatter3d:
    """
    Creates a Plotly 3D wireframe line plot for a bounding box [cx, cy, cz, dx, dy, dz].
    """
    corners = get_box_corners(box)
    lines = [
        [0, 1], [1, 3], [3, 2], [2, 0],  # bottom square
        [4, 5], [5, 7], [7, 6], [6, 4],  # top square
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
    ]
    xl, yl, zl = [], [], []
    for p1, p2 in lines:
        xl += [corners[p1][0], corners[p2][0], None]
        yl += [corners[p1][1], corners[p2][1], None]
        zl += [corners[p1][2], corners[p2][2], None]

    return go.Scatter3d(
        x=xl,
        y=yl,
        z=zl,
        mode="lines",
        line=dict(color=color, width=width),
        name=name
    )


def make_bbox_mesh(box: Union[np.ndarray, List[float]],
                   color: str = "red",
                   name: str = "Box Volume",
                   opacity: float = 0.25) -> go.Mesh3d:
    """
    Creates a semi-transparent 3D solid box mesh for Plotly.
    """
    cx, cy, cz, dx, dy, dz = box
    x_min, x_max = cx - dx / 2.0, cx + dx / 2.0
    y_min, y_max = cy - dy / 2.0, cy + dy / 2.0
    z_min, z_max = cz - dz / 2.0, cz + dz / 2.0

    x = [x_min, x_max, x_max, x_min, x_min, x_max, x_max, x_min]
    y = [y_min, y_min, y_max, y_max, y_min, y_min, y_max, y_max]
    z = [z_min, z_min, z_min, z_min, z_max, z_max, z_max, z_max]

    # 12 triangles for 6 faces
    i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
    j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
    k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]

    return go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color=color,
        opacity=opacity,
        name=name,
        showlegend=False,
        hoverinfo="skip"
    )


def make_bbox_traces(box: Union[np.ndarray, List[float]],
                     color: str = "red",
                     name: str = "Box",
                     width: int = 7,
                     opacity: float = 0.3) -> List[Any]:
    """
    Returns both the 3D solid translucent mesh and thick wireframe outlines
    so the bounding box is prominently visible from all angles.
    """
    wireframe = make_bbox_lines(box, color=color, name=name, width=width)
    mesh = make_bbox_mesh(box, color=color, name=f"{name} Volume", opacity=opacity)
    return [mesh, wireframe]


