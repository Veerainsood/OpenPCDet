import math
import torch
# Added-------------------------------------------------------------------------------
import numpy as np
from . import common_utils
# Added-------------------------------------------------------------------------------
try:
    from kornia.geometry.conversions import (
        convert_points_to_homogeneous,
        convert_points_from_homogeneous,
    )
except:
    pass 
    # print('Warning: kornia is not installed. This package is only required by CaDDN')

# Added-------------------------------------------------------------------------------
def random_world_flip(params, reverse=False, points_3d=None, boxes_3d=None):
    if reverse:
        params = params[::-1]
    for cur_axis in params:
        if cur_axis == 'x':
            if points_3d is not None:
                points_3d[:, 1] = -points_3d[:, 1]
            if boxes_3d is not None:
                boxes_3d[:, 1] = -boxes_3d[:, 1]
                boxes_3d[:, 6] = -boxes_3d[:, 6]
        elif cur_axis == 'y':
            if points_3d is not None:
                points_3d[:, 0] = -points_3d[:, 0]
            if boxes_3d is not None:
                boxes_3d[:, 0] = -boxes_3d[:, 0]
                boxes_3d[:, 6] = -(boxes_3d[:, 6] + np.pi)
        else:
            raise NotImplementedError
    return points_3d, boxes_3d

def random_world_rotation(params, reverse=False, points_3d=None, boxes_3d=None):
    if reverse:
        params = -params
    if points_3d is not None:
        points_3d = common_utils.rotate_points_along_z(points_3d.unsqueeze(0), points_3d.new_tensor([params]))[0]
    if boxes_3d is not None:
        boxes_3d[:, 0:3] = common_utils.rotate_points_along_z(boxes_3d[:, 0:3].unsqueeze(0), boxes_3d.new_tensor([params]))[0]
        boxes_3d[:, 6] += params
    return points_3d, boxes_3d


def random_world_scaling(params, reverse=False, points_3d=None, boxes_3d=None):
    if reverse:
        params = 1.0 / params
    if points_3d is not None:
        points_3d[:, :3] *= params
    if boxes_3d is not None:
        boxes_3d[:, :6] *= params
    return points_3d, boxes_3d


def random_world_translation(params, reverse=False, points_3d=None, boxes_3d=None):
    if reverse:
        params = -params
    if points_3d is not None:
        points_3d[:, :3] += points_3d.new_tensor(params)
    if boxes_3d is not None:
        boxes_3d[:, :3] += boxes_3d.new_tensor(params)
    return points_3d, boxes_3d

def imrescale(params, reverse=False, points_2d=None, boxes_2d=None):
    w_scale, h_scale = params
    if reverse:
        w_scale, h_scale = 1.0 / w_scale, 1.0 / h_scale
    if points_2d is not None:
        points_2d[:, 0:2] *= points_2d.new_tensor([w_scale, h_scale])
    if boxes_2d is not None:
        boxes_2d[:, :4] *= boxes_2d.new_tensor([w_scale, h_scale, w_scale, h_scale])
    return points_2d, boxes_2d


def imflip(params, reverse=False, points_2d=None, boxes_2d=None):
    enable_x, rescale_w = params
    if enable_x:
        if points_2d is not None:
            points_2d[:, 0] = rescale_w - 1 - points_2d[..., 0]
        if boxes_2d is not None:
            flipped = boxes_2d.clone()
            flipped[:, 0] = rescale_w - 1 - boxes_2d[..., 2]
            flipped[:, 2] = rescale_w - 1 - boxes_2d[..., 0]
            boxes_2d = flipped
    return points_2d, boxes_2d


def points_lidar2img(points_3d, proj_mat, with_depth=False):
    """Project points from lidar coordicates to image coordinates.

    Args:
        points_3d (torch.Tensor): Points in shape (N, 3).
        proj_mat (torch.Tensor): (3, 4), transformation matrix between coordinates(left R).
        with_depth (bool, optional): Whether to keep depth in the output.
            Defaults to False.

    Returns:
        torch.Tensor: Points in image coordinates with shape [N, 2].
    """
    # (N, 4)
    points_4 = torch.cat([points_3d, points_3d.new_ones((points_3d.shape[0], 1))], dim=-1)
    point_2d = torch.matmul(points_4, proj_mat.t())
    point_2d_res = point_2d[..., :2] / torch.clamp(point_2d[..., 2:3], min=1e-5, max=1e5)

    if with_depth:
        return torch.cat([point_2d_res, point_2d[..., 2:3]], dim=-1)
    return point_2d_res
# Added-------------------------------------------------------------------------------

def project_to_image(project, points):
    """
    Project points to image
    Args:
        project [torch.tensor(..., 3, 4)]: Projection matrix
        points [torch.Tensor(..., 3)]: 3D points
    Returns:
        points_img [torch.Tensor(..., 2)]: Points in image
        points_depth [torch.Tensor(...)]: Depth of each point
    """
    # Reshape tensors to expected shape
    points = convert_points_to_homogeneous(points)
    points = points.unsqueeze(dim=-1)
    project = project.unsqueeze(dim=1)

    # Transform points to image and get depths
    points_t = project @ points
    points_t = points_t.squeeze(dim=-1)
    points_img = convert_points_from_homogeneous(points_t)
    points_depth = points_t[..., -1] - project[..., 2, 3]

    return points_img, points_depth


def normalize_coords(coords, shape):
    """
    Normalize coordinates of a grid between [-1, 1]
    Args:
        coords: (..., 3), Coordinates in grid
        shape: (3), Grid shape
    Returns:
        norm_coords: (.., 3), Normalized coordinates in grid
    """
    min_n = -1
    max_n = 1
    shape = torch.flip(shape, dims=[0])  # Reverse ordering of shape

    # Subtract 1 since pixel indexing from [0, shape - 1]
    norm_coords = coords / (shape - 1) * (max_n - min_n) + min_n
    return norm_coords


def bin_depths(depth_map, mode, depth_min, depth_max, num_bins, target=False):
    """
    Converts depth map into bin indices
    Args:
        depth_map: (H, W), Depth Map
        mode: string, Discretiziation mode (See https://arxiv.org/pdf/2005.13423.pdf for more details)
            UD: Uniform discretiziation
            LID: Linear increasing discretiziation
            SID: Spacing increasing discretiziation
        depth_min: float, Minimum depth value
        depth_max: float, Maximum depth value
        num_bins: int, Number of depth bins
        target: bool, Whether the depth bins indices will be used for a target tensor in loss comparison
    Returns:
        indices: (H, W), Depth bin indices
    """
    if mode == "UD":
        bin_size = (depth_max - depth_min) / num_bins
        indices = ((depth_map - depth_min) / bin_size)
    elif mode == "LID":
        bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
        indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
    elif mode == "SID":
        indices = num_bins * (torch.log(1 + depth_map) - math.log(1 + depth_min)) / \
            (math.log(1 + depth_max) - math.log(1 + depth_min))
    else:
        raise NotImplementedError

    if target:
        # Remove indicies outside of bounds
        mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
        indices[mask] = num_bins

        # Convert to integer
        indices = indices.type(torch.int64)
    return indices
