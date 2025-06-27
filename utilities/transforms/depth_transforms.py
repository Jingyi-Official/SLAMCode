'''
Description: 
Author: 
Date: 2022-09-19 21:49:23
LastEditTime: 2023-03-01 16:38:36
LastEditors: Jingyi Wan
Reference: 
'''
import numpy as np
import torch
import torchvision.transforms as transforms
from utilities.transforms.grid_transforms import transform_grid_pts


class Depth_Transforms(object):
    def __init__(self, scale, max_depth):
        self.depth_transform = transforms.Compose(
            [DepthScale(scale),
             DepthFilter(max_depth)])

    def __call__(self, depth):
        return self.depth_transform(depth)

class DepthScale(object):
    """scale depth to meters"""

    def __init__(self, scale):
        self.scale = 1./scale

    def __call__(self, depth):
        depth = depth.astype(np.float32)
        return depth * self.scale

class DepthFilter(object):
    """scale depth to meters"""

    def __init__(self, max_depth):
        self.max_depth = max_depth

    def __call__(self, depth):
        far_mask = depth > self.max_depth
        depth[far_mask] = 0.
        return depth



'''
description: from depth (u,v) into pint cloud in camera coordinate pc(x,y,z)

depth: depth[0] seems the same thing, torch.Size([680, 1200])
fx: camera fx, intrinsic info of camera
fy: camera fy, intrinsic info of camera
cx: camera max depth, intrinsic info of camera ? refinement or what?
cy: camera min depth, intrinsic info of camera
depth_type: str = "z",
skip=1,

return: point cloud which correspond to the depth image
'''
def pointcloud_from_depth_torch(
    depth,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    T: float = None,
    depth_type: str = "z",
    skip=1,
) -> np.ndarray:
    assert depth_type in ["z", "euclidean"], "Unexpected depth_type"

    # set the grid according to depth size -----------------------
    rows, cols = depth.shape # '680, 1200'
    c, r = np.meshgrid(np.arange(cols, step=skip), np.arange(rows, step=skip), sparse=True)
    c = torch.from_numpy(c).to(depth.device) # torch.Size([1, 1200])
    r = torch.from_numpy(r).to(depth.device) # torch.Size([680, 1])


    # set the nan to FloatTensor in order to be processed and transform from depth image coordinates to camera coordinates-----------------------------------
    depth = depth[::skip, ::skip] # torch.Size([680, 1200])
    valid = ~torch.isnan(depth)
    nan_tensor = torch.FloatTensor([float('nan')]).to(depth.device)
    z = torch.where(valid, depth, nan_tensor) # torch.Size([680, 1200])
    x = torch.where(valid, z * (c - cx) / fx, nan_tensor) # torch.Size([680, 1200])
    y = torch.where(valid, z * (r - cy) / fy, nan_tensor) # torch.Size([680, 1200])
    pc = torch.dstack((x, y, z)) # torch.Size([680, 1200, 3])

    if depth_type == "euclidean":
        norm = torch.linalg.norm(pc, axis=2) # torch.Size([680, 1200])
        pc = pc * (z / norm)[:, :, None] # torch.Size([680, 1200, 3])


    if T is not None:
        shape = pc.shape
        pc_xyz_cc = pc.reshape(-1,3)
        pc_xyz_wc = transform_grid_pts(pc_xyz_cc[:,0],pc_xyz_cc[:,1],pc_xyz_cc[:,2],T)
        pc = torch.transpose(torch.stack(pc_xyz_wc), 0, 1)
        pc = pc.reshape(shape)

    return pc

