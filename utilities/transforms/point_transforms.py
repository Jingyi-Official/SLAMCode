'''
Description: 
Author: 
Date: 2022-09-19 21:49:23
LastEditTime: 2023-03-02 13:49:18
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Reference: 
'''
import torch
import numpy as np

'''
description: 
return {*}
reference: adapted from https://github.com/wkentaro/morefusion/blob/main/morefusion/geometry/estimate_pointcloud_normals.py
'''
def estimate_pointcloud_normals(points):
    # These lookups denote yx offsets from the anchor point for 8 surrounding
    # directions from the anchor A depicted below.
    #  -----------
    # | 7 | 6 | 5 |
    #  -----------
    # | 0 | A | 4 |
    #  -----------
    # | 1 | 2 | 3 |
    #  -----------
    assert points.shape[2] == 3

    d = 2
    H, W = points.shape[:2]

    # expand the dim of points for the later sliding operation
    points = torch.nn.functional.pad(
        points,
        pad=(0, 0, d, d, d, d),
        mode="constant",
        value=float('nan'),
    ) # torch.Size([684, 1204, 3])

    lookups = torch.tensor(
        [(-d, 0), (-d, d), (0, d), (d, d), (d, 0), (d, -d), (0, -d), (-d, -d)]
    ).to(points.device)

    # index for each point
    j, i = torch.meshgrid(torch.arange(W), torch.arange(H))
    i = i.transpose(0, 1).to(points.device) # torch.Size([680, 1200])
    j = j.transpose(0, 1).to(points.device) # torch.Size([680, 1200])
    k = torch.arange(8).to(points.device) # tensor([0, 1, 2, 3, 4, 5, 6, 7], device='cuda:0')

    # transform the original point cloud coordinates into the padded point cloud coordinates
    i1 = i + d # torch.Size([680, 1200])
    j1 = j + d # torch.Size([680, 1200])
    points1 = points[i1, j1] # torch.Size([680, 1200, 3])

    lookup = lookups[k]
    i2 = i1[None, :, :] + lookup[:, 0, None, None] # torch.Size([1, 680, 1200]) + torch.Size([8, 1, 1]) = torch.Size([8, 680, 1200])
    j2 = j1[None, :, :] + lookup[:, 1, None, None] # torch.Size([1, 680, 1200]) + torch.Size([8, 1, 1]) = torch.Size([8, 680, 1200])
    points2 = points[i2, j2] # torch.Size([8, 680, 1200, 3])

    lookup = lookups[(k + 2) % 8]
    i3 = i1[None, :, :] + lookup[:, 0, None, None] # torch.Size([8, 680, 1200])
    j3 = j1[None, :, :] + lookup[:, 1, None, None]
    points3 = points[i3, j3] # torch.Size([8, 680, 1200, 3])

    diff = torch.linalg.norm(points2 - points1, dim=3) + torch.linalg.norm(points3 - points1, dim=3)
    diff[torch.isnan(diff)] = float('inf')
    indices = torch.argmin(diff, dim=0)

    normals = torch.cross(
        points2[indices, i, j] - points1[i, j],
        points3[indices, i, j] - points1[i, j],
    )
    normals /= torch.linalg.norm(normals, dim=2, keepdims=True)
    return normals


def get_grid_pts(dims, transform):
    x = np.arange(dims[0])
    y = np.arange(dims[1])
    z = np.arange(dims[2])
    x = x * transform[0, 0] + transform[0, 3]
    y = y * transform[1, 1] + transform[1, 3]
    z = z * transform[2, 2] + transform[2, 3]
    return (x,y,z)


def get_homo_pc(pc):
    pc = pc.reshape(-1, 3)
    homo_pc = np.ones((pc.shape[0],4))
    homo_pc[:,:3] = pc

    return homo_pc


'''
description: 
param {tensor} input tensor (187,27,3)
param {transform} rotation and translation
param {scale} scale
return {*} transformed input
'''
def scale_input(tensor, transform=None, scale=None):
    if transform is not None:
        t_shape = tensor.shape 
        tensor = transform_3D_grid(tensor.view(-1, 3), transform=transform)
        tensor = tensor.view(t_shape)

    if scale is not None:
        tensor = tensor * scale

    return tensor

def transform_3D_grid(grid_3d, transform=None, scale=None):
    if scale is not None:
        grid_3d = grid_3d * scale
    if transform is not None:
        R1 = transform[None, None, None, 0, :3]
        R2 = transform[None, None, None, 1, :3]
        R3 = transform[None, None, None, 2, :3]

        grid1 = (R1 * grid_3d).sum(-1, keepdim=True)
        grid2 = (R2 * grid_3d).sum(-1, keepdim=True)
        grid3 = (R3 * grid_3d).sum(-1, keepdim=True)

        grid_3d = torch.cat([grid1, grid2, grid3], dim=-1)

        trans = transform[None, None, None, :3, 3]
        grid_3d = grid_3d + trans

    return grid_3d



'''
description: 
param {*} points: (x_dim, y_dim, z_dim, 3) or (N, 3)
param {*} transform: 4x4
param {*} scale
return {*}
'''

def transform_points(points, transform):
    R1 = transform[None, None, None, 0, :3]
    R2 = transform[None, None, None, 1, :3]
    R3 = transform[None, None, None, 2, :3]

    points1 = (R1 * points).sum(-1, keepdim=True)
    points2 = (R2 * points).sum(-1, keepdim=True)
    points3 = (R3 * points).sum(-1, keepdim=True)

    points = torch.cat([points1, points2, points3], dim=-1)

    trans = transform[None, None, None, :3, 3]
    points = points + trans

    return points

def scale_points(points, scale):
    return points * scale