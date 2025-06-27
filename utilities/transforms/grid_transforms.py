'''
Description: 
Author: 
Date: 2023-03-01 16:33:23
LastEditTime: 2023-03-13 13:12:38
LastEditors: Jingyi Wan
Reference: 
'''


import torch
import numpy as np
from utilities.transforms.point_transforms import transform_points,scale_points




'''
description: 
param {*} grid_range: the range of the axis [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]
param {*} grid_dim: the dim of each axis [256, 256, 256]
param {*} device
param {*} transform
param {*} scale
return {*}
'''
def make_3D_grid(grid_range, grid_dim, device, transform=None, scale=None):
    # generate the canonical coord
    x, y, z = torch.meshgrid(
        torch.linspace(grid_range[0][0], grid_range[0][1], steps=grid_dim[0], device=device),
        torch.linspace(grid_range[1][0], grid_range[1][1], steps=grid_dim[1], device=device),
        torch.linspace(grid_range[2][0], grid_range[2][1], steps=grid_dim[2], device=device),
    )
    
    grid = torch.cat((x[..., None], # torch.Size([256, 256, 256, 1])
                      y[..., None],
                      z[..., None]), 
                      dim=3) # torch.Size([256, 256, 256, 3])

    if transform is not None:
        transform = torch.from_numpy(transform).float().to(device)
    
    if scale is not None:
        scale = torch.from_numpy(scale).float().to(device)
    
    # transform from the canonical coord into the loaded coord
    grid = transform_points(scale_points(grid, scale=scale),transform=transform)

    return grid


'''
description: 
param {*} grid_range: the range of the axis [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]
param {*} grid_dim: the dim of each axis [256, 256, 256]
param {*} device
return {*}
'''
def make_grid(grid_range, grid_dim, device):
    x, y, z = torch.meshgrid(
        torch.linspace(grid_range[0][0], grid_range[0][1], steps=grid_dim[0], device=device),
        torch.linspace(grid_range[1][0], grid_range[1][1], steps=grid_dim[1], device=device),
        torch.linspace(grid_range[2][0], grid_range[2][1], steps=grid_dim[2], device=device),
    )
    return x, y, z
    



# '''
# description: 
# param {*} grid_range # [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]
# param {*} dim
# param {*} device
# param {*} transform
# param {*} scale
# return {*}
# '''
# def make_3D_grid(grid_range, dim, device, transform=None, scale=None):
#     # generate the canonical coord
#     t = torch.linspace(grid_range[0], grid_range[1], steps=dim, device=device) # torch.Size([256]) divide -1(included) to 1(included) as 256 parts
#     grid = torch.meshgrid(t, t, t) # generate the coordinates, x,y,z corresponding to t
#     grid_3d = torch.cat(
#         (grid[0][..., None], # torch.Size([256, 256, 256, 1])
#          grid[1][..., None],
#          grid[2][..., None]), dim=3
#     ) # torch.Size([256, 256, 256, 3])

#     if transform is not None:
#         transform = torch.from_numpy(transform).float().to(device)
    
#     if scale is not None:
#         scale = torch.from_numpy(scale).float().to(device)
    
#     # transform from the canonical coord into the loaded coord
#     grid_3d = transform_3D_grid(grid_3d, transform=transform, scale=scale)

#     return grid_3d



def get_grid_pts(dims, transform):
    x = np.arange(dims[0])
    y = np.arange(dims[1])
    z = np.arange(dims[2])

    (x,y,z) = transform_grid_pts(x, y, z, transform)
    
    return (x,y,z)


def get_grid_pts(dims, transform):
    x = np.arange(dims[0])
    y = np.arange(dims[1])
    z = np.arange(dims[2])
    x = x * transform[0, 0] + transform[0, 3]
    y = y * transform[1, 1] + transform[1, 3]
    z = z * transform[2, 2] + transform[2, 3]
    return (x,y,z)




def transform_grid_pts(x, y, z, transform):
    x = x * transform[0, 0] + transform[0, 3]
    y = y * transform[1, 1] + transform[1, 3]
    z = z * transform[2, 2] + transform[2, 3]

    return (x,y,z)


# Get voxel grid coordinates
def make_volume(dim):
    x, y, z = torch.meshgrid(
        torch.arange(0, dim[0]),
        torch.arange(0, dim[1]),
        torch.arange(0, dim[2]),
    )
    
    return torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1).long()



def xyz_to_points(xyz):
    '''
    xyz: the index of the x/y/z in each axis
    '''
    grid= torch.meshgrid(torch.Tensor(xyz[0]), torch.Tensor(xyz[1]), torch.Tensor(xyz[2]))
    pts = torch.cat(
        (grid[0][..., None], 
        grid[1][..., None],
        grid[2][..., None]), dim=3
    ).view(-1, 3)
    return pts

