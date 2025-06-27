'''
Description: 
Author: 
Date: 2022-06-17 15:18:51
LastEditTime: 2023-03-01 16:35:04
LastEditors: Jingyi Wan
Reference: 
'''
import torch
import numpy as np

def get_ray_direction_camcoord(B, H, W, fx, fy, cx, cy, depth_type='z'): # get the camera coordinate when depth = 1
    c, r = torch.meshgrid(torch.arange(W), # torch.Size([1200, 680])
                          torch.arange(H))
    c, r = c.t().float(), r.t().float() # torch.Size([680, 1200])
    size = [B, H, W] # [1, 680, 1200]

    C = torch.empty(size) # torch.Size([1, 680, 1200])
    R = torch.empty(size) # torch.Size([1, 680, 1200])
    C[:, :, :] = c[None, :, :]
    R[:, :, :] = r[None, :, :]

    z = torch.ones(size)
    x = (C - cx) / fx
    y = (R - cy) / fy

    dirs = torch.stack((x, y, z), dim=3)
    if depth_type == 'euclidean':
        norm = torch.norm(dirs, dim=3)
        dirs = dirs * (1. / norm)[:, :, :, None]

    return dirs

'''
description: 
param {T_WC} pose for each chosen frame
param {dirs_C} transformed x y for the image, z = 1
return {origins} translation parameters
return {dirs_W} rotated dirs
'''
def origin_dirs_W(T_WC, dirs_C):
    R_WC = T_WC[:, :3, :3] # rotation parameters
    dirs_W = (R_WC * dirs_C[..., None, :]).sum(dim=-1) # direction in the world
    origins = T_WC[:, :3, -1] # translation parameters

    return origins, dirs_W


'''
description: 
param {grid_range} range of one side for the generated canonical coord
param {dim} how many voxels
param {device} 
param {transform} from canonical center to the loaded mesh center
param {scale} from canonical scale to the loaded mesh scale
return {*}
'''



