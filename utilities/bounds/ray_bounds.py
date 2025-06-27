'''
Description: 
Author: 
Date: 2022-06-23 13:37:45
LastEditTime: 2022-07-06 19:34:47
LastEditors: Jingyi Wan
Reference: 
'''
import torch
from utilities.geometry import origin_dirs_W


def grad_ray(T_WC_sample, dirs_C_sample, n_samples):
    """ Returns the negative of the viewing direction vector """
    _, dirs_W = origin_dirs_W(T_WC_sample, dirs_C_sample)
    grad = - dirs_W[:, None, :].repeat(1, n_samples, 1)

    return grad

'''
description: bounds by ray intersection

param {depth_sample} The actual depth of point sampled on surface
param {z_vals} The depth of sampled points along the ray
param {dirs_C_sample} Intrinsic parameters, the camera coordinates when depth = 1
param {T_WC_sample} Extrinsinc parameters
param {do_grad} 

return {*}
'''
def bounds_ray(depth_sample, z_vals, dirs_C_sample, T_WC_sample=None, do_grad=False):
    bounds = depth_sample[:, None] - z_vals
    z_to_euclidean_depth = dirs_C_sample.norm(dim=-1)
    bounds = z_to_euclidean_depth[:, None] * bounds

    grad = None
    if do_grad:
        grad = grad_ray(T_WC_sample, dirs_C_sample, z_vals.shape[1] - 1)

    return bounds, grad

    