'''
Description: 
Author: 
Date: 2022-06-23 13:37:59
LastEditTime: 2022-10-07 11:00:46
LastEditors: Jingyi Wan
Reference: 
'''
import torch
from utilities.bounds.ray_bounds import bounds_ray
from utilities.geometry import origin_dirs_W




def grad_ray(T_WC_sample, dirs_C_sample, n_samples):
    """ Returns the negative of the viewing direction vector """
    _, dirs_W = origin_dirs_W(T_WC_sample, dirs_C_sample)
    grad = - dirs_W[:, None, :].repeat(1, n_samples, 1)

    return grad


cosSim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

def bounds_normal(
    depth_sample, z_vals, dirs_C_sample, norm_sample, normal_trunc_dist,
    T_WC_sample, do_grad,
):
    ray_bounds, _ = bounds_ray(depth_sample, z_vals, dirs_C_sample)

    costheta = torch.abs(cosSim(-dirs_C_sample, norm_sample))

    # only apply correction out to truncation distance
    sub = normal_trunc_dist * (1. - costheta)
    normal_bounds = ray_bounds - sub[:, None]

    trunc_ixs = ray_bounds < normal_trunc_dist
    trunc_vals = (ray_bounds * costheta[:, None])[trunc_ixs]
    normal_bounds[trunc_ixs] = trunc_vals

    grad = None
    if do_grad:
        grad = grad_ray(T_WC_sample, dirs_C_sample, z_vals.shape[1] - 1)

    return normal_bounds, grad


