'''
Description: 
Author: 
Date: 2022-06-23 13:38:14
LastEditTime: 2023-03-27 03:36:14
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Reference: 
'''

import torch
from utilities.geometry import origin_dirs_W

def bounds_pc(pc, z_vals, depth_sample, do_grad=True):
    with torch.set_grad_enabled(False):
        surf_pc = pc[:, 0] # torch.Size([373, 3]) get the point exactly on the surface
        diff = pc[:, :, None] - surf_pc # torch.Size([373, 27, 1, 3]) - torch.Size([373, 3])) = torch.Size([373, 27, 373, 3]) difference of each point from all sampled surf_pc
        dists = diff.norm(dim=-1) # torch.Size([373, 27, 373])
        dists, closest_ixs = dists.min(axis=-1) # torch.Size([373, 27])
        behind_surf = z_vals > depth_sample[:, None] # torch.Size([373, 27]) - torch.Size([373, 1]) = torch.Size([373, 27]) 1048:4001
        dists[behind_surf] *= -1
        bounds = dists

        grad = None
        if do_grad:
            ix1 = torch.arange(
                diff.shape[0])[:, None].repeat(1, diff.shape[1]) # torch.Size([187, 27])
            ix2 = torch.arange(
                diff.shape[1])[None, :].repeat(diff.shape[0], 1) # torch.Size([187, 27])
            grad = diff[ix1, ix2, closest_ixs] # torch.Size([187, 27, 3]) # the distance of the closest point to the surface
            grad = grad[:, 1:] # leave the surface
            grad = grad / grad.norm(dim=-1)[..., None]
            # flip grad vectors behind the surf
            grad[behind_surf[:, 1:]] *= -1

        # # vis gradient vector
        # surf_pc_tm = trimesh.PointCloud(
        #     surf_pc.reshape(-1, 3).cpu(), colors=[255, 0, 0])
        # pc_tm = trimesh.PointCloud(pc[:, 1:].reshape(-1, 3).cpu())
        # closest_surf_pts = surf_pc[closest_ixs].reshape(-1, 3)
        # lines = torch.cat((
        #     closest_surf_pts[:, None, :],
        #     pc.reshape(-1, 3)[:, None, :]), dim=1)
        # paths = trimesh.load_path(lines.cpu())
        # trimesh.Scene([surf_pc_tm, pc_tm, paths]).show()

    return bounds, grad

def bounds_ray(depth_sample, z_vals, dirs_C_sample, T_WC_sample, do_grad):
    bounds = depth_sample[:, None] - z_vals 
    z_to_euclidean_depth = dirs_C_sample.norm(dim=-1) 
    bounds = z_to_euclidean_depth[:, None] * bounds

    grad = None
    if do_grad:
        grad = grad_ray(T_WC_sample, dirs_C_sample, z_vals.shape[1] - 1)

    return bounds, grad


def bounds_normal(
    depth_sample, z_vals, dirs_C_sample, norm_sample, normal_trunc_dist,
    T_WC_sample, do_grad,
):
    ray_bounds = bounds_ray(depth_sample, z_vals, dirs_C_sample)

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


def grad_ray(T_WC_sample, dirs_C_sample, n_samples):
    """ Returns the negative of the viewing direction vector """
    _, dirs_W = origin_dirs_W(T_WC_sample, dirs_C_sample)
    grad = - dirs_W[:, None, :].repeat(1, n_samples, 1)

    return grad