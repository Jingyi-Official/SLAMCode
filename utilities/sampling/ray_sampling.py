'''
Description: 
Author: 
Date: 2022-06-17 15:49:40
LastEditTime: 2023-03-06 16:15:47
LastEditors: Jingyi Wan
Reference: 
'''
import torch
from utilities.geometry import origin_dirs_W




def sample_points(depth_sample, T_sample, dir_camcoord_sample, min_depth, dist_behind_surf, n_strat_samples, strat_bin_len, n_surf_samples, surf_std):
    min_depth = min_depth
    max_depth = depth_sample + dist_behind_surf

    n_strat_samples = n_strat_samples
    n_surf_samples = n_surf_samples

    pc, z_vals = sample_along_rays(
        T_sample,
        dir_camcoord_sample,
        min_depth,
        max_depth,
        n_stratified_samples = n_strat_samples,
        stratified_bin_length = strat_bin_len,
        n_surf_samples = n_surf_samples,
        surf_std = surf_std,
        gt_depth=depth_sample,
        grad=False,
    )

    return pc, z_vals
'''
description: 
param {T_WC} pose for each chosen pixels [187,4,4]
param {min_depth} set 0.07
param {max_depth} for each chosen pixel, the depth value[187]
param {n_stratified_samples} set 19 as the number of stratified samples
param {n_surf_samples} set 8 as the number of surface samples
param {dirs_C} directions [187,3]
param {gt_depth} depth sample [187]
param {grad} When sampling, there is no gradient

return {*}
'''
'''
T_sample,
    min_depth,
    max_depth,
    n_strat_samples,
    n_surf_samples,
    dir_camcoord_sample,
    depth_sample,
    grad=False,

'''


def sample_along_rays(
    T_WC,
    dirs_C,
    min_depth,
    max_depth,
    n_stratified_samples=19,
    stratified_bin_length=None,
    n_surf_samples=8,
    surf_std=0.1,
    gt_depth=None,
    grad=False,
):
    with torch.set_grad_enabled(grad):
        # rays in world coordinate
        origins, dirs_W = origin_dirs_W(T_WC, dirs_C)

        origins = origins.view(-1, 3) # torch.Size([187, 3])
        dirs_W = dirs_W.view(-1, 3) # torch.Size([187, 3])
        n_rays = dirs_W.shape[0] # 187

        # stratified sampling along rays # [total_n_rays, n_stratified_samples]
        z_vals = None
        if n_stratified_samples>0:
            z_vals = stratified_sample(
                min_depth, max_depth,
                n_rays, T_WC.device,
                n_stratified_samples, bin_length=stratified_bin_length
            )


        # if gt_depth is given, first sample at surface then around surface
        surface_z_vals = None #include depth
        if gt_depth is not None and n_surf_samples > 0:
            surface_z_vals = gt_depth[:, None]
            
            if n_surf_samples>1:
                offsets = torch.normal(torch.zeros(gt_depth.shape[0], n_surf_samples - 1), surf_std).to(T_WC.device) # generator=torch.manual_seed(1234)
                near_surf_z_vals = gt_depth[:, None] + offsets
                near_surf_z_vals = torch.clamp(
                    near_surf_z_vals,
                    torch.full(near_surf_z_vals.shape, min_depth).to(
                        T_WC.device),
                    max_depth[:, None]
                )
                surface_z_vals = torch.cat((surface_z_vals, near_surf_z_vals), dim=1)


        if z_vals is not None and surface_z_vals is not None:
            z_vals = torch.cat((surface_z_vals, z_vals), dim=1)
        elif z_vals is None and surface_z_vals is not None:
            z_vals = surface_z_vals

        # point cloud of 3d sample locations
        pc = origins[:, None, :] + (dirs_W[:, None, :] * z_vals[:, :, None])

    return pc, z_vals


'''
description: 
Random samples between min and max depth
One sample from within each bin.

If n_stratified_samples is passed then use fixed number of bins,
else if bin_length is passed use fixed bin size.

param {min_depth} 
param {max_depth}
param {n_rays}
param {device} 
param {n_stratified_samples}
param {bin_length} if not set the number of samples, use the fixed bin length

return {z_vals} sampled point along the ray (with incremental prob as the noise)

other notes:
bin_limits: 
bin_length: length for each bin (1/19 * l)
'''
def stratified_sample(
    min_depth, # 0.07
    max_depth, # torch.Size([187])
    n_rays, # 187
    device,
    n_stratified_samples, # 19
    bin_length=None,
):
    if n_stratified_samples is not None:  # fixed number of bins
        n_bins = n_stratified_samples
        if isinstance(max_depth, torch.Tensor):
            sample_range = (max_depth - min_depth)[:, None] # torch.Size([187, 1])
            bin_limits = torch.linspace(
                0, 1, n_bins + 1,
                device=device)[None, :]
            bin_limits = bin_limits.repeat(n_rays, 1) * sample_range # torch.Size([187, 20])
            bin_limits = bin_limits + min_depth
            bin_length = sample_range / (n_bins)
        else:
            bin_limits = torch.linspace(
                min_depth,
                max_depth,
                n_bins + 1,
                device=device,
            )[None, :]
            bin_length = (max_depth - min_depth) / (n_bins)

    elif bin_length is not None:  # fixed size of bins
        bin_limits = torch.arange(
            min_depth,
            max_depth,
            bin_length,
            device=device,
        )[None, :]
        n_bins = bin_limits.size(1) - 1

    increments = torch.rand(n_rays, n_bins, device=device) * bin_length # torch.Size([187, 19])
    # increments = 0.5 * torch.ones(n_rays, n_bins, device=device) * bin_length
    lower_limits = bin_limits[..., :-1] # torch.Size([187, 19])
    z_vals = lower_limits + increments

    return z_vals


def collate_fn(min_depth, max_depth, n_samples, device):
    max_n = int(max(n_samples).item())
    
    # padded_z = [torch.cat((torch.linspace(0, 1, int(each.item()) + 1,device=device), torch.full((max_n-int(each.item()),),float("inf"), device=device)), dim=0) for each in n_samples]
    padded_z = [torch.cat((torch.linspace(0, 1, int(each.item()) + 1,device=device), torch.full((max_n-int(each.item()),),min_depth, device=device)), dim=0) for each in n_samples]


    

    
    # # lower_limits = bin_limits[..., :-1] # torch.Size([187, 19])
    # # z_vals = lower_limits + increments
    # z_vals = None

    return torch.stack(padded_z)