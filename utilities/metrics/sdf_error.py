'''
Description: 
Author: 
Date: 2022-09-19 21:49:24
LastEditTime: 2022-11-14 18:37:33
LastEditors: Jingyi Wan
Reference: 
'''
import torch
import numpy as np

def binned_errors(
    sdf_diff, gt_sdf,
    bin_limits=np.array([-1e99, 0., 0.1, 0.2, 0.5, 1., 1e99])
):
    """
        Sort loss into bins at different distances from the surface.
        sdf_diff: np array, absolute difference between predicted and gt sdf
    """
    if isinstance(gt_sdf, torch.Tensor):
        bins_lb = torch.tensor(bin_limits[:-1]).to(gt_sdf.device)
        bins_ub = torch.tensor(bin_limits[1:]).to(gt_sdf.device)
        locical_op = torch.logical_and
    else:
        bins_lb = bin_limits[:-1]
        bins_ub = bin_limits[1:]
        locical_op = np.logical_and

    lb_masks = gt_sdf >= bins_lb[:, None]
    ub_masks = gt_sdf < bins_ub[:, None]
    masks = locical_op(lb_masks, ub_masks)

    masked_diffs = sdf_diff * masks
    bins_loss = masked_diffs.sum(1)
    bins_loss = bins_loss / masks.sum(1)

    return bins_loss.tolist()


def binned_metrics(dep_var, ind_var, bin_limits=np.array([-1e99, 0., 0.1, 0.2, 0.5, 1., 1e99])):
    if isinstance(ind_var, torch.Tensor):
        bins_lb = torch.tensor(bin_limits[:-1]).to(ind_var.device)
        bins_ub = torch.tensor(bin_limits[1:]).to(ind_var.device)
        locical_op = torch.logical_and
    else:
        bins_lb = bin_limits[:-1]
        bins_ub = bin_limits[1:]
        locical_op = np.logical_and

    lb_masks = ind_var >= bins_lb[:, None]
    ub_masks = ind_var < bins_ub[:, None]

    masks = locical_op(lb_masks, ub_masks)
    masked_diffs = dep_var * masks
    
    bins_loss = masked_diffs.sum(1)
    bins_loss = bins_loss / masks.sum(1)

    # bins_loss = torch.nan_to_num(bins_loss)

    return bins_loss.tolist()



def binned_metrics_ind_stats(ind_var, bin_limits=np.array([-1e99, 0., 0.1, 0.2, 0.5, 1., 1e99])):
    '''
    Added the number of points in the indep region
    Added the region variance
    '''
    if isinstance(ind_var, torch.Tensor):
        bins_lb = torch.tensor(bin_limits[:-1]).to(ind_var.device)
        bins_ub = torch.tensor(bin_limits[1:]).to(ind_var.device)
        locical_op = torch.logical_and
    else:
        bins_lb = bin_limits[:-1]
        bins_ub = bin_limits[1:]
        locical_op = np.logical_and

    lb_masks = ind_var >= bins_lb[:, None]
    ub_masks = ind_var < bins_ub[:, None]

    masks = locical_op(lb_masks, ub_masks) # whether each place corresponding to the relevant region
    
    # get statistics
    bins = masks.sum(1) # number of points in the region
    bins_mean = bins.cpu().numpy().mean()
    bins_std = bins.cpu().numpy().std()
    
    return bins.tolist(), bins_mean, bins_std




def binned_metrics_dep_stats(dep_var, ind_var, bin_limits=np.array([-1e99, 0., 0.1, 0.2, 0.5, 1., 1e99])):
    '''
    Added the number of points in the indep region
    Added the region variance
    '''
    if isinstance(ind_var, torch.Tensor):
        bins_lb = torch.tensor(bin_limits[:-1]).to(ind_var.device)
        bins_ub = torch.tensor(bin_limits[1:]).to(ind_var.device)
        locical_op = torch.logical_and
    else:
        bins_lb = bin_limits[:-1]
        bins_ub = bin_limits[1:]
        locical_op = np.logical_and

    lb_masks = ind_var >= bins_lb[:, None]
    ub_masks = ind_var < bins_ub[:, None]

    masks = locical_op(lb_masks, ub_masks) # whether each place corresponding to the relevant region
    
    # corresponing dependent variable
    masked_dep_diffs = dep_var * masks # error in different range
    
    bins_dep_stats = masked_dep_diffs.sum(1) / masks.sum(1) # average dep variable in the region
    bins_dep_stats_mean = bins_dep_stats.cpu().numpy().mean()
    bins_dep_stats_std = bins_dep_stats.cpu().numpy().std()
    
    return bins_dep_stats.tolist(), bins_dep_stats_mean, bins_dep_stats_std



def binned_metrics_with_statistics(dep_var, ind_var, bin_limits=np.array([-1e99, 0., 0.1, 0.2, 0.5, 1., 1e99])):
    '''
    Added the number of points in the indep region
    Added the region variance
    '''
    if isinstance(ind_var, torch.Tensor):
        bins_lb = torch.tensor(bin_limits[:-1]).to(ind_var.device)
        bins_ub = torch.tensor(bin_limits[1:]).to(ind_var.device)
        locical_op = torch.logical_and
    else:
        bins_lb = bin_limits[:-1]
        bins_ub = bin_limits[1:]
        locical_op = np.logical_and

    lb_masks = ind_var >= bins_lb[:, None]
    ub_masks = ind_var < bins_ub[:, None]

    masks = locical_op(lb_masks, ub_masks) # whether each place corresponding to the relevant region
    
    
    # corresponding independent variable
    masked_ind_diffs = ind_var * masks # gtsdf in different range
    
    # corresponing dependent variable
    masked_dep_diffs = dep_var * masks # error in different range
    
    # get statistics
    bins = masks.sum(1) # number of points in the region
    bins_mean = bins.cpu().numpy().mean()
    bins_std = bins.cpu().numpy().std()

    bins_dep_stats = masked_dep_diffs.sum(1) / bins # average dep variable in the region
    bins_dep_stats = torch.nan_to_num(bins_dep_stats)
    bins_dep_stats_mean = bins_dep_stats.cpu().numpy().mean()
    bins_dep_stats_std = bins_dep_stats.cpu().numpy().std()

    # bins_indep_stats = masked_ind_diffs.sum(1) / bins
    # bins_indep_stats = torch.nan_to_num(bins_indep_stats)
    # bins_indep_stats_mean = bins_indep_stats.cpu().numpy().mean()
    # bins_indep_stats_std = bins_indep_stats.cpu().numpy().std()
    
    
    return bins.tolist(), bins_mean, bins_std, bins_dep_stats.tolist(), bins_dep_stats_mean, bins_dep_stats_std
    # return bins_dep_stats.tolist(), bins.tolist(), bins_indep_stats.tolist(), bins_mean, bins_std, bins_dep_stats_mean, bins_dep_stats_std, bins_indep_stats_mean, bins_indep_stats_std