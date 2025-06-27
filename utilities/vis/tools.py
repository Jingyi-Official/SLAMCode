'''
Description: 
Author: 
Date: 2022-11-08 19:25:21
LastEditTime: 2022-11-08 19:25:57
LastEditors: Jingyi Wan
Reference: 
'''
import numpy as np
import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import ListedColormap


def get_colormap(sdf_range=[-2, 2], surface_cutoff=0.01):
    white = np.array([1., 1., 1., 1.])
    sdf_range[1] += surface_cutoff - (sdf_range[1] % surface_cutoff)
    sdf_range[0] -= surface_cutoff - (-sdf_range[0] % surface_cutoff)

    positive_n_cols = int(sdf_range[1] / surface_cutoff)
    viridis = cm.get_cmap('viridis', positive_n_cols)
    positive_colors = viridis(np.linspace(0.2, 1, int(positive_n_cols)))
    positive_colors[0] = white

    negative_n_cols = int(-sdf_range[0] / surface_cutoff)
    redpurple = cm.get_cmap('RdPu', negative_n_cols).reversed()
    negative_colors = redpurple(np.linspace(0., 0.7, negative_n_cols))
    negative_colors[-1] = white

    colors = np.concatenate(
        (negative_colors, white[None, :], positive_colors), axis=0)
    sdf_cmap = ListedColormap(colors)

    norm = mpl.colors.Normalize(sdf_range[0], sdf_range[1])
    sdf_cmap_fn = cm.ScalarMappable(norm=norm, cmap=sdf_cmap)
    # plt.colorbar(sdf_cmap_fn)
    # plt.show()
    return sdf_cmap_fn