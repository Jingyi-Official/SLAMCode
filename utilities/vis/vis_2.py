import numpy as np
import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import ListedColormap



def get_colormap(sdf_range=[-2, 2], surface_cutoff=0.01):
    white = np.array([1., 1., 1., 1.])
    sdf_range[1] += surface_cutoff - (sdf_range[1] % surface_cutoff) # 2.0
    sdf_range[0] -= surface_cutoff - (-sdf_range[0] % surface_cutoff) # -2.0

    positive_n_cols = int(sdf_range[1] / surface_cutoff) # 200
    viridis = cm.get_cmap('viridis', positive_n_cols) # Typically, Colormap instances are used to convert data values (floats) from the interval [0, 1] to the RGBA color that the respective Colormap represents.
    positive_colors = viridis(np.linspace(0.2, 1, int(positive_n_cols))) # (200, 4)
    positive_colors[0] = white

    negative_n_cols = int(-sdf_range[0] / surface_cutoff) # 200
    redpurple = cm.get_cmap('RdPu', negative_n_cols).reversed()
    negative_colors = redpurple(np.linspace(0., 0.7, negative_n_cols)) # (200, 4)
    negative_colors[-1] = white

    colors = np.concatenate(
        (negative_colors, white[None, :], positive_colors), axis=0) # (401, 4)
    sdf_cmap = ListedColormap(colors) # The colormap used to map normalized data values to RGBA colors.

    norm = mpl.colors.Normalize(sdf_range[0], sdf_range[1]) # The normalizing object which scales data, typically into the interval [0, 1].initializes its scaling based on the first data processed.
    sdf_cmap_fn = cm.ScalarMappable(norm=norm, cmap=sdf_cmap)
    # plt.colorbar(sdf_cmap_fn)
    # plt.show()
    return sdf_cmap_fn