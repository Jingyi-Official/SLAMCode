'''
Description: the encoding of the tensorf: tensorvm
Author: 
Date: 2023-02-27 09:35:48
LastEditTime: 2023-03-18 21:45:20
LastEditors: Jingyi Wan
Reference: 
'''
import torch
from torch import nn
import torchvision
import numpy as np
from torchtyping import TensorType
import torch.nn.functional as F
from utilities.tools.grid_sample_gradfix import grid_sample


class VMEncoding(nn.Module):
    """Learned vector-matrix encoding proposed by TensoRF

    Args:
        resolution: Resolution of grid.
        num_components: Number of components per dimension.
        init_scale: Initialization scale.
    """

    plane_coef: TensorType[3, "num_components", "resolution", "resolution"]
    line_coef: TensorType[3, "num_components", "resolution", 1]

    def __init__(
        self,
        resolution: int = 128,
        num_components: int = 24,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__()

        self.resolution = resolution
        self.num_components = num_components

        self.plane_coef = nn.Parameter(init_scale * torch.randn((3, num_components, resolution, resolution)))
        self.line_coef = nn.Parameter(init_scale * torch.randn((3, num_components, resolution, 1)))

    
        self.apply(self._init_weights)

    def _init_weights(self, module, init_fn=torch.nn.init.xavier_normal_):
        if isinstance(module, nn.Linear):
            init_fn(module.weight)
    
    def get_out_dim(self) -> int:
        return self.num_components * 3

    def forward(self, in_tensor: TensorType["bs":..., "input_dim"], do_grad=False) -> TensorType["bs":..., "output_dim"]:
        """Compute encoding for each position in in_positions

        Args:
            in_tensor: position inside bounds in range [-1,1],

        Returns: Encoded position
        """
        plane_coord = torch.stack([in_tensor[..., [0, 1]], in_tensor[..., [0, 2]], in_tensor[..., [1, 2]]])  # [3,...,2]
        line_coord = torch.stack([in_tensor[..., 2], in_tensor[..., 1], in_tensor[..., 0]])  # [3, ...]
        line_coord = torch.stack([torch.zeros_like(line_coord), line_coord], dim=-1)  # [3, ...., 2]

        # Stop gradients from going to sampler
        plane_coord = plane_coord.view(3, -1, 1, 2) #.detach()
        line_coord = line_coord.view(3, -1, 1, 2) #.detach()

        plane_features = grid_sample(self.plane_coef, plane_coord)  # [3, Components, -1, 1]
        line_features = grid_sample(self.line_coef, line_coord)  # [3, Components, -1, 1]

        features = plane_features * line_features  # [3, Components, -1, 1]
        features = torch.moveaxis(features.view(3 * self.num_components, *in_tensor.shape[:-1]), 0, -1)

        return features

