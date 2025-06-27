'''
Description: the encoding of the tensorf: tensorcp; rewrite from nerfstudio
Author: 
Date: 2022-11-22 10:38:59
LastEditTime: 2023-03-18 20:26:50
LastEditors: Jingyi Wan
Reference: 
'''
"""
Implements image encoders
"""
import torch
from torch import nn
import torchvision
import numpy as np
from torchtyping import TensorType
import torch.nn.functional as F
from utilities.tools.grid_sample_gradfix import grid_sample


class CPDecoding(nn.Module):
    """Learned CANDECOMP/PARFAC (CP) decomposition encoding used in TensoRF

    Args:
        resolution: Resolution of grid.
        num_components: Number of components per dimension.
        init_scale: Initialization scale.
    """

    def __init__(self, 
                 resolution: int = 256, 
                 num_components: int = 24, 
                 init_scale: float = 0.1) -> None:
        super().__init__()

        self.resolution = resolution
        self.num_components = num_components

        # TODO Learning rates should be different for these
        self.line_coef = nn.Parameter(init_scale * torch.randn((3, num_components, resolution, 1)))

        self.apply(self._init_weights)

    def _init_weights(self, module, init_fn=torch.nn.init.xavier_normal_):
        if isinstance(module, nn.Parameter):
            init_fn(module.weight)
        
    
    def get_out_dim(self) -> int:
        return self.num_components

    def forward(self, in_tensor: TensorType["bs":..., "input_dim"], do_grad=False) -> TensorType["bs":..., "output_dim"]:
        line_coord = torch.stack([in_tensor[..., 2], in_tensor[..., 1], in_tensor[..., 0]])  # [3, ...]
        line_coord = torch.stack([torch.zeros_like(line_coord), line_coord], dim=-1)  # [3, ...., 2]

        # Stop gradients from going to sampler
        line_coord = line_coord.view(3, -1, 1, 2)

        line_features = grid_sample(self.line_coef, line_coord)  # [3, Components, -1, 1]

        features = torch.prod(line_features, dim=0)
        features = torch.moveaxis(features.view(self.num_components, *in_tensor.shape[:-1]), 0, -1)

        out_tensor = torch.sum(features, dim=-1)

        # if do_grad:
        #     line_coef_grad = torch.zeros_like(self.line_coef) # torch.Size([3, 16, 128, 1])
        #     line_coef_grad[...,1:-1,:] = (self.line_coef[...,2:,:] - self.line_coef[...,0:-2,:]) 
        #     line_coef_grad = line_coef_grad * (self.resolution - 1) / 4
        #     line_feature_grad = F.grid_sample(line_coef_grad, line_coord, align_corners=True) 
        #     dsdx = line_feature_grad[0] * line_features[[1,2]].prod(dim=0) # torch.Size([16, 27000, 1])
        #     # dsdx = torch.moveaxis(dsdx.view(self.num_components, *in_tensor.shape[:-1]), 0, -1)
        #     dsdy = line_feature_grad[1] * line_features[[0,2]].prod(dim=0)
        #     # dsdy = torch.moveaxis(dsdy.view(self.num_components, *in_tensor.shape[:-1]), 0, -1)
        #     dsdz = line_feature_grad[2] * line_features[[0,1]].prod(dim=0)
        #     # dsdz = torch.moveaxis(dsdz.view(self.num_components, *in_tensor.shape[:-1]), 0, -1)
        #     feature_grad = torch.cat([dsdx, dsdy, dsdz], dim=-1) # torch.Size([16, 27000, 3])
        #     feature_grad = torch.moveaxis(feature_grad.view(self.num_components, *in_tensor.shape), 0, -1) # torch.Size([16, 27000, 3])
        # else:
        #     feature_grad = None
        
        return out_tensor  # [..., Components]

    @torch.no_grad()
    def upsample_grid(self, resolution: int) -> None:
        """Upsamples underyling feature grid

        Args:
            resolution: Target resolution.
        """

        self.line_coef.data = F.interpolate(
            self.line_coef.data, size=(resolution, 1), mode="bilinear", align_corners=True
        )

        self.resolution = resolution

