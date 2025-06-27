'''
Description: the revised version inspried by tensorf: to increase the inference ability, the encoding of the tensorf but replace the spatials into the mlp 
Author: 
Date: 2022-11-22 10:38:59
LastEditTime: 2023-03-18 21:45:30
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
from model.decoder.mlp import MLP


class CMLPEncoding(nn.Module):
    """Learned CANDECOMP/PARFAC (CP) decomposition encoding used in TensoRF

    Args:
        resolution: Resolution of grid.
        num_components: Number of components per dimension.
        init_scale: Initialization scale.
    """

    def __init__(self, 
                 in_dim,
                 num_layers,
                 layer_width,
                 out_dim,
                 skip_connections,
                 activation,
                 out_activation,
                 ) -> None:
        super().__init__()

        self.x_decoder = MLP(in_dim=in_dim, num_layers=num_layers,layer_width=layer_width, out_dim=out_dim, skip_connections=skip_connections, activation=activation, out_activation=out_activation)
        self.y_decoder = MLP(in_dim=in_dim, num_layers=num_layers,layer_width=layer_width, out_dim=out_dim, skip_connections=skip_connections, activation=activation, out_activation=out_activation)
        self.z_decoder = MLP(in_dim=in_dim, num_layers=num_layers,layer_width=layer_width, out_dim=out_dim, skip_connections=skip_connections, activation=activation, out_activation=out_activation)
        
    
    def get_out_dim(self) -> int:
        return self.num_components

    def forward(self, in_tensor: TensorType["bs":..., "input_dim"], do_grad=False) -> TensorType["bs":..., "output_dim"]:
        x_features = self.x_decoder(in_tensor[..., 0].unsqueeze(dim=-1))
        y_features = self.y_decoder(in_tensor[..., 1].unsqueeze(dim=-1))
        z_features = self.z_decoder(in_tensor[..., 2].unsqueeze(dim=-1))

        features = x_features * y_features * z_features
        
        return features 
    

