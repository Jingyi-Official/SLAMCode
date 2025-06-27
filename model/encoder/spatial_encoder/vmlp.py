'''
Description: the revised version inspried by tensorf: to increase the inference ability, the encoding of the tensorf but replace the spatials into the mlp 
Author: 
Date: 2022-11-22 10:38:59
LastEditTime: 2023-03-18 21:58:30
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


class VMLPEncoding(nn.Module):
    """Learned CANDECOMP/PARFAC (CP) decomposition encoding used in TensoRF

    Args:
        resolution: Resolution of grid.
        num_components: Number of components per dimension.
        init_scale: Initialization scale.
    """

    def __init__(self, 
                 v_in_dim,
                 v_num_layers,
                 v_layer_width,
                 v_out_dim,
                 v_skip_connections,
                 v_activation,
                 v_out_activation,
                 p_in_dim,
                 p_num_layers,
                 p_layer_width,
                 p_out_dim,
                 p_skip_connections,
                 p_activation,
                 p_out_activation,
                 ) -> None:
        super().__init__()

        self.x_decoder = MLP(in_dim=v_in_dim, num_layers=v_num_layers,layer_width=v_layer_width, out_dim=v_out_dim, skip_connections=v_skip_connections, activation=v_activation, out_activation=v_out_activation)
        self.y_decoder = MLP(in_dim=v_in_dim, num_layers=v_num_layers,layer_width=v_layer_width, out_dim=v_out_dim, skip_connections=v_skip_connections, activation=v_activation, out_activation=v_out_activation)
        self.z_decoder = MLP(in_dim=v_in_dim, num_layers=v_num_layers,layer_width=v_layer_width, out_dim=v_out_dim, skip_connections=v_skip_connections, activation=v_activation, out_activation=v_out_activation)
        
        self.yz_decoder = MLP(in_dim=p_in_dim, num_layers=p_num_layers,layer_width=p_layer_width, out_dim=p_out_dim, skip_connections=p_skip_connections, activation=p_activation, out_activation=p_out_activation)
        self.xz_decoder = MLP(in_dim=p_in_dim, num_layers=p_num_layers,layer_width=p_layer_width, out_dim=p_out_dim, skip_connections=p_skip_connections, activation=p_activation, out_activation=p_out_activation)
        self.xy_decoder = MLP(in_dim=p_in_dim, num_layers=p_num_layers,layer_width=p_layer_width, out_dim=p_out_dim, skip_connections=p_skip_connections, activation=p_activation, out_activation=p_out_activation)
    
    def get_out_dim(self) -> int:
        return self.num_components

    def forward(self, in_tensor: TensorType["bs":..., "input_dim"], do_grad=False) -> TensorType["bs":..., "output_dim"]:
        x_features = self.x_decoder(in_tensor[..., 0].unsqueeze(dim=-1))
        yz_features = self.yz_decoder(in_tensor[..., [1, 2]])

        y_features = self.y_decoder(in_tensor[..., 1].unsqueeze(dim=-1))
        xz_features = self.xz_decoder(in_tensor[..., [0, 2]])

        z_features = self.z_decoder(in_tensor[..., 2].unsqueeze(dim=-1))
        xy_features = self.xy_decoder(in_tensor[..., [0, 1]])

        features = x_features*yz_features + y_features*xz_features + z_features*xy_features
        
        return features 
    

