'''
Description: 
Author: 
Date: 2022-09-19 21:50:03
LastEditTime: 2023-03-18 19:25:32
LastEditors: Jingyi Wan
Reference: 
'''

import torch
import torch.nn as nn
from utilities.transforms.point_transforms import transform_points, scale_points
from utilities.tools.calculate import cal_gradient_torch

class TensoRF(nn.Module):
    def __init__(
        self,
        scale_input,
        transform_input,
        scale_output,
        positional_encoder,
        decoder,
        **kwargs,
    ) -> None:
        super().__init__()

        self.scale_input = scale_input
        self.transform_input = transform_input

        self.positional_encoder = positional_encoder
        self.decoder = decoder

        self.scale_output = scale_output

        self.apply(self._init_weights)


    def _init_weights(self, module, init_fn=torch.nn.init.xavier_normal_):
        if isinstance(module, nn.Linear):
            init_fn(module.weight)
    
    
    def forward(self, x, noise_std=None, do_grad=False):
        
        x_ts = scale_points(transform_points(x, transform=self.transform_input), scale=self.scale_input).squeeze()
        
        x_pe = self.positional_encoder(x_ts)
        
        y = self.decoder(x_pe)

        if noise_std is not None: 
            noise = torch.randn(y.shape, device=x.device) * noise_std
            y = y + noise
            
        y = y * self.scale_output
        y = y.squeeze(-1)

        grad = cal_gradient_torch(x, y) if do_grad else None
        
        return y, grad


