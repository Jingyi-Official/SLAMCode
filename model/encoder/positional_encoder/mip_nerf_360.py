'''
Description: 
Author: 
Date: 2022-09-19 21:50:03
LastEditTime: 2023-03-18 16:18:21
LastEditors: Jingyi Wan
Reference: 
'''
import torch
import numpy as np


class PositionalEncoding(torch.nn.Module):
    def __init__(
        self,
        in_dim=3,
        min_deg=0,
        max_deg=5,
        n_freqs=6,
        freq_factor=1,
        include_input=True,
    ):
        super(PositionalEncoding, self).__init__()
        
        self.dirs = torch.tensor([
            0.8506508, 0, 0.5257311,
            0.809017, 0.5, 0.309017,
            0.5257311, 0.8506508, 0,
            1, 0, 0,
            0.809017, 0.5, -0.309017,
            0.8506508, 0, -0.5257311,
            0.309017, 0.809017, -0.5,
            0, 0.5257311, -0.8506508,
            0.5, 0.309017, -0.809017,
            0, 1, 0,
            -0.5257311, 0.8506508, 0,
            -0.309017, 0.809017, -0.5,
            0, 0.5257311, 0.8506508,
            -0.309017, 0.809017, 0.5,
            0.309017, 0.809017, 0.5,
            0.5, 0.309017, 0.809017,
            0.5, -0.309017, 0.809017,
            0, 0, 1,
            -0.5, 0.309017, 0.809017,
            -0.809017, 0.5, 0.309017,
            -0.809017, 0.5, -0.309017
        ]).reshape(-1, 3).T

        self.in_dim=in_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.n_freqs = n_freqs
        self.freq_bands = freq_factor * 2.0 ** torch.linspace(self.min_deg, self.max_deg, self.n_freqs) # tensor([ 1.,  2.,  4.,  8., 16., 32.])
        self.include_input = include_input

        self.out_dim = 2 * self.dirs.shape[1] * self.n_freqs
        if self.include_input:
            self.out_dim += self.in_dim
 

    def forward(self, x):
        x_proj = torch.matmul(x, self.dirs.to(x.device)) # project into the 21 dim vectors 
        x_bands = torch.reshape(x_proj[..., None] * self.freq_bands.to(x.device), list(x_proj.shape[:-1]) + [-1])
        x_encoded = torch.sin(torch.cat([x_bands, x_bands + torch.pi / 2.0], dim=-1)) 
        
        if self.include_input:
            x_encoded = torch.cat([x, x_encoded], dim=-1)

        return x_encoded
    
    
    
    