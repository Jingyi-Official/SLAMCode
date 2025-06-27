'''
Description: 
Author: 
Date: 2022-11-18 14:58:54
LastEditTime: 2023-03-02 15:16:08
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Reference: 
'''
import torch
import numpy as np
import torch.autograd.profiler as profiler


class PositionalEncoding(torch.nn.Module):
    """
    Implement NeRF's positional encoding
    """

    def __init__(self, 
                 in_dim=3,
                 min_deg=0,
                 max_deg=6, 
                 n_freqs=5,
                 freq_factor=torch.pi,
                 include_input=True,
            ):
        super().__init__()

        self.in_dim = in_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.n_freqs = n_freqs
        self.freq_bands = freq_factor * 2.0 ** torch.linspace(self.min_deg, self.max_deg, self.n_freqs)
        self.include_input = include_input

        self.out_dim = 2 * self.in_dim * self.n_freqs 
        if self.include_input:
            self.out_dim += self.in_dim

    
    def forward(self, x):
        x_bands = torch.reshape(x[..., None] * self.freq_bands.type_as(x).to(x.device), list(x.shape[:-1]) + [-1])
        x_encoded = torch.sin(torch.cat([x_bands, x_bands + torch.pi / 2.0], dim=-1))

        if self.include_input:
            x_encoded = torch.cat([x_encoded, x], dim=-1)
        
        return x_encoded