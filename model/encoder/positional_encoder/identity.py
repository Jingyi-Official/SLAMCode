'''
Description: modified from nerftudio
Author: 
Date: 2023-03-18 16:11:48
LastEditTime: 2023-03-18 16:38:55
LastEditors: Jingyi Wan
Reference: 
'''
import torch 
from torchtyping import TensorType


class PositionalEncoding(torch.nn.Module):
    """Identity encoding (Does not modify input)"""
    def __init__(self, in_dim: int) -> None:
        super(PositionalEncoding, self).__init__()
        
        if in_dim <= 0:
            raise ValueError("Input dimension should be greater than zero")
        self.in_dim = in_dim


    def get_out_dim(self) -> int:
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        return self.in_dim

    def forward(self, in_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
        return in_tensor
    

