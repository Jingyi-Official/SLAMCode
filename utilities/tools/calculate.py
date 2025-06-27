'''
Description: 
Author: 
Date: 2022-09-19 21:49:24
LastEditTime: 2023-03-07 15:50:39
LastEditors: Jingyi Wan
Reference: 
'''
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
# from functorch import vmap, jacrev # , grad

def cal_gradient_torch(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    return points_grad


'''
description: 
param {*} inputs have to be (-1,3)
param {*} model
return {*}
'''
# def cal_gradient_functorch(inputs, model):
#     points_grad = vmap(jacrev(model))(inputs)
#     return points_grad

