'''
Description: This is for the coordinates transformations, the input pc should be homogenuous
Author: 
Date: 2022-12-30 16:40:42
LastEditTime: 2022-12-30 17:40:45
LastEditors: Jingyi Wan
Reference: 
'''
import numpy as np

def convert_wc_to_cc(Pw, T):
    """
    world coord -> camera coord: R * pc + t):
    Pw = [x,x,x,1]
    T = [R|t]
    :return: Pc
    """
    Pc = np.matmul(T, Pw)
    
    return Pc[:,:3,:]

def convert_cc_to_uvd(Pc, K):
    """
    camera coord -> pixel coord: (f / dx) * (X / Z) = f * (X / Z) / dx
    :return:
    """
    uvd = np.matmul(K, Pc)
    return uvd

