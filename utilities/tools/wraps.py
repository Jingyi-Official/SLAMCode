'''
Description: 
Author: 
Date: 2022-10-21 10:59:33
LastEditTime: 2022-10-21 11:09:39
LastEditors: Jingyi Wan
Reference: 
'''
import numpy as np

def list_to_dict(inputs, keys=None):
    if keys == None:
        keys = np.arange(0, len(inputs), 1)
        keys = [str(k) for k in keys]

    return dict(zip(keys, inputs))
