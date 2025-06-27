'''
Description: 
Author: 
Date: 2022-09-19 21:49:23
LastEditTime: 2023-03-01 18:39:04
LastEditors: Jingyi Wan
Reference: 
'''
from torch.utils.data import Dataset
import torch
import numpy as np
from utilities.dataformat.frame_data import expand_data
import copy
from tqdm import *


# class CacheDataset(Object):
#     def __call__(self, dataset, interval, mode="depth"):
#         if mode=="depth":
#             return DepthCacheDataset(dataset, interval)
#         elif mode=="rgbd":
#             return RGBDCacheDataset(dataset, interval)
    

class DepthCacheDataset(Dataset):
    
    def __init__(self, dataset, interval):
        self.dataset = dataset
        self.interval = interval
        self.cache = self.get_cache()

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        cache_ixs = self.get_cache_idxs(idx)

        depth = np.concatenate(([x[None, :] for x in self.cache[cache_ixs, 0]]))
        T = np.concatenate(([x[None, :] for x in self.cache[cache_ixs, 1]]))

        sample = {
            "depth": depth,
            "T": T
        }

        return sample


    def get_cache(self):
        n_frames = len(self.dataset)
        keep_ixs = np.arange(0, n_frames, self.interval) # interval = 5
        cache = []

        for idx in tqdm(range(n_frames)):
            if idx not in keep_ixs:
                continue

            sample = self.dataset[idx]
            cache.append((sample["depth"], sample["T"]))
        
        cache=np.array(cache)

        return cache

    def get_frame_idxs(self, idx):
        return np.arange(idx)

    def get_cache_idxs(self, idx):
        if idx > len(self.dataset):
            idx = len(self.dataset)
            
        frame_ixs = self.get_frame_idxs(idx)
        return [int(x/self.interval) for x in frame_ixs if x%self.interval==0]



class RGBDCacheDataset(Dataset):
    def __init__(self, dataset, interval):
        self.dataset = dataset
        self.interval = interval
        
        self.cache = self.get_cache()

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        cache_ixs = self.get_cache_idxs(idx)

        rgb = np.concatenate(([x[None, :] for x in self.cache[cache_ixs, 0]]))
        depth = np.concatenate(([x[None, :] for x in self.cache[cache_ixs, 1]]))
        T = np.concatenate(([x[None, :] for x in self.cache[cache_ixs, 2]]))

        sample = {
            "rgb": rgb,
            "depth": depth,
            "T": T
        }

        return sample


    def get_cache(self):
        n_frames = len(self.dataset)
        keep_ixs = np.arange(0, n_frames, self.interval) # interval = 5
        cache = []

        for idx in tqdm(range(n_frames)):
            if idx not in keep_ixs:
                continue

            sample = self.dataset[idx]
            cache.append((sample["rgb"], sample["depth"], sample["T"]))
        
        cache=np.array(cache)

        return cache

    def get_frame_idxs(self, idx):
        return np.arange(idx)

    def get_cache_idxs(self, idx):
        if idx > len(self.dataset):
            idx = len(self.dataset)
            
        frame_ixs = self.get_frame_idxs(idx)
        return [int(x/self.interval) for x in frame_ixs if x%self.interval==0]






    
