'''
Description: 
Author: 
Date: 2022-09-19 21:49:23
LastEditTime: 2023-03-28 19:17:49
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Reference: 
'''
from torch.utils.data import Dataset
import torch
import trimesh
import pysdf
import numpy as np
import cv2
import os
import json
import scipy
from utilities.geometry import get_ray_direction_camcoord
from utilities.transforms.grid_transforms import get_grid_pts

class ScanNetDataset(Dataset):
    
    def __init__(self,
                 root_dir,
                 rgb_K_file, # intrinsic
                 depth_K_file,
                 T_file, # extrinsic
                 rgb_transform=None,
                 depth_transform=None,
                 scene_folder=None,
                 ):

        # read raw data from path -------------------------------------------
        self.root_dir = root_dir
        self.rgb_dir = os.path.join(self.root_dir, "frames", "color/")
        self.depth_dir = os.path.join(self.root_dir, "frames", "depth/")

        # intrinsic info
        self.rgb_Ks = np.loadtxt(os.path.join(root_dir, rgb_K_file)).reshape(3, 3)
        self.depth_Ks = np.loadtxt(os.path.join(root_dir, depth_K_file)).reshape(3, 3)
        self.Ks = self.depth_Ks
        
        # extrinsic info
        self.Ts = np.loadtxt(os.path.join(root_dir, T_file)).reshape(-1, 4, 4)

        # data augmentation / data conversion -------------------------------
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform

        # other gt files for eval
        self.scene_folder = scene_folder

    def __len__(self):
        return self.Ts.shape[0]

    def __getitem__(self, idx):
        # read raw data from path -------------------------------------------
        rgb_file = os.path.join(self.rgb_dir, "%d.jpg"%(idx))
        depth_file = os.path.join(self.depth_dir, "%d.png"%(idx))
        
        rgb = cv2.imread(rgb_file)
        depth = cv2.imread(depth_file, -1) 
        T = self.Ts[idx] 

        sample = {"rgb": rgb, "depth": depth, "T": T, "ndepth": depth}

        # data augmentation / data conversion -------------------------------
        if self.rgb_transform:
            sample["rgb"] = self.rgb_transform(sample["rgb"])

        if self.depth_transform:
            sample["ndepth"] = self.depth_transform(sample["ndepth"])
            sample["depth"] = self.depth_transform(sample["depth"])

        return sample

    @property
    def fps(self):
        return 30
    
    @property
    def depth_H(self):
        return 480
    
    @property
    def depth_W(self):
        return 640
    
    @property
    def rgb_H(self):
        return 968
    
    @property
    def rgb_W(self):
        return 1296
        
    @property
    def oriented_bounds(self):
        T_extent_to_scene, bounds_extents =  trimesh.bounds.oriented_bounds(self.scene_mesh)
        return T_extent_to_scene, bounds_extents

    @property
    def inv_bounds_transform(self):
        T_extent_to_scene, bounds_extents = self.oriented_bounds
        return T_extent_to_scene

    @property
    def bounds_transform(self):
        T_extent_to_scene, bounds_extents = self.oriented_bounds
        return np.linalg.inv(T_extent_to_scene)

    @property
    def bounds_extents(self):
        T_extent_to_scene, bounds_extents = self.oriented_bounds
        return bounds_extents

    @property
    def dir_camcoord(self, type='z'):
        return get_ray_direction_camcoord(1, self.depth_H, self.depth_W, self.depth_Ks[0,0], self.depth_Ks[1,1], self.depth_Ks[0,2], self.depth_Ks[1,2], depth_type='z')

    @property
    def scene_mesh(self):
        return self.get_scene_mesh()
  
    def get_scene_mesh(self):
        return trimesh.load(self.get_scene_mesh_file())

    def get_scene_mesh_file(self):
        return os.path.join(self.scene_folder, 'mesh.obj')
     
    @property
    def scene_sdf_habitat(self):
        # default scene sdf provided
        queried_sdf = self.get_habitat_sdf()
        queries = get_grid_pts(queried_sdf.shape, self.habitat_transform)
        scene_sdf_habitat = scipy.interpolate.RegularGridInterpolator(queries, queried_sdf)

        return scene_sdf_habitat

    def get_habitat_sdf(self, queried_sdf_file = '1cm/sdf.npy'):
        queried_sdf_file = os.path.join(self.scene_folder, queried_sdf_file)
        return np.abs(np.load(queried_sdf_file))
  
    def get_habitat_queries(self):
        queried_sdf = self.get_habitat_sdf()
        queries = get_grid_pts(queried_sdf.shape, self.habitat_transform)

        return queries

    @property
    def scene_sdf_stage_habitat(self):
        # default scene stage sdf provided
        # queried_stage_sdf = self.get_habitat_stage_sdf()
        # queries = get_grid_pts(queried_stage_sdf.shape, self.habitat_transform)
        # scene_stage_sdf_habitat = scipy.interpolate.RegularGridInterpolator(queries, queried_stage_sdf)

        # return scene_stage_sdf_habitat
        return None
    
    # def get_habitat_stage_sdf(self, queried_stage_sdf_file = '1cm/stage_sdf.npy'):
    #     queried_stage_sdf_file = os.path.join(self.scene_folder, queried_stage_sdf_file)
    #     return np.load(queried_stage_sdf_file)

    @property
    def scene_sdf_pysdf(self):
        # scene sdf from pymesh by provided mesh
        mesh = self.scene_mesh
        scene_sdf_pysdf = pysdf.SDF(mesh.vertices, mesh.faces)

        return scene_sdf_pysdf


    @property
    def habitat_transform(self):
        return self.get_habitat_transform()

    def get_habitat_transform(self, queries_transf_file = '1cm/transform.txt'):
        queries_transf_file = os.path.join(self.scene_folder, queries_transf_file)
        return np.loadtxt(queries_transf_file)


    @property
    def scene_min_xy(self, bounds_file = 'bounds.txt'):
        bounds_file = os.path.join(self.root_dir, bounds_file)
        return np.loadtxt(bounds_file)

    @property
    def scene_islands(self, islands_file = 'unnavigable.txt'):
        islands_file = os.path.join(self.root_dir, islands_file)
        return np.loadtxt(islands_file)

    
    @property
    def scene_root_dir(self):
        return self.root_dir

    @property
    def scene_rgb_dir(self):
        return self.rgb_dir

    @property
    def scene_depth_dir(self):
        return self.depth_dir

    @property
    def up_camera(self):
        return np.array([0., 0., 1.])

    @property
    def up_world(self):
        return np.argmax(np.abs(np.matmul(self.up_camera, self.bounds_transform[:3, :3])))

    @property
    def up_grid(self):
        return self.bounds_transform[:3, self.up_world]

    @property
    def aligned_up(self):
        return np.dot(self.up_grid, self.up_camera) > 0
    
    @property
    def scene_bounds(self):
        return self.scene_mesh.bounds