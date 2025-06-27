import trimesh
import numpy as np
from scipy.spatial import cKDTree as KDTree

def accuracy_comp(mesh_gt, mesh_rec, samples=200000):
    rec_pc = trimesh.sample.sample_surface(mesh_rec, samples)
    rec_pc_tri = trimesh.PointCloud(vertices=rec_pc[0])

    gt_pc = trimesh.sample.sample_surface(mesh_gt, samples)
    gt_pc_tri = trimesh.PointCloud(vertices=gt_pc[0])

    acc = accuracy(gt_pc_tri.vertices, rec_pc_tri.vertices)
    comp = completion(gt_pc_tri.vertices, rec_pc_tri.vertices)

    return acc, comp

def accuracy(gt_points, rec_points):
    gt_points_kd_tree = KDTree(gt_points)
    distances, _ = gt_points_kd_tree.query(rec_points)
    acc = np.mean(distances)
    return acc


def completion(gt_points, rec_points):
    rec_points_kd_tree = KDTree(rec_points)
    distances, _ = rec_points_kd_tree.query(gt_points)
    comp = np.mean(distances)
    return comp