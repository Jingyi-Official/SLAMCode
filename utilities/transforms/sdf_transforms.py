'''
Description: 
Author: 
Date: 2022-09-19 21:49:23
LastEditTime: 2023-03-03 11:51:34
LastEditors: Jingyi Wan
Reference: 
'''
import torch
import skimage.measure
import trimesh
from scipy.spatial import KDTree

def sdf_render_depth(z_vals, sdf):
    """
    Basic method for rendering depth from SDF using samples along a ray.
    Assumes z_vals are ordered small -> large.
    """
    # assert (z_vals[0].sort()[1].cpu() == torch.arange(len(z_vals[0]))).all()

    n = sdf.size(1)  # n_samples
    inside = sdf < 0
    ixs = torch.arange(n, 0, -1).to(sdf.device)
    mul = inside * ixs
    max_ix = mul.argmax(dim=1)

    arange = torch.arange(z_vals.size(0))
    depths = z_vals[arange, max_ix] + sdf[arange, max_ix]

    # if no zero crossing found
    depths[max_ix == sdf.shape[1] - 1] = 0.

    # print("number of rays without zero crossing found",
    #       (depths == 0.).sum().item(), "out of",
    #       depths.numel())

    return depths

def sdf_render_mesh(sdf, scale=None, transform=None, crop_dist=torch.inf): # 0.25
    """
    Run marching cubes on sdf tensor to return mesh.
    """
    if isinstance(sdf, torch.Tensor):
        sdf = sdf.detach().cpu().numpy()
    mesh = marching_cubes_trimesh(sdf)

    # Transform to [-1, 1] range
    mesh.apply_translation([-0.5, -0.5, -0.5])
    mesh.apply_scale(2)

    # Transform to scene coordinates
    if scale is not None:
        mesh.apply_scale(scale)
    if transform is not None:
        mesh.apply_transform(transform)

    mesh.visual.face_colors = [160, 160, 160, 255]

    
    if crop_dist is not torch.inf:
        tree = KDTree(pc)
        dists, _ = tree.query(mesh.vertices, k=1)
        keep_ixs = dists < crop_dist
        face_mask = keep_ixs[mesh.faces].any(axis=1)
        mesh.update_faces(face_mask)
        mesh.remove_unreferenced_vertices()
        # sdf_mesh.visual.vertex_colors[~keep_ixs, 3] = 10
    
    
    return mesh

def marching_cubes_trimesh(numpy_3d_sdf_tensor, level=0.0):
    """
    Convert sdf samples to triangular mesh.
    """
    vertices, faces, vertex_normals, _ = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level,
    )

    dim = numpy_3d_sdf_tensor.shape[0]
    vertices = vertices / (dim - 1) # normalize vertex positions
    mesh = trimesh.Trimesh(vertices=vertices,
                           vertex_normals=vertex_normals,
                           faces=faces)

    return mesh