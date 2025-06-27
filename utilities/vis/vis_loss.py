'''
Description: 
Author: 
Date: 2023-02-13 16:53:20
LastEditTime: 2023-02-13 19:42:13
LastEditors: Jingyi Wan
Reference: 
'''
import torch
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import open3d.core as o3c

def vis_loss(loss, model, evaluator):
    
    pts = evaluator.eval_mesh_grid
    
    # get prediction for the mesh grid
    with torch.set_grad_enabled(False):
        sdf_pred = evaluator.get_sdf_pred_chunks(
            pc = pts,
            fc_sdf_map = model
        )

    # get gt for the mesh grid
    sdf_gt, valid_mask = evaluator.get_sdf_gt(evaluator.eval_mesh_grid.cpu().detach().numpy(), evaluator.scene_sdf_kit, handle_oob='mask')
    valid_mask = np.logical_and(sdf_gt != 0., valid_mask)

    # filter invalid
    pts = pts[valid_mask]
    sdf_gt = sdf_gt[valid_mask]
    sdf_pred = sdf_pred[valid_mask]
    sdf_pred = sdf_pred.cpu().detach().numpy()

    # get sign difference
    # mul = sdf_gt*sdf_pred
    # filter_mask = mul<0
    
    # pts = pts[filter_mask]
    # sdf_gt = sdf_gt[filter_mask]
    # sdf_pred = sdf_pred[filter_mask]
    
    # # visualize
    gt_mesh = o3d.io.read_triangle_mesh(evaluator.scene_dataset.get_scene_mesh_file())
    
    # gui.Application.instance.initialize()	
    # window = gui.Application.instance.create_window('My First Window', 800, 600)
    
    # # 创建显示场景
    # scene = gui.SceneWidget()
    # scene.scene = rendering.Open3DScene(window.renderer)
    # window.add_child(scene)
    
    # # 添加物体
    # material = rendering.MaterialRecord()
    # material.shader = 'defaultLit'
    # scene.scene.add_geometry("gt_mesh", gt_mesh, material)
    # pcd = o3d.t.geometry.PointCloud(o3c.Tensor(pts, dtype=np.float32))
    # pcd.point['colors'] = o3c.Tensor(np.zeros(pts.shape, dtype=np.float32))
    # scene.scene.add_geometry('points', pcd, material)


    
    # # 设置相机属性
    # bounds = gt_mesh.get_axis_aligned_bounding_box()
    # scene.setup_camera(60, bounds, bounds.get_center())

    # gui.Application.instance.run()

