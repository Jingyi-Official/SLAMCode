'''
Description: 
Author: 
Date: 2022-06-23 13:59:21
LastEditTime: 2022-07-03 19:02:27
LastEditors: Jingyi Wan
Reference: 
'''
import open3d as o3d

def vis_pc(points):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pc])

def vis_pc_save(points,savepath):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pc])
    o3d.io.write_point_cloud(savepath, pc)
    print('finshi')

def vis_pc_and_norm_custom(points, norms):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.normals = o3d.utility.Vector3dVector(norms)
    o3d.visualization.draw_geometries([pc],"vis_pc_and_norm_custom", point_show_normal=True, mesh_show_wireframe=True,mesh_show_back_face=True)


def vis_pc_and_norm_knn(points, n_knn=20):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=n_knn))
    print(pc.normals)
    o3d.visualization.draw_geometries([pc], f"vis_pc_and_norm_knn with {n_knn}", point_show_normal=True, mesh_show_wireframe=True,mesh_show_back_face=True)


def vis_pc_and_norm_radius(points, r=0.01):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.estimate_normals(search_param=KDTreeSearchParamRadius(radius=r))
    print(pc.normals)
    o3d.visualization.draw_geometries([pc], f"vis_pc_and_norm_radius with {r}", point_show_normal=True, mesh_show_wireframe=True,mesh_show_back_face=True)


def vis_pc_and_norm_hybrid(points, r=0.01, n_knn=20):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.estimate_normals(search_param=KDTreeSearchParamHybrid(radius=r, max_nn=n_knn))
    print(pc.normals)
    o3d.visualization.draw_geometries([pc], f"vis_pc_and_norm with r_{r}, knn_{n_knn}", point_show_normal=True, mesh_show_wireframe=True,mesh_show_back_face=True)


def vis_2_pc(pc1, pc2):
    pt1=o3d.geometry.PointCloud()
    pt1.points=o3d.utility.Vector3dVector(pc1)
    pt1.paint_uniform_color([0.7, 0.7, 0.7])


    pt2=o3d.geometry.PointCloud()
    pt2.points=o3d.utility.Vector3dVector(pc2)
    pt2.paint_uniform_color([0,0,1])
    
    o3d.visualization.draw_geometries([pt1,pt2],window_name='vis sampled points')

def vis_whole_sampled_point_bounds(pc1,pc2,points,closest):
    pt1=o3d.geometry.PointCloud()
    pt1.points=o3d.utility.Vector3dVector(pc1)
    pt1.paint_uniform_color([0.7, 0.7, 0.7])


    pt2=o3d.geometry.PointCloud()
    pt2.points=o3d.utility.Vector3dVector(pc2)
    pt2.paint_uniform_color([0,0,1])

    p=o3d.geometry.PointCloud()
    p.points=o3d.utility.Vector3dVector(points)
    p.paint_uniform_color([1,0,0])


    p2=o3d.geometry.PointCloud()
    p2.points=o3d.utility.Vector3dVector(closest)
    p2.paint_uniform_color([0,1,0])

    import torch
    import numpy as np
    p_list = torch.cat((points, closest))
    start = np.arange(points.shape[0])
    end = np.arange(points.shape[0]) + points.shape[0]
    lines_box = np.concatenate([[start,end]],axis=1)
    lines_box = lines_box.T

    line_set = o3d.geometry.LineSet()
    line_set.lines = o3d.utility.Vector2iVector(lines_box)
    colors = np.array([[0, 1, 0] for j in range(len(lines_box))])
    line_set.colors = o3d.utility.Vector3dVector(colors)
    line_set.points = o3d.utility.Vector3dVector(p_list)

    # line_set = o3d.geometry.LineSet(
    #     points = o3d.utility.Vector3dVector(p_list),
    #     lines = o3d.utility.Vector2iVector(lines_box)
    # )
    colors = np.array([[1, 1, 0] for j in range(len(lines_box))])
    line_set.colors = o3d.utility.Vector3dVector(colors)
    #将矩形框加入到窗口中
    # vis.add_geometry(line_set) 

    o3d.visualization.draw_geometries([pt1,pt2,p,p2,line_set],window_name='vis sampled points')


def vis_compare(pc,source, target):
    pt1=o3d.geometry.PointCloud()
    pt1.points=o3d.utility.Vector3dVector(pc.reshape(-1,3))
    pt1.paint_uniform_color([0.7, 0.7, 0.7])


    lines = target - source
    pt2=o3d.geometry.PointCloud()
    pt2.points=o3d.utility.Vector3dVector(source.reshape(-1,3))
    pt2.normals = o3d.utility.Vector3dVector(lines)
    pt2.paint_uniform_color([0, 0.651, 0.929])


    pt3=o3d.geometry.PointCloud()
    pt3.points=o3d.utility.Vector3dVector(target.reshape(-1,3))
    pt3.paint_uniform_color([0,0,1])
    

    o3d.visualization.draw_geometries([pt1,pt2,pt3],window_name='normal vis',width=800,height=600)

def vis_compare_2(pc,source, target1, target2):
    pt1=o3d.geometry.PointCloud()
    pt1.points=o3d.utility.Vector3dVector(pc.reshape(-1,3))
    pt1.paint_uniform_color([0.7, 0.7, 0.7])


    pt2=o3d.geometry.PointCloud()
    pt2.points=o3d.utility.Vector3dVector(target1.reshape(-1,3))
    pt2.paint_uniform_color([0, 0.651, 0.929])



    pt3=o3d.geometry.PointCloud()
    pt3.points=o3d.utility.Vector3dVector(target2.reshape(-1,3))
    pt3.paint_uniform_color([0,0,1])
    

    o3d.visualization.draw_geometries([pt1,pt2,pt3],window_name='normal vis',width=800,height=600)


if __name__ == '__main__':
    filename = '/media/wanjingyi/Diskroom/isdf_data/03_out/sequences/03/velodyne/000800.bin'
    import numpy as np
    scan = np.fromfile(filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    points = scan[:, 0:3]    # get xyz
    vis_pc_save(points,'/media/wanjingyi/Diskroom/isdf_data/03_out/sequences/03/velodyne/000800.ply')