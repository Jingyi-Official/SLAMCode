'''
Description: 
Author: 
Date: 2022-09-19 21:49:24
LastEditTime: 2023-02-09 18:54:32
LastEditors: Jingyi Wan
Reference: 
'''
import os
import numpy


def get_depth_intrinsic_scene(input_file):
    output_file = os.path.join(os.path.dirname(input_file), 'depth_K_file.txt')
    info = {}
    with open(input_file, 'r') as f:
        for line in f.read().splitlines():
            split = line.split(' = ')
            info[split[0]] = split[1]
        fx = float(info['fx_depth'])
        fy = float(info['fy_depth'])
        cx = float(info['mx_depth'])
        cy = float(info['my_depth'])
        H = int(info['depthHeight'])
        W = int(info['depthWidth'])

    intrinsic_data = numpy.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    numpy.savetxt(output_file, intrinsic_data)
    
def get_rgb_intrinsic(input_file):
    output_file = os.path.join(os.path.dirname(os.path.dirname(input_file)), 'rgb_K_file.txt')
    intrinsic_data = numpy.loadtxt(input_file)[:3,:3]
    numpy.savetxt(output_file, intrinsic_data)
    

if __name__ == "__main__":
    input_file = '/media/wanjingyi/Diskroom/iSDF/data/seqs/scene0031_00/scene0031_00.txt'
    get_depth_intrinsic_scene(input_file)

    input_file = '/media/wanjingyi/Diskroom/iSDF/data/seqs/scene0031_00/intrinsic/intrinsic_color.txt'
    get_rgb_intrinsic(input_file)


    
