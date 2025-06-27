'''
Description: 
Author: 
Date: 2022-06-17 14:39:58
LastEditTime: 2023-01-26 16:55:25
LastEditors: Jingyi Wan
Reference: 
'''
import torch
import torch.nn.functional as F


def sample_pixels(n_rays, n_frames, H, W, loss = None, do_sample_active = False, device = None):
    
    if do_sample_active:
        indices_b, indices_h, indices_w = active_sample_pixels(n_rays, n_frames, H, W, loss, device)
    else:
        indices_b, indices_h, indices_w = random_sample_pixels(n_rays, n_frames, H, W, device)
    
    return indices_b, indices_h, indices_w  


'''
description: uniformly sample pixels in the frames

param {n_rays} : how many rays to sample for each frame
param {n_frames}: how many frames in the trainers.frame for the training
param {h}: height/rows of one frame
param {w}: width/columns of one fram
param {device} : cuda or cpu

return {*}
'''
def random_sample_pixels(n_rays, n_frames, h, w, device):
    total_rays = n_rays * n_frames
    indices_h = torch.randint(0, h, (total_rays,), device=device) # a tensor uniform between [0,h)
    indices_w = torch.randint(0, w, (total_rays,), device=device)

    indices_b = torch.arange(n_frames, device=device)
    indices_b = indices_b.repeat_interleave(n_rays)

    return indices_b, indices_h, indices_w



def active_sample_pixels(n_rays, n_frames, H, W, loss, device):
    total_rays = n_rays * n_frames
    # prevent nans or 0
    loss = loss + 0.00001 # prevent nans

    # here treat each frame equally
    pdf= F.normalize(loss, p=1, dim=(1, 2))

    H_repeat = int(H/loss.shape[1])
    H_pdf = pdf.sum(dim=1)
    H_pdf = (H_pdf/H_repeat).repeat_interleave(H_repeat)
    indices_h = torch.multinomial(H_pdf, total_rays) % H

    W_repeat = int(W/loss.shape[2])
    W_pdf = pdf.sum(dim=2)
    W_pdf = (W_pdf/W_repeat).repeat_interleave(W_repeat)
    indices_w = torch.multinomial(W_pdf, total_rays) % W

    indices_b = torch.arange(n_frames, device=device)
    indices_b = indices_b.repeat_interleave(n_rays)

    return indices_b, indices_h, indices_w




