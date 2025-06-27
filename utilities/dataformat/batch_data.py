import torch

def get_selected_frames_batch(depth_batch, T_batch, normal_batch, idxs, filter_invalid = False):
    if not filter_invalid:
        depth_batch_selected = depth_batch[idxs]
        T_batch_selected = T_batch[idxs]
        normal_batch_selected = normal_batch[idxs] if normal_batch is not None else None
        # normal_batch_selected = normal_batch if normal_batch is not None else None

    return depth_batch_selected, T_batch_selected, normal_batch_selected


def get_selected_pixels_batch(depth_batch_selected=None, T_batch_selected=None, normal_batch_selected=None, dir_camcoord=None, indices_b=None, indices_h=None, indices_w=None, filter_invalid=True):
    if filter_invalid:
        depth_sample = depth_batch_selected[indices_b, indices_h, indices_w].view(-1)
        # filter invalid depth
        mask_valid_depth = depth_sample != 0

        # if normal is used, we also need to filter invalid normal
        if normal_batch_selected is not None:
            normal_sample = normal_batch_selected[indices_b,indices_h,indices_w, :].view(-1, 3)
            mask_invalid_normal = torch.isnan(normal_sample[..., 0])
            mask_valid_depth = torch.logical_and(mask_valid_depth, ~mask_invalid_normal)
            normal_sample = normal_sample[mask_valid_depth]
        else:
            normal_sample = None

        depth_sample = depth_sample[mask_valid_depth]
        indices_b = indices_b[mask_valid_depth]
        indices_h = indices_h[mask_valid_depth]
        indices_w = indices_w[mask_valid_depth]


    T_sample = T_batch_selected[indices_b]
    dir_camcoord_sample = dir_camcoord[0, indices_h, indices_w, :].view(-1, 3)

    return depth_sample, T_sample, normal_sample, dir_camcoord_sample, [indices_b, indices_h, indices_w]
    