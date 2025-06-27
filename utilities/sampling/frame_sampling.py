import numpy as np

def sample_frames(n_window, n_frames, n_limit, losses, do_frame_active=True):
    if do_frame_active:
        idxs = active_sample_frames(n_window, n_frames, n_limit, losses)
        
    else:
        print('For now, the method is required to run by frame active sampling. do_frame_active have to be TRUE. ')


    return idxs

def active_sample_frames(n_window, n_frames, n_limit, losses):
    idxs = active_sample_frames_by_loss(n_window, n_frames, n_limit, losses)

    return idxs


def active_sample_frames_by_loss(n_window, n_frames, n_limit, losses):
    options = n_frames - n_limit
    select_size = n_window - n_limit
    
    denom = losses[:-n_limit].sum()
    loss_dist = losses[:-n_limit]/denom
    loss_dist_np = loss_dist.cpu().numpy()

    rand_ints = np.random.choice(
        np.arange(0, options),
        size=select_size,
        replace=False,
        p=loss_dist_np)

    last = n_frames - 1
    idxs = [*rand_ints, last - 1, last]

    return idxs