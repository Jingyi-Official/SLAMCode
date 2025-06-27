import numpy as np
import torch
import trimesh

'''
frame_id: index for frame
rgb_batch: nomalized rgb /255 torch.tensor
rgb_batch_np: un-nomalized rgb, np
depth_batch: torch.tensor,
depth_batch_np: np,
T_WC_batch=torch.tensor,
T_WC_batch_np=np,
normal_batch=None,
frame_avg_losses=None,
T_WC_track=None,
T_WC_gt=None,
'''
class FrameData:
    def __init__(
        self, 
        frame_id=None,
        rgb_batch=None,
        rgb_batch_np=None,
        depth_batch=None,
        depth_batch_np=None,
        T_batch=None,
        T_batch_np=None,
        normal_batch=None,
        frame_avg_losses=None,
        frame_loss_approxes = None,
        sample_avg_losses=None,
    ):
        super(FrameData, self).__init__()

        self.frame_id = frame_id
        self.rgb_batch = rgb_batch
        self.rgb_batch_np = rgb_batch_np
        self.depth_batch = depth_batch
        self.depth_batch_np = depth_batch_np
        self.T_batch = T_batch
        self.T_batch_np = T_batch_np

        self.normal_batch = normal_batch
        self.frame_avg_losses = frame_avg_losses
        self.frame_loss_approxes = frame_loss_approxes
        self.sample_avg_losses = sample_avg_losses

        self.count = 0 if frame_id is None else len(frame_id)

    def add_frame_data(self, data, replace):
        """
        Add new FrameData to existing FrameData.
        """
        self.frame_id = expand_data(
            self.frame_id, data.frame_id, replace)

        self.rgb_batch = expand_data(
            self.rgb_batch, data.rgb_batch, replace)
        self.rgb_batch_np = expand_data(
            self.rgb_batch_np, data.rgb_batch_np, replace)

        self.depth_batch = expand_data(
            self.depth_batch, data.depth_batch, replace)
        self.depth_batch_np = expand_data(
            self.depth_batch_np, data.depth_batch_np, replace)

        self.T_batch = expand_data(
            self.T_batch, data.T_batch, replace)
        self.T_batch_np = expand_data(
            self.T_batch_np, data.T_batch_np, replace)

        self.normal_batch = expand_data(
            self.normal_batch, data.normal_batch, replace)

        self.frame_avg_losses = expand_data(
            self.frame_avg_losses, data.frame_avg_losses, replace)

        self.frame_loss_approxes = expand_data(
            self.frame_loss_approxes, data.frame_loss_approxes, replace)

        self.sample_avg_losses = expand_data(
            self.sample_avg_losses, data.sample_avg_losses, replace)

            


    def __len__(self):
        return 0 if self.frame_id is None else len(self.frame_id)


def expand_data(batch, data, replace=False):
    """
    Add new FrameData attribute to exisiting FrameData attribute.
    Either concatenate or replace last row in batch.
    """
    cat_fn = np.concatenate
    if torch.is_tensor(data):
        cat_fn = torch.cat

    if batch is None:
        batch = data

    else:
        if replace is False:
            batch = cat_fn((batch, data))
        else:
            batch[-1] = data[0]

    return batch



def save_trajectory(traj, file_name, format="replica", timestamps=None):
    traj_file = open(file_name, "w")

    if format == "replica":
        for idx, T_WC in enumerate(traj):
            time = timestamps[idx]
            traj_file.write('{} '.format(time))
            np.savetxt(traj_file, T_WC[:3, :].reshape([1, 12]), fmt="%f")
    elif format == "TUM":
        for idx, T_WC in enumerate(traj):
            quat = trimesh.transformations.quaternion_from_matrix(T_WC[:3, :3])
            quat = np.roll(quat, -1)
            trans = T_WC[:3, 3]
            time = timestamps[idx]

            traj_file.write('{} '.format(time))
            np.savetxt(traj_file, trans.reshape([1, 3]), fmt="%f", newline=" ")
            np.savetxt(traj_file, quat.reshape([1, 4]), fmt="%f",)

    traj_file.close()
