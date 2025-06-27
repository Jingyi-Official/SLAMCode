'''
Description: This script is to using the gt sdf for training.
Author: 
Date: 2022-11-18 14:50:49
LastEditTime: 2023-03-13 11:47:52
LastEditors: Jingyi Wan
Reference: pixelNeRF
'''
import torch
import trimesh
from utilities.tools.timing import start_timing, end_timing
from trainer.base import BaseTrainer


class GtTrainer(BaseTrainer):

    
    def training(self):
        start, end = start_timing()

        # get samples ------------------------------------------------------------------------------------------------
        samples = self.get_samples_batch()
        pc = samples["pc"]
        z_vals = samples["z_vals"]
        dir_camcoord_sample = samples["dir_camcoord_sample"]
        depth_sample = samples["depth_sample"]
        T_sample = samples["T_sample"]
        normal_sample = samples["normal_sample"]
        indices_b = samples["indices_b"]
        indices_h = samples["indices_h"]
        indices_w = samples["indices_w"]
        idxs = samples["idxs"]

        # get prediction --------------------------------------------------------------------------------------------
        do_sdf_grad = self.eik_weight != 0 or self.grad_weight != 0
        if do_sdf_grad:
            pc.requires_grad_()

        sdf, sdf_grad = self.model(pc, noise_std=self.noise_std, do_grad=do_sdf_grad)

        # get esmated label / compute bounds ---------------------------------------------------------------------------------------------
        bounds, grad_vec = self.get_bounds(
            self.bounds_method, 
            dir_camcoord_sample, 
            depth_sample, 
            T_sample, 
            z_vals, 
            pc, 
            self.trunc_distance, 
            normal_sample, 
            do_grad=True, 
        )

        iso_start, iso_end = start_timing()
        sdf_gt, valid_mask = self.evaluator.get_sdf_gt(pc.cpu().detach().numpy(), self.evaluator.scene_sdf_kit, handle_oob='mask')
        bounds = torch.from_numpy(sdf_gt).type_as(bounds)#.to(self.device).type(torch.DoubleTensor)
        iso_time = end_timing(iso_start, iso_end)

        # calculate loss ----------------------------------------------------------------------------------------------
        # sdf loss
        sdf_loss_mat, free_space_ixs = self.sdf_loss(sdf, bounds, self.trunc_distance, loss_type="L1") # torch.Size([181, 27])

        # eik loss
        eik_loss_mat = None
        if self.eik_weight != 0:
            eik_loss_mat = self.eik_loss(sdf_grad)

        # grad loss
        grad_loss_mat = None
        if self.grad_weight != 0:
            grad_loss_mat = self.grad_loss(sdf_grad, normal_sample, grad_vec, do_orien_loss = False)

        total_loss, total_loss_mat, losses = self.tot_loss(
            sdf_loss_mat, grad_loss_mat, eik_loss_mat,
            free_space_ixs, bounds, self.eik_apply_dist,
            self.trunc_weight, self.grad_weight, self.eik_weight,
        )

        self.frames.frame_loss_approxes[idxs], self.frames.frame_avg_losses[idxs] = self.frame_avg_loss(total_loss_mat, indices_b, indices_h, indices_w, len(idxs), self.depth_H, self.depth_W, self.loss_approx_factor, self.device, mode = 'block') 

        total_loss.backward()
        self.optimiser.step()
        for param_group in self.optimiser.param_groups:
            params = param_group["params"]
            for param in params:
                param.grad = None

        step_time = end_timing(start, end) - iso_time
        time_s = step_time / 1000.
        self.tot_step_time += 1 * time_s
        
        print(f"Mean: {sdf.mean()}; Std: {sdf.std()}")
        print(f"Max: {sdf.max()}; Min: {sdf.min()}")

        return losses, step_time

                