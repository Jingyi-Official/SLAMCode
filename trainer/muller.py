'''
Author: This is the scripts to literally try the muller methods all setting.
Date: 2023-03-23 13:38:08
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-03-28 19:47:31
FilePath: /code_template/trainer/muller.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import torch
from torchinfo import summary
from trainer.base import BaseTrainer
from utilities.tools.timing import start_timing, end_timing
from utilities.sampling.frame_sampling import sample_frames
from utilities.sampling.ray_sampling import sample_points
from utilities.sampling.pixel_sampling import sample_pixels, random_sample_pixels
from utilities.dataformat.batch_data import get_selected_frames_batch, get_selected_pixels_batch

import numpy as np

n_points = 100000

class UniformTrainer(BaseTrainer):    
    
    # pixel coord to camera coord
    def sample_tedrahedron(self): #get the uniform pc_c，pixel coord to camera coord
        max_depth = 20
        W = self.depth_W - 1  
        H = self.depth_H - 1

        u = (torch.rand(n_points,1) - 0.5)  # right or not?
        v = torch.rand(n_points,1) - 0.5
        w = torch.rand(n_points,1)**(1/3.)

        x = u*w+0.5
        y = v*w+0.5
        z = w

        x = (x*W - self.scene_dataset.Ks[0, 2])*max_depth/self.scene_dataset.Ks[0, 0]
        y = (y*H - self.scene_dataset.Ks[1, 2])*max_depth/self.scene_dataset.Ks[1, 1]
        z = z * max_depth

        return torch.cat([x,y,z,torch.ones(x.shape, device=x.device)],dim=-1)
    
    def fit(self, dataset): 
        self.log.info("Loading dataset information.")
        self.scene_dataset = dataset
        self.scene_dataset_size = len(dataset)
        self.depth_H = self.scene_dataset.depth_H
        self.depth_W = self.scene_dataset.depth_W
        self.dir_camcoord = self.scene_dataset.dir_camcoord.to(self.device)
        self.scene_rgb_dir = self.scene_dataset.scene_rgb_dir
        self.scene_depth_dir = self.scene_dataset.scene_depth_dir
        self.dir_camcoord = self.scene_dataset.dir_camcoord.to(self.device)
        self.log.info("Finish loading dataset information.")

        self.log.info("Adapt model setting according to data.")
        self.model.transform_input = torch.from_numpy(self.scene_dataset.inv_bounds_transform).float().to(self.device) # 将mesh.obj转换到原点为中所需的transform
        summary(self.model, input_size=(1000, 27, 3))
        self.log.info("Finish adapting model setting.")

        
        self.log.info("Set standard tedrahedron.")
        self.pc_c = self.sample_tedrahedron().to(self.device)
        self.log.info("Finish setting standard tedrahedron.")
        
        
        self.log.info(f"Starting training for max {self.max_steps} steps...")

        for t in range(self.max_steps):

            if t==0:
                self.log.info("Set frame 0 as keyframe by default.")
                
                new_frame_id = self.get_current_frame_id()
                frame_data = self.get_frame(new_frame_id)
                self.add_frame(frame_data)
                self.log.info(f"Point to new frame --> {new_frame_id}.")

                self.steps_since_frame = 0
                self.last_is_keyframe = True
                self.optim_frames = self.iters_start

                self.noise_std = self.noise_frame

            elif self.steps_since_frame == self.optim_frames and self.last_is_keyframe:
                self.log.info("Last is keyframe, thus add new frame.")

                self.copy_model()
                new_frame_id = self.get_current_frame_id()
                if new_frame_id >= self.scene_dataset_size:
                    self.log.info("Out of sequence.")
                    break
                frame_data = self.get_frame(new_frame_id)
                self.add_frame(frame_data)
                self.log.info(f"Point to new frame --> {new_frame_id}.")

                self.steps_since_frame = 0
                self.last_is_keyframe = False
                self.optim_frames = self.iters_per_frame
                
                self.noise_std = self.noise_frame

            elif self.steps_since_frame == self.optim_frames and not self.last_is_keyframe:
                self.log.info("Not sure whether last is keyframe. Thus to check.")
                
                self.last_is_keyframe = self.check_last_is_keyframe()
                self.log.info(f"Last is keyframe: {self.last_is_keyframe} ")
                
                if self.last_is_keyframe:
                    self.optim_frames = self.iters_per_kf
                    self.noise_std = self.noise_kf
                    
                    # record keyframe info
                    # self.logger.log_image(key="Keyframes", images=[os.path.join(self.scene_dataset.scene_rgb_dir, f"frame%06d.png"%(self.frames.frame_id[-1]))])
                    # self.logger.log_image(key="Keyframes", images=[os.path.join(self.scene_dataset.scene_rgb_dir, f"%d.jpg"%(self.frames.frame_id[-1]))])

                else:
                    new_frame_id = self.get_current_frame_id()
                    if new_frame_id >= self.scene_dataset_size:
                        self.log.info("Out of sequence.")
                        break
                    frame_data = self.get_frame(new_frame_id)
                    self.add_frame(frame_data)
                    self.log.info(f"Point to new frame --> {new_frame_id}.")

                    self.steps_since_frame = 0
                    self.last_is_keyframe = False
                    self.optim_frames = self.iters_per_frame
                    
                    self.noise_std = self.noise_frame


            # optimisation step---------------------------------------------
            losses, step_time = self.training()
            self.steps_since_frame += 1

            # logging during training
            status = " ".join([k + ': {:.6f}  '.format(losses[k]) for k in losses.keys()])
            self.log.info(f"Step: {t} Iteration: {self.steps_since_frame} {status} step_time: {step_time}")
            self.logger.log_metrics(metrics=losses,step=t)
            self.logger.log_metrics(metrics={"step_time": step_time},step=t)

            # evaluation step-----------------------------------------------
            if self.eval_freq_s > 0:
                elapsed_eval = self.tot_step_time - self.last_eval
                if t > self.iters_start and elapsed_eval > self.eval_freq_s:
                    self.last_eval = self.tot_step_time - self.tot_step_time % self.eval_freq_s

                    self.log.info(f'After {elapsed_eval}s, evaluating...')
                    metrics = self.evaluating()

                    self.log.info(f"Cache Evaluation: cache_l1_error_avg: {metrics['cache_l1_error_avg']} cache_coll_cost_error: {metrics['cache_coll_cost_error']}")
                    self.log.info(f"Volume Evaluation: volume_l1_error_avg: {metrics['volume_l1_error_avg']} volume_coll_cost_error: {metrics['volume_coll_cost_error']}")
                    # self.log.info(f"Scene Evaluation: scene_l1_error_avg: {metrics['scene_l1_error_avg']} scene_coll_cost_error: {metrics['scene_coll_cost_error']}")
                    self.log.info(f"Mesh Evaluation: mesh_accuracy: {metrics['mesh_accuracy']} mesh_completion: {metrics['mesh_completion']}")
                    self.logger.log_metrics(metrics=metrics,step=t)

        
        # final evaluation
        self.log.info(f"Finish training. Final evaluation:")
        metrics = self.evaluating(save=True)
        self.log.info(f"Cache Evaluation: cache_l1_error_avg: {metrics['cache_l1_error_avg']} cache_coll_cost_error: {metrics['cache_coll_cost_error']}")
        self.log.info(f"Volume Evaluation: volume_l1_error_avg: {metrics['volume_l1_error_avg']} volume_coll_cost_error: {metrics['volume_coll_cost_error']}")
        # self.log.info(f"Scene Evaluation: scene_l1_error_avg: {metrics['scene_l1_error_avg']} scene_coll_cost_error: {metrics['scene_coll_cost_error']}")
        self.log.info(f"Mesh Evaluation: mesh_accuracy: {metrics['mesh_accuracy']} mesh_completion: {metrics['mesh_completion']}")
        self.logger.log_metrics(metrics=metrics,step=t)
   
    def training(self):
        start, end = start_timing()

        # get samples
        samples = self.get_samples_batch()
        idxs = samples["idxs"]
        pc = samples["pc"] # torch.Size([8793, 3])
        z_vals = samples["z_vals"] # torch.Size([8793])
        depth_sample = samples["depth_sample"] # 
        indices_b = samples["indices_b"] # torch.Size([8793])
        indices_h = samples["indices_h"] # torch.Size([8793])
        indices_w = samples["indices_w"] # torch.Size([8793])
        # T_sample = samples["T_sample"] # torch.Size([8793, 4, 4])
        surface_pc = samples["surface_pc"] # torch.Size([183, 3])
        # surface_depth_sample = samples["surface_depth_sample"] # torch.Size([183])
        # normal_sample = samples["surface_normal_sample"]
        # surface_indices_b = samples["surface_indices_b"] # torch.Size([183])
        # surface_indices_h = samples["surface_indices_h"] # torch.Size([183])
        # surface_indices_w = samples["surface_indices_w"] # torch.Size([183])
        # surface_T_sample = samples["surface_T_sample"] # torch.Size([183, 4, 4])
        normal_sample = None
        with torch.set_grad_enabled(False): # could be write as kdtree to speed up the process
            diff = pc[:, None] - surface_pc  # torch.Size([8793, 183, 3])
            dists = diff.norm(dim=-1) # torch.Size([8793, 183])
            dists, closest_ixs = dists.min(axis=-1) # torch.Size([8793])
            # closest_vects = diff[torch.arange(diff.shape[0]), closest_ixs]
            # normal_vects = normal_sample[closest_ixs]
            # behind_surf = torch.bmm(closest_vects.unsqueeze(1),normal_vects.unsqueeze(-1)).squeeze() > 0 # hadmard product torch.mul(closest_vects,normal_vects).sum(dim=-1)
            depth_diff = z_vals - depth_sample
            behind_surf = depth_diff > 0
            dists[behind_surf] *= -1
            bounds = dists
            grad_vec = None
            # grad_vec = closest_vects

        # filter again based on the bounds
        # mask = torch.logical_and(bounds>=-0.2,bounds<=20)
        mask = torch.logical_and(depth_diff < 0.2,z_vals > 0.07)
        pc = pc[mask]
        bounds = bounds[mask]
        indices_b = indices_b[mask]
        indices_h = indices_h[mask]
        indices_w = indices_w[mask]

        # new_start, new_end = start_timing()
        # # get gt
        # sdf_gt, valid_mask = self.evaluator.get_sdf_gt(pc.cpu().detach().numpy(), self.evaluator.scene_sdf_kit, handle_oob='mask')
        # bounds = torch.from_numpy(sdf_gt).type_as(bounds)#.to(self.device).type(torch.DoubleTensor)

        # # filter again based on the bounds
        # mask = torch.logical_and(bounds>=-0.2,bounds<=20)
        # pc = pc[mask]
        # bounds = bounds[mask]
        # indices_b = indices_b[mask]
        # indices_h = indices_h[mask]
        # indices_w = indices_w[mask]

        # new_time = end_timing(new_start, new_end)

        # get prediction --------------------------------------------------------------------------------------------
        do_sdf_grad = self.eik_weight != 0 or self.grad_weight != 0
        if do_sdf_grad:
            pc.requires_grad_()

        sdf, sdf_grad = self.model(pc, noise_std=self.noise_std, do_grad=do_sdf_grad) # torch.Size([181, 27])

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

        step_time = end_timing(start, end)
        time_s = step_time / 1000.
        self.tot_step_time += 1 * time_s 

        '''
        visualize to see the reliability
        '''
        np.save('/media/wanjingyi/Diskroom/code_template/test/test_outputs/samples.npy', pc.detach().cpu())

        return losses, step_time
    
    
    def get_samples_batch(self):
        # get all the key frames
        depth_batch = self.frames.depth_batch
        T_batch = self.frames.T_batch
        normal_batch = self.frames.normal_batch if self.do_normal else None
        n_frames = len(self.frames)
        # assert n_frames == depth_batch.shape[0] == T_batch.shape[0] == normal_batch.shape[0]
        
        # memory limit thus have to sample frames -------------------------------------------------------
        if n_frames > self.window_size:
            idxs = sample_frames(self.window_size, n_frames, self.window_limit_size, self.frames.frame_avg_losses, self.do_frame_active)
        else:
            idxs = np.arange(n_frames)

        # get chosen valid frames
        depth_batch_selected, T_batch_selected, normal_batch_selected = get_selected_frames_batch(depth_batch, T_batch, normal_batch, idxs)
        
        # get unformly samples
        pc, z_vals, depth_sample, indices_b, indices_h, indices_w, T_sample, \
            surface_pc, surface_depth_sample, surface_indices_b, surface_indices_h, surface_indices_w, surface_T_sample, surface_normal_sample = self.sample_points(depth_batch_selected, T_batch_selected, normal_batch_selected)

        samples = {
            "idxs": idxs,
            "pc": pc,
            "z_vals": z_vals,
            "depth_sample": depth_sample,
            "indices_b": indices_b,
            "indices_h": indices_h,
            "indices_w": indices_w,
            # "T_sample": T_sample,
            "surface_pc": surface_pc,
            "surface_depth_sample": surface_depth_sample,
            # "surface_indices_b": surface_indices_b,
            # "surface_indices_h": surface_indices_h,
            # "surface_indices_w": surface_indices_w,
            # "surface_T_sample": surface_T_sample,
            # "surface_normal_sample": surface_normal_sample,
        }

        return samples

    def sample_points(self, depth_batch_selected, T_batch_selected, normal_batch_selected, include_surface=True):

        # -------------------- sample uniformly on space -------------------- 
        
        # filter the depth / filter the pc that's too far away from the camera/surface
        max_depth_selected = torch.amax(depth_batch_selected,dim=(1,2)) #[N_frames]
        mask = self.pc_c[:,2].unsqueeze(1) <= (max_depth_selected+0.1) #[N_self_pc_c, N_frames]
        # assert mask[:,0].equal(self.pc_c[:,2]<=max_depth_selected[0])
        
        # get filtered pc in camera coord
        masked_pc_c = self.pc_c.expand(mask.shape[-1], *self.pc_c.shape)[mask.T] #[N_pc_c, 4]
        
        # get corresponding frame idx
        indices_b = torch.arange(depth_batch_selected.shape[0], device=depth_batch_selected.device).repeat_interleave(mask.sum(dim=0)) #[N_pc_c]
        # get corresponding pose and translate to the camera view in world coord
        masked_T_selected = T_batch_selected[indices_b]
        masked_pc_w = torch.bmm(masked_T_selected, masked_pc_c[:,:, None]).squeeze()[:,:3]

        # from the world coord to the image coord/ get the indices and the z_vals ?? round vs floor
        indices_w = torch.round((masked_pc_c[:, 0] * self.scene_dataset.Ks[0, 0] / masked_pc_c[:, 2]) + self.scene_dataset.Ks[0, 2]).long()  # pix_x = [nx*ny*nz]
        indices_h = torch.round((masked_pc_c[:, 1] * self.scene_dataset.Ks[1, 1] / masked_pc_c[:, 2]) + self.scene_dataset.Ks[1, 2]).long()  # pix_y = [nx*ny*nz]
        z_vals = masked_pc_c[:, 2] # pix_z

        # get corresponding depth point
        depth_sample = depth_batch_selected[indices_b, indices_h, indices_w]


        # -------------------- sample randomly on surface --------------------

        surface_indices_b, surface_indices_h, surface_indices_w = sample_pixels(self.n_rays_train, depth_batch_selected.shape[0], self.depth_H, self.depth_W, self.frames.frame_loss_approxes, self.do_sample_active, device=self.device)
        surface_depth_sample, surface_T_sample, surface_normal_sample, dir_camcoord_sample, [surface_indices_b, surface_indices_h, surface_indices_w] = get_selected_pixels_batch(depth_batch_selected, T_batch_selected, normal_batch_selected, self.dir_camcoord, surface_indices_b, surface_indices_h, surface_indices_w)

        # memory limit thus have to sample points --------------------------------------------------------
        surface_pc_w, surface_z_vals = sample_points(surface_depth_sample, surface_T_sample, dir_camcoord_sample, self.min_depth_train, self.dist_behind_surf_train, 0, 0, 1, self.surf_std_train)
        
        return masked_pc_w, z_vals, depth_sample, indices_b, indices_h, indices_w, masked_T_selected, \
        surface_pc_w.squeeze(), surface_depth_sample, surface_indices_b, surface_indices_h, surface_indices_w, surface_T_sample, surface_normal_sample

    
    def get_bounds(self, pc, z_vals, depth_sample, do_grad=True):
        with torch.set_grad_enabled(False):
            surf_pc = pc[:, 0] # torch.Size([187, 3]) get the point exactly on the surface
            diff = pc[:, :, None] - surf_pc # torch.Size([187, 27, 1, 3]) - torch.Size([187, 3]) = torch.Size([187, 27, 187, 3]) difference of each point from all sampled surf_pc
            dists = diff.norm(dim=-1) # torch.Size([187, 27, 187])
            dists, closest_ixs = dists.min(axis=-1) # torch.Size([187, 27])
            behind_surf = z_vals > depth_sample[:, None] # torch.Size([187, 27]) 1048:4001
            dists[behind_surf] *= -1
            bounds = dists

            grad = None
            if do_grad:
                ix1 = torch.arange(
                    diff.shape[0])[:, None].repeat(1, diff.shape[1]) # torch.Size([187, 27])
                ix2 = torch.arange(
                    diff.shape[1])[None, :].repeat(diff.shape[0], 1) # torch.Size([187, 27])
                grad = diff[ix1, ix2, closest_ixs] # torch.Size([187, 27, 3]) # the distance of the closest point to the surface
                grad = grad[:, 1:] # leave the surface
                grad = grad / grad.norm(dim=-1)[..., None]
                # flip grad vectors behind the surf
                grad[behind_surf[:, 1:]] *= -1

        return bounds, grad
    
    def frame_avg_loss(self, total_loss_mat, indices_b, indices_h, indices_w, B, H, W, loss_approx_factor, device, mode = 'block'):

        full_loss = torch.zeros([B,H,W], device = device)
        full_loss[indices_b, indices_h, indices_w] = total_loss_mat.detach()
        
        if mode=='pixel':
            loss_approx = self.pixel_approx_loss(full_loss)
            frame_avg_loss = loss_approx

        elif mode=='block':
            loss_approx = self.block_approx_loss(full_loss, factor=loss_approx_factor)
            frame_avg_loss = loss_approx.sum(dim=(1, 2)) / (loss_approx_factor * loss_approx_factor)
        else:
            loss_approx = None
            frame_avg_loss = None
            print('TBD')

        return loss_approx, frame_avg_loss
