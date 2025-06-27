'''
Description: Try to uniformly sample points in space on the ray, Group 1 | Group 1.1
Author: 
Date: 2023-01-30 16:11:27
LastEditTime: 2023-03-28 19:13:44
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Reference: 
'''

import torch
import numpy as np
from utilities.dataformat.frame_data import FrameData
from utilities.tools.timing import start_timing, end_timing
from utilities.sampling.frame_sampling import sample_frames
from utilities.dataformat.batch_data import get_selected_frames_batch


from trainer.base import BaseTrainer
import random
from utilities.transforms.grid_transforms import xyz_to_points


class UniformTrainer(BaseTrainer):

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
        self.log.info("Finish adapting model setting.")


        self.log.info(f"Starting training for max {self.max_steps} steps...")

        self.vox_wcoords = xyz_to_points(self.scene_dataset.get_habitat_queries()).to(self.device)
        self.vox_wcoords = torch.cat([self.vox_wcoords, torch.ones(len(self.vox_wcoords), 1, device=self.vox_wcoords.device)], dim=1).float()
        

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
                    # self.logger.log_image(key="Keyframes", images=[os.path.join(self.scene_rgb_dir, f"frame%06d.png"%(self.frames.frame_id[-1]))])

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

            # debug
            # vis_loss(losses, self.model, self.evaluator)
            
            # evaluation step-----------------------------------------------
            elapsed_eval = self.tot_step_time - self.last_eval
            if t > self.iters_start and elapsed_eval > self.eval_freq_s:
                self.last_eval = self.tot_step_time - self.tot_step_time % self.eval_freq_s

                self.log.info(f'After {elapsed_eval}s, evaluating...')
                metrics = self.evaluating()

                self.log.info(f"Cache Evaluation: cache_l1_error_avg: {metrics['cache_l1_error_avg']} cache_coll_cost_error: {metrics['cache_coll_cost_error']}")
                # self.log.info(f"Volume Evaluation: volume_l1_error_avg: {metrics['volume_l1_error_avg']} volume_coll_cost_error: {metrics['volume_coll_cost_error']}")
                # self.log.info(f"Scene Evaluation: scene_l1_error_avg: {metrics['scene_l1_error_avg']} scene_coll_cost_error: {metrics['scene_coll_cost_error']}")
                # self.log.info(f"Mesh Evaluation: mesh_accuracy: {metrics['mesh_accuracy']} mesh_completion: {metrics['mesh_completion']}")
                self.logger.log_metrics(metrics=metrics,step=t)

        
        # final evaluation
        self.log.info(f"Finish training. Final evaluation:")
        metrics = self.evaluating(save=True)
        # self.log.info(f"Cache Evaluation: cache_l1_error_avg: {metrics['cache_l1_error_avg']} cache_coll_cost_error: {metrics['cache_coll_cost_error']}")
        # self.log.info(f"Volume Evaluation: volume_l1_error_avg: {metrics['volume_l1_error_avg']} volume_coll_cost_error: {metrics['volume_coll_cost_error']}")
        # self.log.info(f"Scene Evaluation: scene_l1_error_avg: {metrics['scene_l1_error_avg']} scene_coll_cost_error: {metrics['scene_coll_cost_error']}")
        # self.log.info(f"Mesh Evaluation: mesh_accuracy: {metrics['mesh_accuracy']} mesh_completion: {metrics['mesh_completion']}")
        self.logger.log_metrics(metrics=metrics,step=t)

    def training(self):
        start, end = start_timing()

        # get samples ------------------------------------------------------------------------------------------------
        samples = self.get_samples_batch()
        pts = samples["pc"] # torch.Size([183, 27, 3])
        zs = samples["z_vals"] # torch.Size([183, 27])
        dir_camcoord_sample = samples["dir_camcoord_sample"] # torch.Size([183, 3])
        depth_sample = samples["depth_sample"] # torch.Size([183])
        T_sample = samples["T_sample"] # torch.Size([183, 4, 4])
        normal_sample = samples["normal_sample"]

        bounds, grad_vec = self.get_bounds(
            self.bounds_method, 
            dir_camcoord_sample, 
            depth_sample, 
            T_sample, 
            zs, 
            pts, 
            self.trunc_distance, 
            normal_sample, 
            do_grad=True, 
        )

        new_start, new_end = start_timing()
        samples = self.get_samples_batch_new()
        pc = samples["pc"]
        indices_b = samples["indices_b"]
        indices_h = samples["indices_h"]
        indices_w = samples["indices_w"]
        idxs = samples["idxs"]
        sdf_gt, valid_mask = self.evaluator.get_sdf_gt(pc.cpu().detach().numpy(), self.evaluator.scene_sdf_kit, handle_oob='mask')
        bounds = torch.from_numpy(sdf_gt).type_as(bounds)#.to(self.device).type(torch.DoubleTensor)
        new_time = end_timing(new_start, new_end)

        
        do_sdf_grad = self.eik_weight != 0 or self.grad_weight != 0
        if do_sdf_grad:
            pc.requires_grad_()

        sdf = self.model(pc, noise_std=self.noise_std) # torch.Size([181, 27])

        # this is for regularizations
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

        step_time = end_timing(start, end) - new_time
        time_s = step_time / 1000.
        self.tot_step_time += 1 * time_s 
        

        return losses, step_time

    def get_samples_batch_new(self):
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
        depth_batch_selected, T_batch_selected, normal_batch_selected = get_selected_frames_batch(depth_batch, T_batch, normal_batch, idxs) # torch.Size([1, 680, 1200]) # torch.Size([1, 4, 4])
        
        
        # --------------------------- Changed part -----------------------------
        pc = []
        z_vals = []
        indices_h = []
        indices_w = []
        indices_b = []
        for i in range(len(idxs)):
            # convert volume from world coordinates to camera coordinates
            vox_ccoords = torch.matmul(torch.inverse((T_batch_selected[i]).to(self.vox_wcoords.device)), self.vox_wcoords.transpose(1, 0)).transpose(1, 0).float()  # [nx*ny*nz, 4]
            
            # Convert volume from camera coordinates to pixel coordinates; project all the voxels back to image plane
            pix_x = torch.round((vox_ccoords[:, 0] * self.scene_dataset.Ks[0, 0] / vox_ccoords[:, 2]) + self.scene_dataset.Ks[0, 2]).long()  # [nx*ny*nz]
            pix_y = torch.round((vox_ccoords[:, 1] * self.scene_dataset.Ks[1, 1] / vox_ccoords[:, 2]) + self.scene_dataset.Ks[1, 2]).long()  # [nx*ny*nz]
            pix_z = vox_ccoords[:, 2]

            # Eliminate pixels outside view frustum ------------------------------------------------------
            valid_pix = (pix_x >= 0) & (pix_x < self.depth_W) & (pix_y >= 0) & (pix_y < self.depth_H) & (pix_z > 0)  # [n_valid]
            valid_vox = self.vox_wcoords[valid_pix, :3] #.to(self.device)
            pix_x = pix_x[valid_pix] #.to(self.device)
            pix_y = pix_y[valid_pix] #.to(self.device)
            pix_z = pix_z[valid_pix] #.to(self.device)

            # random choose the points
            index = torch.LongTensor(random.sample(range(valid_vox.shape[0]), self.n_points)).to(self.device)
            pc = torch.cat((pc, torch.index_select(valid_vox, 0, index)), 0) if pc!=[] else torch.index_select(valid_vox, 0, index)
            z_vals = torch.cat((z_vals, torch.index_select(pix_z, 0, index)), 0) if z_vals!=[] else torch.index_select(pix_z, 0, index)
            indices_h = torch.cat((indices_h,torch.index_select(pix_y, 0, index)), 0) if indices_h!=[] else torch.index_select(pix_y, 0, index)
            indices_w = torch.cat((indices_w,torch.index_select(pix_x, 0, index)), 0) if indices_w!=[] else torch.index_select(pix_x, 0, index)
            b = torch.tensor([i],device=self.device)
            b = b.repeat_interleave(len(index))
            indices_b = torch.cat((indices_b, b), dim=0) if indices_b!=[] else b


        # --------------------------- Changed part  Finished -----------------------------
        samples = {
            "idxs": idxs,
            "indices_b": indices_b,
            "indices_h": indices_h,
            "indices_w": indices_w,
            "pc": pc,
            "z_vals": z_vals,
        }

        return samples

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
    

    def __init__(
        self,
        device,
        model,
        optimizer,
        callbacks,
        logger,
        log,
        evaluator,
        max_steps,
        do_normal,
        do_frame_active,
        do_sample_active,
        bounds_method,
        max_time_since_kf,
        window_size,
        window_limit_size,
        n_rays_train,
        min_depth_train,
        n_strat_samples_train,
        n_surf_samples_train,
        strat_bin_len_train,
        surf_std_train,
        dist_behind_surf_train,
        n_rays_check,
        min_depth_check,
        n_strat_samples_check,
        n_surf_samples_check,
        strat_bin_len_check,
        surf_std_check,
        dist_behind_surf_check,
        kf_dist_th,
        kf_pixel_ratio,
        trunc_distance,
        trunc_weight,
        grad_weight,
        eik_apply_dist,
        eik_weight,
        loss_approx_factor,
        iters_start,
        iters_per_frame,
        iters_per_kf,
        noise_std,
        noise_frame,
        noise_kf,
        n_points,
    ):
        super(BaseTrainer, self).__init__()

        self.device = device
        self.model = model
        self.optimiser = optimizer #(params=self.model.parameters())
        self.callbacks = callbacks
        self.logger = logger
        self.log = log
        self.evaluator = evaluator

        self.model = self.model.to(self.device)
        self.model.train()
        
        # Log gradients, parameters and model topology (100 steps by default)
        self.logger.watch(self.model, log="all")

        # Hyperparameters ----------------
        self.max_steps = max_steps
        self.do_normal = do_normal
        self.do_frame_active = do_frame_active
        self.do_sample_active = do_sample_active
        self.bounds_method = bounds_method

        self.max_time_since_kf = max_time_since_kf
        
        # for sample frames
        self.window_size = window_size
        self.window_limit_size = window_limit_size

        # for sample pixels
        self.n_rays_train = n_rays_train
        self.n_rays_check = n_rays_check

        # for sample points
        self.min_depth_train = min_depth_train
        self.n_strat_samples_train = n_strat_samples_train
        self.n_surf_samples_train = n_surf_samples_train
        self.dist_behind_surf_train = dist_behind_surf_train
        self.strat_bin_len_train = strat_bin_len_train
        self.surf_std_train = surf_std_train

        self.min_depth_check = min_depth_check
        self.n_strat_samples_check = n_strat_samples_check
        self.n_surf_samples_check = n_surf_samples_check
        self.dist_behind_surf_check = dist_behind_surf_check
        self.strat_bin_len_check = strat_bin_len_check
        self.surf_std_check = surf_std_check

        # for getting bounds
        # ray bounds:
        # normal bounds:
        self.trunc_distance = trunc_distance # to decide whether it's free space or near surface
        # pc bounds:

        
        # for loss
        self.trunc_weight = trunc_weight # lambda
        self.grad_weight = grad_weight
        self.eik_apply_dist = eik_apply_dist
        self.eik_weight = eik_weight

        # assert
        if self.grad_weight==0:
            self.do_normal=False
        
        # for active sampling loss
        self.loss_approx_factor = loss_approx_factor


        # flags for training -----------------------
        self.last_is_keyframe = True
        self.tot_step_time = 0.

        self.steps_since_frame = 0
        self.optim_frames = 0

        self.iters_per_frame = iters_per_frame # for normal frames
        self.noise_frame = noise_frame 
        self.iters_start = iters_start
        self.noise_std = noise_std

        self.iters_per_kf = iters_per_kf # for keyframe
        self.noise_kf = noise_kf
        
        # for checking whther it's keyframe
        self.kf_dist_th = kf_dist_th
        self.kf_pixel_ratio = kf_pixel_ratio

        # for evaluation
        self.last_eval = 0
        self.eval_freq_s = 1
        self.vis_freq_t = 1

        # Buffer ---------------------
        self.frames = FrameData()

        self.n_points = n_points

