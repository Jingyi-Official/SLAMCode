import copy
import torch
import numpy as np
from utilities.dataformat.frame_data import FrameData
from utilities.transforms.depth_transforms import pointcloud_from_depth_torch
from utilities.transforms.point_transforms import estimate_pointcloud_normals
from utilities.transforms.sdf_transforms import sdf_render_depth
from utilities.tools.timing import start_timing, end_timing
from utilities.sampling.frame_sampling import sample_frames
from utilities.sampling.pixel_sampling import sample_pixels
from utilities.sampling.ray_sampling import sample_points
from utilities.dataformat.batch_data import get_selected_frames_batch, get_selected_pixels_batch
from utilities.bounds import batch_bounds, normal_bounds, ray_bounds
from torchinfo import summary
from utilities.tools.calculate import cal_gradient_torch

class BaseTrainer():
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
        
        # TODO callbacks function 
        
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

        # # assert
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
        # self.model.scale_input = 1./torch.max(torch.abs(torch.from_numpy(self.scene_dataset.bounds_corners).float().to(self.device))) # for tensorf
        summary(self.model, input_size=(1000, 27, 3))
        self.log.info("Finish adapting model setting.")

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

    
    def get_current_frame_id(self):
        return int(self.tot_step_time * self.scene_dataset.fps)   


    def get_frame(self, idx):
        sample = self.scene_dataset[idx]

        rgb_np = sample["rgb"][None, ...]
        depth_np = sample["ndepth"][None, ...]
        T_np = sample["T"][None, ...]

        rgb = torch.from_numpy(rgb_np).float().to(self.device)
        depth = torch.from_numpy(depth_np).float().to(self.device)
        T = torch.from_numpy(T_np).float().to(self.device)

        data = FrameData(
            frame_id=np.array([idx]),
            rgb_batch=rgb,
            rgb_batch_np=rgb_np,
            depth_batch=depth,
            depth_batch_np=depth_np,
            T_batch=T,
            T_batch_np=T_np,
        )
        
        if self.do_normal:
            # pc = pointcloud_from_depth_torch(depth[0], self.scene_dataset.Ks[0,0], self.scene_dataset.Ks[1,1], self.scene_dataset.Ks[0,2], self.scene_dataset.Ks[1,2], T[0])
            pc = pointcloud_from_depth_torch(depth[0], self.scene_dataset.Ks[0,0], self.scene_dataset.Ks[1,1], self.scene_dataset.Ks[0,2], self.scene_dataset.Ks[1,2], None)
            normals = estimate_pointcloud_normals(pc)
            data.normal_batch = normals[None, :]

        if self.do_frame_active:
            data.frame_avg_losses = torch.zeros([1], device=self.device)
            data.frame_loss_approxes = torch.zeros([1, self.loss_approx_factor, self.loss_approx_factor], device=self.device)

        if self.do_sample_active:
            data.sample_avg_losses = torch.zeros([1], device=self.device)


        return data
    
    def add_frame(self, frame_data):
        self.frames.add_frame_data(frame_data, replace = self.last_is_keyframe is False)
        

    def check_last_is_keyframe(self):
        # determine by whether the model loss large
        self.log.info('Check whether is keyframe by model.')
        is_keyframe_by_model = self.check_is_keyframe_by_model()
        
        # determine by the time interval
        self.log.info('Check whether is keyframe by time.')
        is_keyframe_by_time = self.check_is_keyframe_by_time()

        is_keyframe = is_keyframe_by_model or is_keyframe_by_time

        return is_keyframe
        

    def check_is_keyframe_by_model(self):
        samples = self.get_samples_batch_check()
        pc = samples["pc"]
        z_vals = samples["z_vals"]
        depth_sample = samples["depth_sample"]

        with torch.set_grad_enabled(False):
            sdf, _ = self.frozen_sdf_map(pc, noise_std=self.noise_std)
            sdf = sdf.view(pc.shape[:2])

        z_vals, ind1 = z_vals.sort(dim=-1) # arange with order
        ind0 = torch.arange(z_vals.shape[0])[:, None].repeat(
            1, z_vals.shape[1])
        sdf = sdf[ind0, ind1] # get sdf with the z order

        view_depth = sdf_render_depth(z_vals, sdf)

        loss = torch.abs(view_depth - depth_sample) / depth_sample

        below_th = loss < self.kf_dist_th
        size_loss = below_th.shape[0]
        below_th_prop = below_th.sum().float() / size_loss
        is_keyframe = below_th_prop.item() < self.kf_pixel_ratio

        self.log.info(f"Proportion of loss below threshold {below_th_prop.item()}, for KF should be less than {self.kf_pixel_ratio}, Therefore is keyframe:{is_keyframe}")
        
        
        return is_keyframe
    
    def check_is_keyframe_by_time(self):
        time_since_kf = self.tot_step_time - self.frames.frame_id[-2] / self.scene_dataset.fps
        is_keyframe = False
        if time_since_kf > self.max_time_since_kf:
            self.log.info(f"More than {self.max_time_since_kf} seconds since last kf, so add new")
            is_keyframe = True

        return is_keyframe
    
    def copy_model(self):
        self.frozen_sdf_map = copy.deepcopy(self.model)


    
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
        
        # memory limit thus have to sample pixels --------------------------------------------------------
        indices_b, indices_h, indices_w = sample_pixels(self.n_rays_train, len(idxs), self.depth_H, self.depth_W, self.frames.frame_loss_approxes, self.do_sample_active, device=self.device)
        
        # get chosen valid samples
        depth_sample, T_sample, normal_sample, dir_camcoord_sample, [indices_b, indices_h, indices_w] = get_selected_pixels_batch(depth_batch_selected, T_batch_selected, normal_batch_selected, self.dir_camcoord, indices_b, indices_h, indices_w)

        # memory limit thus have to sample points --------------------------------------------------------
        pc, z_vals = sample_points(depth_sample, T_sample, dir_camcoord_sample, self.min_depth_train, self.dist_behind_surf_train, self.n_strat_samples_train, self.strat_bin_len_train, self.n_surf_samples_train, self.surf_std_train)

        
        samples = {
            "idxs": idxs,
            "indices_b": indices_b,
            "indices_h": indices_h,
            "indices_w": indices_w,
            "depth_sample": depth_sample,
            "T_sample": T_sample,
            "normal_sample": normal_sample,
            "dir_camcoord_sample": dir_camcoord_sample,
            "pc": pc,
            "z_vals": z_vals,
        }

        return samples

    def get_samples_batch_check(self):
        # get all the key frames
        depth_batch = self.frames.depth_batch
        T_batch = self.frames.T_batch
        normal_batch = None # doesn't consider normal batch here
        n_frames = len(self.frames)
        assert n_frames == depth_batch.shape[0] == T_batch.shape[0]
        idxs = np.array([n_frames-1])

        # get chosen valid frames
        depth_batch_selected, T_batch_selected, normal_batch_selected = get_selected_frames_batch(depth_batch, T_batch, normal_batch, idxs)
        
        # memory limit thus have to sample pixels --------------------------------------------------------
        indices_b, indices_h, indices_w = sample_pixels(self.n_rays_check, len(idxs), self.depth_H, self.depth_W, self.frames.frame_loss_approxes, self.do_sample_active, device=self.device)
        
        # get chosen valid samples
        depth_sample, T_sample, normal_sample, dir_camcoord_sample, [indices_b, indices_h, indices_w] = get_selected_pixels_batch(depth_batch_selected, T_batch_selected, normal_batch_selected, self.dir_camcoord, indices_b, indices_h, indices_w)

        
        # memory limit thus have to sample points --------------------------------------------------------
        pc, z_vals = sample_points(depth_sample, T_sample, dir_camcoord_sample, self.min_depth_check, self.dist_behind_surf_check, self.n_strat_samples_check, self.strat_bin_len_check, self.n_surf_samples_check, self.surf_std_check)

        samples = {
            "idxs": idxs,
            "indices_b": indices_b,
            "indices_h": indices_h,
            "indices_w": indices_w,
            "depth_sample": depth_sample,
            "T_sample": T_sample,
            "normal_sample": normal_sample,
            "dir_camcoord_sample": dir_camcoord_sample,
            "pc": pc,
            "z_vals": z_vals,
        }

        return samples
    
    def get_bounds(
        self,
        bounds_method,
        dir_camcoord_sample,
        depth_sample,
        T_sample,
        z_vals,
        pc,
        normal_trunc_dist,
        normal_sample,
        do_grad=True,
    ):
        """ do_grad: compute approximate gradient vector. """
        assert bounds_method in ["ray", "normal", "pc"]

        if bounds_method == "ray":
            bounds, grad = ray_bounds.bounds_ray(depth_sample, z_vals, dir_camcoord_sample, T_sample, do_grad)

        elif bounds_method == "normal":
            bounds, grad = normal_bounds.bounds_normal(depth_sample, z_vals, dir_camcoord_sample, normal_sample, normal_trunc_dist, T_sample, do_grad)

        else:
            bounds, grad = batch_bounds.bounds_pc(pc, z_vals, depth_sample, do_grad)

        return bounds, grad

    
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

        sdf, sdf_grad = self.model(pc, noise_std=self.noise_std, do_grad=do_sdf_grad) # torch.Size([181, 27])

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
        

        return losses, step_time


    # losses ---------------------------------------------------------------------------------
    def grad_loss(self,sdf_grad, normal_sample, grad_vec, do_orien_loss = False):
        pred_norms = sdf_grad[:, 0] # get the norm from the surface
        cosSim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        surf_loss_mat = 1 - cosSim(pred_norms, normal_sample)

        grad_vec[torch.where(grad_vec[..., 0].isnan())] = normal_sample[torch.where(grad_vec[..., 0].isnan())[0]]
        grad_loss_mat = 1 - cosSim(grad_vec, sdf_grad[:, 1:])
        grad_loss_mat = torch.cat((surf_loss_mat[:, None], grad_loss_mat), dim=1)

        if do_orien_loss:
            grad_loss_mat = (grad_loss_mat > 1).float()

        return grad_loss_mat
    
    def eik_loss(self, sdf_grad):
        eik_loss_mat = torch.abs(sdf_grad.norm(2, dim=-1) - 1)
        return eik_loss_mat

    def sdf_loss(self, sdf, bounds, t, loss_type="L1"):
        """
            params:
            sdf: predicted sdf values.
            bounds: upper bound on abs(sdf)
            t: truncation distance up to which the sdf value is directly supevised. # to decide whether it's free space or near surface
            loss_type: L1 or L2 loss.
        """
        free_space_loss_mat, trunc_loss_mat = self.full_sdf_loss(sdf, bounds)

        free_space_ixs = bounds > t
        free_space_loss_mat[~free_space_ixs] = 0.
        trunc_loss_mat[free_space_ixs] = 0.

        sdf_loss_mat = free_space_loss_mat + trunc_loss_mat

        if loss_type == "L1":
            sdf_loss_mat = torch.abs(sdf_loss_mat)
        elif loss_type == "L2":
            sdf_loss_mat = torch.square(sdf_loss_mat)
        else:
            raise ValueError("Must be L1 or L2")

        return sdf_loss_mat, free_space_ixs

    def full_sdf_loss(self, sdf, target_sdf, free_space_factor=5.0):
        """
        For samples that lie in free space before truncation region:
            loss(sdf_pred, sdf_gt) =  { max(0, sdf_pred - sdf_gt), if sdf_pred >= 0
                                    { exp(-sdf_pred) - 1, if sdf_pred < 0

        For samples that lie in truncation region:
            loss(sdf_pred, sdf_gt) = sdf_pred - sdf_gt
        """

        free_space_loss_mat = torch.max(
            torch.nn.functional.relu(sdf - target_sdf),
            torch.exp(-free_space_factor * sdf) - 1.
        )
        trunc_loss_mat = sdf - target_sdf

        return free_space_loss_mat, trunc_loss_mat
    
    def tot_loss(
        self, sdf_loss_mat, grad_loss_mat, eik_loss_mat,
        free_space_ixs, bounds, eik_apply_dist,
        trunc_weight, grad_weight, eik_weight,
    ):
        sdf_loss_mat[~free_space_ixs] *= trunc_weight
        # print("zero losses",
        #       sdf_loss_mat.numel() - sdf_loss_mat.nonzero().shape[0])

        losses = {"sdf_loss": sdf_loss_mat.mean().item()}
        tot_loss_mat = sdf_loss_mat

        # surface normal loss
        if grad_loss_mat is not None:
            tot_loss_mat = tot_loss_mat + grad_weight * grad_loss_mat
            losses["grad_loss"] = grad_loss_mat.mean().item()
            losses["surf_grad_loss"] = grad_loss_mat[:,0].mean().item()
            losses["space_grad_loss"] = grad_loss_mat[:,1:].mean().item()

        # eikonal loss
        if eik_loss_mat is not None:
            eik_loss_mat[bounds < eik_apply_dist] = 0.
            eik_loss_mat = eik_loss_mat * eik_weight
            tot_loss_mat = tot_loss_mat + eik_loss_mat
            losses["eikonal_loss"] = eik_loss_mat.mean().item()

        tot_loss = tot_loss_mat.mean()
        losses["total_loss"] = tot_loss

        return tot_loss, tot_loss_mat, losses


    def frame_avg_loss(self, total_loss_mat, indices_b, indices_h, indices_w, B, H, W, loss_approx_factor, device, mode = 'block'):
        full_loss = torch.zeros([B,H,W], device = device)
        full_loss[indices_b, indices_h, indices_w] = total_loss_mat.detach().sum(-1)
        
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
        
        
    def pixel_approx_loss(self, full_loss):  
        '''
        This function is used to calculate the pixel avg loss in a frame as the frame loss.
        '''
        loss_approx = full_loss.sum(dim=(1, 2)) 

        actives = torch.where(full_loss != 0, 1, 0).view(-1,full_loss.shape[1],full_loss.shape[2])
        actives = actives.sum(dim=(1, 2))

        loss_approx = loss_approx / actives

        return loss_approx


    def block_approx_loss(self, full_loss, factor=8):
        '''
        This function is used to calculate the block avg loss in a frame as the frame loss.
        '''
        w_block = full_loss.shape[2] // factor
        h_block = full_loss.shape[1] // factor

        loss_approx = full_loss.view(-1, factor, h_block, factor, w_block)
        loss_approx = loss_approx.sum(dim=(2, 4)) 

        actives = torch.where(full_loss != 0, 1, 0).view(-1, factor, h_block, factor, w_block)
        actives = actives.sum(dim=(2, 4))

        actives[actives == 0] = 1.0
        loss_approx = loss_approx / actives

        return loss_approx

    def evaluating(self, cache=True, volume=True, scene=True, surface=True, mesh=True, slice=True, save=False):
        '''
        cache: visible region
        volume: mesh grid
        scene: for the navigation part
        mesh: for the reconstructed mesh
        surface: the surface error
        '''

        metrics = {}

        if cache:
            metrics['cache_l1_error'], metrics['cache_l1_error_avg'], metrics['cache_l1_error_binned'], metrics['cache_coll_cost_error'] = self.evaluator.eval_sdf_cache(self.tot_step_time * self.scene_dataset.fps, self.model, save=save)
        
        if volume:
            metrics['volume_l1_error'], metrics['volume_l1_error_avg'], metrics['volume_l1_error_binned'], metrics['volume_coll_cost_error'] = self.evaluator.eval_sdf_volume(self.model, save=save)
        
        if scene:
            metrics['scene_l1_error'], metrics['scene_l1_error_avg'], metrics['scene_l1_error_binned'], metrics['scene_coll_cost_error'] = self.evaluator.eval_sdf_scene(self.model, save=save)
        
        if surface:
            metrics['surface_l1_error'], metrics['surface_l1_error_avg'], metrics['surface_l1_error_binned'], metrics['surface_coll_cost_error'] = self.evaluator.eval_sdf_surface(self.tot_step_time * self.scene_dataset.fps, self.model, save=save)
        
        if mesh:
            metrics['mesh_accuracy'], metrics["mesh_completion"] = self.evaluator.eval_mesh(self.model, save=save)

        if slice:
            self.evaluator.eval_slice(self.model, save=save)


        return metrics




