import os
import numpy as np
import torch
import trimesh
import scipy
import copy
import cv2
from dataset.cache import DepthCacheDataset
from utilities.sampling.pixel_sampling import sample_pixels
from utilities.sampling.ray_sampling import sample_points
from utilities.dataformat.batch_data import get_selected_frames_batch, get_selected_pixels_batch
from utilities.metrics.sdf_error import binned_errors
from utilities.metrics.cost_error import chomp_cost_errors
from utilities.metrics.mesh_error import accuracy_comp, accuracy, completion
from utilities.transforms.grid_transforms import transform_grid_pts, make_3D_grid
from utilities.transforms.sdf_transforms import sdf_render_mesh
from utilities.vis.tools import get_colormap


class BaseEvaluator(object):
    def __init__(self, 
                    device = None,
                    # cache / volumn / scene evaluation ----
                    n_samples = 200000, 
                    dataset = None, 
                    interval = 5, 
                    min_depth_eval_visible = 0.07,
                    dist_behind_surf_eval_visible = 0.1,
                    n_strat_samples_eval_visible = 1,
                    n_surf_samples_eval_visible = 0,
                    strat_bin_len = None,
                    surf_std = 0.1,
                    noise_std = 0,
                    # mesh evaluation ---------------------
                    grid_dim = [256, 256, 256],
                    grid_res = None,
                    save_dir = None,
                ):

        self.device = device
        
        # eval for sdf by cache/volume/scene dataset------------------------
        self.n_samples = n_samples
        
        # eval for cache dataset -------------------------------------------
        self.scene_dataset = copy.deepcopy(dataset)
        self.interval = interval
        self.scene_cache = DepthCacheDataset(self.scene_dataset, self.interval)
        self.scene_sdf_kit = self.scene_dataset.scene_sdf_habitat
        # visible sampling ----------
        self.min_depth_eval_visible = min_depth_eval_visible
        self.dist_behind_surf_eval_visible = dist_behind_surf_eval_visible
        self.n_strat_samples_eval_visible = n_strat_samples_eval_visible
        self.n_surf_samples_eval_visible = n_surf_samples_eval_visible
        self.strat_bin_len = strat_bin_len
        self.surf_std = surf_std
        self.noise_std = noise_std

        # eval for volume dataset ------------------------------------------
        self.eval_volume_grid = self.get_eval_sample_volume()


        # eval for scene dataset -------------------------------------------
        self.scene_stage_sdf_kit = self.scene_dataset.scene_sdf_stage_habitat
        self.eval_scene_grid = self.get_eval_sample_scene()
        

        # eval for mesh ---------------------------------------------------
        self.scene_mesh = self.scene_dataset.scene_mesh
        self.scene_mesh_bounds_transform = self.scene_dataset.bounds_transform
        self.scene_mesh_bounds_extents = self.scene_dataset.bounds_extents
        self.eval_mesh_grid, self.scene_scale, self.grid_dim = self.get_eval_mesh_grid(grid_dim, grid_res)
        
        # eval for slices ---------------------------------------------------
        self.eval_slice_pts = self.get_slice_pts(z_idxs=None, n_slices=6)
        self.cmap = get_colormap(sdf_range=[-2, 2])
        
        # saving---------------------------------------------------------------
        self.save_dir = save_dir


    def get_eval_mesh_grid(self, grid_dim=[256,256,256], grid_res=None, grid_range=[[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]):
        grid_range = np.array(grid_range)
        range_dist = grid_range[:, 1] - grid_range[:, 0]
        scene_scale = self.scene_mesh_bounds_extents / (range_dist * 0.9)
        scene_transform = self.scene_mesh_bounds_transform
        
        if grid_res:
            grid_res = np.array(grid_res)
            grid_dim = (np.ceil(scene_scale / grid_res)).astype(int)
        else:
            grid_dim = np.array(grid_dim)

        
        eval_mesh_grid = make_3D_grid(
            grid_range, 
            grid_dim,
            self.device,
            transform = scene_transform,
            scale = scene_scale,
        ).view(-1, 3)

        return eval_mesh_grid, scene_scale, list(grid_dim)

    def get_slice_pts(self, z_idxs=None, n_slices=6):

        # get the z idx for slicing ------------------------------------------------------------
        if z_idxs is None:
            self.z_idxs  = torch.linspace(30, self.grid_dim[0] - 30, n_slices) 
            self.z_idxs  = torch.round(self.z_idxs).long()

        # select the pc which have the indicated z idx -----------------------------------------------------
        pc = self.eval_mesh_grid.reshape(self.grid_dim[0], self.grid_dim[1], self.grid_dim[2], 3)
        pc = torch.index_select(pc, self.scene_dataset.up_world, self.z_idxs.to(pc.device))

        if not self.scene_dataset.aligned_up:
            indices = np.arange(len(self.z_idxs ))[::-1]
            indices = torch.from_numpy(indices.copy())
            pc = torch.index_select(pc, self.scene_dataset.up_world, indices.to(pc.device))

        return pc

    # def get_sdf_pred(self, pc, model):
    #     with torch.set_grad_enabled(False):
    #         pred = model(pc, noise_std=self.noise_std)

    #     return pred

    def get_sdf_pred(self, pc, model):
        with torch.set_grad_enabled(False):
            pred, _ = model(pc, noise_std=self.noise_std, do_grad=False)

        return pred
 
    def get_sdf_gt(self, pc, scene_sdf_kit, handle_oob='mask', oob_val=0.):
        
        reshaped = False
        if pc.ndim != 2:
            reshaped = True
            pc_shape = pc.shape[:-1]
            pc = pc.reshape(-1, 3)

        '''
        The reason that need to deal with out of bounds (why pc will be out of bounds)
        sampling may exceed the boundary
        '''
        if handle_oob == 'except':
            scene_sdf_kit.bounds_error = True
        elif handle_oob == 'mask':
            dummy_val = 1e99
            scene_sdf_kit.bounds_error = False
            scene_sdf_kit.fill_value = dummy_val
        elif handle_oob == 'fill':
            scene_sdf_kit.bounds_error = False
            scene_sdf_kit.fill_value = oob_val
        else:
            assert True, "handle_oob must take a recognised value."

        sdf = scene_sdf_kit(pc)

        if reshaped:
            sdf = sdf.reshape(pc_shape)

        if handle_oob == 'mask':
            valid_mask = sdf != dummy_val
            return sdf, valid_mask

        return sdf
        
    def get_cache_set(self, time):
        return self.scene_cache[int(time)]

    def get_eval_sample_cache(self, time, min_depth, dist_behind_surf, n_strat_samples, strat_bin_len, n_surf_samples, surf_std):
        cache_set = self.get_cache_set(time)
        idxs = self.scene_cache.get_cache_idxs(int(time))
        n_frames = len(idxs)

        depth_batch_selected = torch.FloatTensor(cache_set["depth"]).to(self.device)
        T_batch_selected = torch.FloatTensor(cache_set["T"]).to(self.device)

        n_rays = self.n_samples // depth_batch_selected.shape[0]


        # random sample pixels for evaluation --------------------------------------------------------
        indices_b, indices_h, indices_w = sample_pixels(n_rays = n_rays, n_frames = n_frames, H=self.scene_dataset.depth_H, W=self.scene_dataset.depth_W, do_sample_active=False, device = self.device)
        
        # get chosen valid samples
        depth_sample, T_sample, normal_sample, dir_camcoord_sample, [indices_b, indices_h, indices_w] = get_selected_pixels_batch(depth_batch_selected=depth_batch_selected, T_batch_selected=T_batch_selected, normal_batch_selected=None, dir_camcoord = self.scene_dataset.dir_camcoord.to(self.device), indices_b=indices_b, indices_h=indices_h, indices_w=indices_w)

        # memory limit thus have to sample points --------------------------------------------------------
        pc, z_vals = sample_points(depth_sample, T_sample, dir_camcoord_sample, min_depth, dist_behind_surf, n_strat_samples, strat_bin_len, n_surf_samples, surf_std)

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
   
    def sdf_l1_error(self, sdf_gt, sdf_pred):
        return torch.abs(sdf_pred - sdf_gt)

    def sdf_binned_error(self, sdf_loss, sdf_gt, bin_limits=np.array([-0.05, -0.04, -0.03, -0.02, -0.01, 0., 0.01, 0.02, 0.03, 0.04, 0.05])):
        sdf_binned_errors = binned_errors(sdf_loss, sdf_gt, bin_limits=bin_limits)
        return dict(zip(bin_limits[1:],sdf_binned_errors))
        # return sdf_binned_errors

    def chomp_cost_error(self, sdf_gt, sdf_pred, epsilons = [1., 1.5, 2.]):
        coll_cost_errors = chomp_cost_errors(sdf_gt, sdf_pred, epsilons = epsilons)
        return dict(zip(epsilons, coll_cost_errors))
        # return coll_cost_errors

    def eval_sdf_cache(self, time, model, save=False):
        # get sample points
        samples = self.get_eval_sample_cache(time, self.min_depth_eval_visible, self.dist_behind_surf_eval_visible, self.n_strat_samples_eval_visible, self.strat_bin_len, self.n_surf_samples_eval_visible, self.surf_std)
        pts = samples["pc"]

        # get prediction
        sdf_pred = self.get_sdf_pred(pts, model)
        sdf_pred = sdf_pred.flatten()
        pts = pts.squeeze()


        # get target
        sdf_gt, valid_mask = self.get_sdf_gt(pts.cpu().detach().numpy(), self.scene_sdf_kit, handle_oob='mask')

        # get masks: gt sdf gives value 0 inside the walls. Don't include this in loss, filter invalid
        valid_mask = np.logical_and(sdf_gt != 0., valid_mask)

        # get valid points for eval
        sdf_gt = sdf_gt[valid_mask]
        sdf_pred = sdf_pred[valid_mask]

        sdf_gt = torch.from_numpy(sdf_gt).to(self.device)

        with torch.set_grad_enabled(False):
            # l1 loss
            l1_error = self.sdf_l1_error(sdf_gt, sdf_pred)
            l1_error_avg = l1_error.mean()
            l1_error_binned = self.sdf_binned_error(l1_error, sdf_gt)

            # chomp cost difference
            coll_cost_error = self.chomp_cost_error(sdf_gt, sdf_pred, epsilons = [1., 1.5, 2.])

        # if save:
        #     # debugging: check the eval_points
        #     np.save(os.path.join(self.save_dir, "eval_sdf_cache_pts.npy"), pts.cpu().detach().numpy())
        #     np.save(os.path.join(self.save_dir, "eval_sdf_cache_valid_mask.npy"), valid_mask)
        #     os.path.join(self.save_dir, "eval_sdf_cache_pts.npy")

        
        
        return l1_error, l1_error_avg, l1_error_binned, coll_cost_error

    def eval_sdf_volume(self, model, save=False):
        # get sample points
        pts = self.get_eval_sample_volume().float().to(self.device)

        # get prediction
        sdf_pred = self.get_sdf_pred(pts, model)
        sdf_pred = torch.squeeze(sdf_pred)

        # get target
        sdf_gt, valid_mask = self.get_sdf_gt(pts.cpu().detach().numpy(), self.scene_sdf_kit, handle_oob='mask')

        # get masks: gt sdf gives value 0 inside the walls. Don't include this in loss, filter invalid
        valid_mask = np.logical_and(sdf_gt != 0., valid_mask)

        # get valid points for eval
        sdf_gt = sdf_gt[valid_mask]
        sdf_pred = sdf_pred[valid_mask]

        sdf_gt = torch.from_numpy(sdf_gt).to(self.device)

        with torch.set_grad_enabled(False):
            # l1 loss
            l1_error = self.sdf_l1_error(sdf_gt, sdf_pred)
            l1_error_avg = l1_error.mean()
            l1_error_binned = self.sdf_binned_error(l1_error, sdf_gt)
        
            # chomp cost difference
            coll_cost_error = self.chomp_cost_error(sdf_gt, sdf_pred, epsilons = [1., 1.5, 2.])

        
        # if save:
        #     # debugging: check the eval_points
        #     np.save(os.path.join(self.save_dir, "eval_sdf_volume_pts.npy"), pts.cpu().detach().numpy())
        #     np.save(os.path.join(self.save_dir, "eval_sdf_volume_valid_mask.npy"), valid_mask)
        #     os.path.join(self.save_dir, "eval_sdf_cache_pts.npy")
        
        return l1_error, l1_error_avg, l1_error_binned, coll_cost_error

    def eval_sdf_scene(self, model, save=False):
        # get sample points
        pts = self.get_eval_sample_scene().float().to(self.device)

        # get prediction
        sdf_pred = self.get_sdf_pred(pts, model)
        sdf_pred = torch.squeeze(sdf_pred)

        # get target
        sdf_gt, valid_mask = self.get_sdf_gt(pts.cpu().detach().numpy(), self.scene_sdf_kit, handle_oob='mask')

        # get masks: gt sdf gives value 0 inside the walls. Don't include this in loss, filter invalid
        valid_mask = np.logical_and(sdf_gt != 0., valid_mask)

        # get valid points for eval
        sdf_gt = sdf_gt[valid_mask]
        sdf_pred = sdf_pred[valid_mask]

        sdf_gt = torch.from_numpy(sdf_gt).to(self.device)

        with torch.set_grad_enabled(False):
            # l1 loss
            l1_error = self.sdf_l1_error(sdf_gt, sdf_pred)
            l1_error_avg = l1_error.mean()
            l1_error_binned = self.sdf_binned_error(l1_error, sdf_gt)
        
            # chomp cost difference
            coll_cost_error = self.chomp_cost_error(sdf_gt, sdf_pred, epsilons = [1., 1.5, 2.])

        # if save:
        #     # debugging: check the eval_points
        #     np.save(os.path.join(self.save_dir, "eval_sdf_scene_pts.npy"), pts.cpu().detach().numpy())
        #     np.save(os.path.join(self.save_dir, "eval_sdf_scene_valid_mask.npy"), valid_mask)
        #     os.path.join(self.save_dir, "eval_sdf_cache_pts.npy")
        
        
        return l1_error, l1_error_avg, l1_error_binned, coll_cost_error

    def eval_sdf_surface(self, time, model, save=False):
        # get sample points
        samples = self.get_eval_sample_cache(time, self.min_depth_eval_visible, self.dist_behind_surf_eval_visible, 0, self.strat_bin_len, 1, self.surf_std)
        pts = samples["pc"]

        # get prediction
        sdf_pred = self.get_sdf_pred(pts, model)
        sdf_pred = sdf_pred.flatten()
        pts = pts.squeeze()


        # get target
        sdf_gt, valid_mask = self.get_sdf_gt(pts.cpu().detach().numpy(), self.scene_sdf_kit, handle_oob='mask')

        # get masks: gt sdf gives value 0 inside the walls. Don't include this in loss, filter invalid
        valid_mask = np.logical_and(sdf_gt != 0., valid_mask)

        # get valid points for eval
        sdf_gt = sdf_gt[valid_mask]
        sdf_pred = sdf_pred[valid_mask]

        sdf_gt = torch.from_numpy(sdf_gt).to(self.device)

        with torch.set_grad_enabled(False):
            # l1 loss
            l1_error = self.sdf_l1_error(sdf_gt, sdf_pred)
            l1_error_avg = l1_error.mean()
            l1_error_binned = self.sdf_binned_error(l1_error, sdf_gt)

            # chomp cost difference
            coll_cost_error = self.chomp_cost_error(sdf_gt, sdf_pred, epsilons = [1., 1.5, 2.])

        # if save:
        #     # debugging: check the eval_points
        #     np.save(os.path.join(self.save_dir, "eval_sdf_cache_pts.npy"), pts.cpu().detach().numpy())
        #     np.save(os.path.join(self.save_dir, "eval_sdf_cache_valid_mask.npy"), valid_mask)
        #     os.path.join(self.save_dir, "eval_sdf_cache_pts.npy")

        
        
        return l1_error, l1_error_avg, l1_error_binned, coll_cost_error
   
    def get_eval_sample_scene(self):
        # get sample points
        pts = self.eval_volume_grid

        # filter points for evaluation
        if self.scene_stage_sdf_kit is not None:
            stage_sdf_gt = self.scene_stage_sdf_kit(pts)
            pts = pts[stage_sdf_gt>0]

            
            min_xy = self.scene_dataset.scene_min_xy
            islands = self.scene_dataset.scene_islands
            px = torch.floor((pts[:, 0] - min_xy[0]) / min_xy[2])
            py = torch.floor((pts[:, 2] - min_xy[1]) / min_xy[2])
            px = torch.clamp(px, min=0, max=islands.shape[1] - 1).int()
            py = torch.clamp(py, min=0, max=islands.shape[0] - 1).int()

            # discard2_pts = eval_pts[islands[py, px] == 1]
            pts = pts[islands[py, px] == 0]

        return pts
   
    def get_eval_sample_volume(self):
        # get sample points
        pts = torch.rand(self.n_samples, 3)
        pts = pts * (torch.tensor(self.scene_dataset.get_habitat_sdf().shape) -1)
        pts = pts * self.scene_dataset.habitat_transform[0, 0]
        pts = pts + self.scene_dataset.habitat_transform[:3, 3]

        return pts

    def get_sdf_pred_chunks(self, pc, model, chunk_size = 200000, to_cpu=False):
        n_pts = pc.shape[0]
        n_chunks = int(np.ceil(n_pts / chunk_size))
        sdfs = []
        for n in range(n_chunks):
            start = n * chunk_size
            end = start + chunk_size
            chunk = pc[start:end, :]

            sdf,_ = model(chunk, do_grad=False) # torch.Size([200000])

            sdf = sdf.squeeze(dim=-1)
            if to_cpu:
                sdf = sdf.cpu()
            sdfs.append(sdf)

        sdfs = torch.cat(sdfs, dim=-1)

        return sdfs
    
    def eval_mesh(self, model, save=False):
        # sdf_pred by mesh grid # Query network on dense 3d grid of points
        with torch.set_grad_enabled(False):
            sdf = self.get_sdf_pred_chunks(
                pc = self.eval_mesh_grid,
                model = model
            )

            sdf = sdf.view(self.grid_dim)
        # get mesh from sdf
        mesh_pred = sdf_render_mesh(sdf, self.scene_scale, self.scene_mesh_bounds_transform,)
        
        # mesh_gt
        mesh_gt = self.scene_dataset.scene_mesh
        
        # eval
        acc, comp = accuracy_comp(mesh_gt, mesh_pred, samples=self.n_samples)

        # save
        if save:
            # debugging: save checkpoint
            torch.save(model.state_dict(), os.path.join(self.save_dir, "model.pth"))
            
            # debugging: check the eval_mesh
            rec_pc = trimesh.sample.sample_surface(mesh_pred, self.n_samples)
            rec_pc_tri = trimesh.PointCloud(vertices=rec_pc[0])

            gt_pc = trimesh.sample.sample_surface(mesh_gt, self.n_samples)
            gt_pc_tri = trimesh.PointCloud(vertices=gt_pc[0])

            acc = accuracy(gt_pc_tri.vertices, rec_pc_tri.vertices)
            comp = completion(gt_pc_tri.vertices, rec_pc_tri.vertices)

            mesh_pred = trimesh.exchange.ply.export_ply(mesh_pred)
            mesh_file = open(os.path.join(self.save_dir, "mesh.ply"), "wb+")
            mesh_file.write(mesh_pred)
            mesh_file.close()

            # rec_pc_tri.export('mesh_pred_sampled_pts.ply')
            # gt_pc_tri.export('mesh_gt_sampled_pts.ply')

        return acc, comp

    def eval_slice(self, model, save=False):
        # get pred sdf
        with torch.set_grad_enabled(False):
            sdf_pred = self.get_sdf_pred_chunks(self.eval_slice_pts.reshape(-1, 3), model)
            sdf_pred = sdf_pred.detach().cpu().numpy()
        
        # get colormap for visualization
        
        sdf_pred_viz = self.cmap.to_rgba(sdf_pred.flatten(), alpha=1., bytes=False)
        sdf_pred_viz = (sdf_pred_viz * 255).astype(np.uint8)[..., :3]
        sdf_pred_viz = sdf_pred_viz.reshape(*self.eval_slice_pts.shape[:3], 3)
        sdf_pred_viz = [np.take(sdf_pred_viz, i, self.scene_dataset.up_world) for i in range(sdf_pred_viz.shape[0])]

        # save slices
        if save:
            # cv2.imwrite(os.path.join(self.save_dir, "slices.ply"), np.vstack(sdf_pred_viz))
            print("todo")



    



