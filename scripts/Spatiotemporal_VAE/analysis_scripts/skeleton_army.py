from common.utils import pool_points, numpy2tensor, tensor2numpy, expand1darr, append_lists

import numpy as np
import umap


class ArmyVisualizer:
    """
    For VAE conditioned on Direction only
    """

    def __init__(self, model_container, kernel_size, scaling, save_video_path, one_picture_mode):
        self.model_container = model_container
        self.save_video_path = save_video_path
        self.kernel_size = kernel_size
        self.scaling = scaling
        self.one_picture_mode = one_picture_mode

        # Resesrved
        self.all_inputs, self.all_latents, self.all_recon = None, None, None
        self.all_tasks, self.all_tasks_mask, self.all_directions = None, None, None
        self.pooled_inputs, self.pooled_latents, self.pooled_recon = None, None, None
        self.pooled_tasks, self.pooled_tasks_mask, self.pooled_directions = None, None, None
        self.pooled_indexes = None
        self.trans_inputs, self.trans_recon = None, None


    def process(self):
        self._forward_pass()

        self._umapping()

        self._pooling_and_transform()

        pass

    def _forward_pass(self):
        """
        1. Forward pass to the network and obtain the output from data generator
        2. Concatenate the outputs
        """
        inputs_list, latents_list, recon_list, directions_list, tasks_list, tasks_mask_list = [], [], [], [], [], []
        for train_data, test_data in self.model_container.data_gen.iterator():
            # Forward pass
            inputs, _, tasks, task_masks, phenos, pheno_masks, directions, leg, leg_masks, _ = train_data
            directions2d = expand1darr(arr=directions, dim=3)
            inputs_tensor, directions2d_tensor = numpy2tensor(self.model_container.device, inputs, directions2d)
            recon_motion_tensor, _, _, motion_z_tensor = self.model_container._forward_pass(inputs_tensor,
                                                                                            directions2d_tensor)
            recon_motion, motion_z = tensor2numpy(recon_motion_tensor, motion_z_tensor)

            # Append outputs to lists
            inputs_list, latents_list, recon_list, directions_list, tasks_list, tasks_mask_list = append_lists(
                [inputs, motion_z, recon_motion, directions, tasks, task_masks],
                inputs_list, latents_list, recon_list, directions_list, tasks_list, tasks_mask_list
            )

        # Concatenate to form final data for drawing
        self.all_inputs, self.all_latents = np.vstack(inputs_list), np.vstack(latents_list)
        self.all_recon, self.all_tasks = np.vstack(recon_list), np.concatenate(tasks_list)
        self.all_tasks_mask, self.all_directions = np.concatenate(tasks_mask_list), np.concatenate(directions_list)

    def _umapping(self):
        umapper = umap.UMAP(n_neighbors=15,
                            n_components=2,
                            min_dist=0.1,
                            metric="euclidean")
        self.all_latents = umapper.fit_transform(self.all_latents)

    def _pooling_and_transform(self):

        # Pooling and slicing
        self.pooled_latents, self.pooled_indexes = pool_points(self.all_latents, self.kernel_size)
        self.pooled_inputs, self.pooled_recon = self.all_inputs[self.pooled_indexes], self.all_recon[self.pooled_indexes]
        self.pooled_tasks, self.pooled_tasks_mask = self.all_tasks[self.pooled_indexes], self.all_tasks_mask[self.pooled_indexes]
        self.pooled_directions = self.all_directions[self.pooled_indexes]

        # Scale and translate the skeleton to their latents
        self.trans_inputs = (self.pooled_inputs * self.scaling - self.pooled_latents)
        self.trans_recon = self.pooled_recon * self.scaling - self.pooled_latents



