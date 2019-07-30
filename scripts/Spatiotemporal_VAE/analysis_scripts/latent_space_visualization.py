import numpy as np
import umap
import os
import pandas as pd
import skvideo.io as skv
import torch
from common.utils import expand1darr, numpy2tensor, tensor2numpy, write_df_pickle, slice_by_mask
from common.visualisation import SkeletonPainter

def convert_direction_convex_combinaiton(labels, frac):
    onehot2d = expand1darr(labels.astype(np.int), 3)
    one_mask = onehot2d[:, 1, 0] == 1
    two_mask = onehot2d[:, 2, 0] == 1
    onehot2d[one_mask, 1, :] = onehot2d[one_mask, 1, :] * frac
    onehot2d[one_mask, 2, :] = 1 - onehot2d[one_mask, 1, :]
    onehot2d[two_mask, 2, :] = onehot2d[two_mask, 2, :] * frac
    onehot2d[two_mask, 1, :] = 1 - onehot2d[two_mask, 2, :]
    return onehot2d

class LatentSpaceSaver_CondDirect:
    """
    Working with scripts.STVAE_run.CSTVAEmodel, with
        (1) direction conditionals
    """

    def __init__(self, model_container, data_gen, fit_samples_num, save_data_dir, df_save_fn, vid_dirname,
                 model_identifier, draw):
        # Copy arguments
        self.model_container = model_container
        self.data_gen = data_gen
        self.fit_samples_num = fit_samples_num
        self.save_data_dir = save_data_dir
        self.df_save_fn = df_save_fn
        self.vid_dirname = vid_dirname
        self.model_identifier = model_identifier
        self.draw = draw

        # Reserved data
        self.x_ep, self.tasks_ep, self.phenos_ep, self.towards_ep, self.leg_ep = None, None, None, None, None
        self.x_base, self.tasks_base, self.phenos_base, self.towards_base, self.leg_base = None, None, None, None, None
        self.recon_motion_ep, self.pose_z_seq_ep, self.recon_pose_z_seq_ep, self.motion_z_ep = None, None, None, None
        self.motion_z_base, self.motion_z_ep_umap = None, None
        self.vid_draw_labels, self.vid_draw_data = None, None

    def process(self):

        self._concat_generator_batches()
        self._forward_pass()
        self._fit_and_transform_umap()
        self._save_for_interactive_plot()
        if self.draw:
            self._define_draw_data()
            self._draw_corresponding_videos()

    def _concat_generator_batches(self):
        # Lists for concatenation
        x_ep_list, tasks_ep_list, phenos_ep_list, towards_ep_list, leg_ep_list = [], [], [], [], []
        # Get data from data generator's first loop
        for train_data, test_data in self.data_gen.iterator():

            # x_fit for umap embedding
            x, x_masks, tasks, task_masks, phenos, pheno_masks, towards, leg, leg_masks = train_data
            masks = (task_masks == 1) & (pheno_masks == 1) & (leg_masks == 1)
            x, tasks, phenos, towards, leg = slice_by_mask(masks, x, tasks, phenos, towards, leg)

            # Produce phenos of equal/similar amounts
            uniphenos, phenos_counts = np.unique(phenos, return_counts=True)
            max_counts = np.sort(phenos_counts)[3]

            # Clap the maximum count of phenotype labels, s.t. certain label won't overrepresent the visualization
            for pheno_idx in range(13):
                x_ep, tasks_ep, phenos_ep, towards_ep, leg_ep = slice_by_mask(phenos == pheno_idx,
                                                                              x, tasks, phenos, towards, leg)
                x_ep_list.append(x_ep[0:max_counts, ])
                tasks_ep_list.append(tasks_ep[0:max_counts, ])
                phenos_ep_list.append(phenos_ep[0:max_counts, ])
                towards_ep_list.append(towards_ep[0:max_counts, ])
                leg_ep_list.append(leg_ep[0:max_counts, ])

            # Concatenate and prepare data
            x_ep = np.vstack(x_ep_list)
            tasks_ep, phenos_ep, towards_ep, leg_ep = [np.concatenate(eplist) for eplist in [
                tasks_ep_list, phenos_ep_list, towards_ep_list, leg_ep_list
            ]]

            # Shuffle
            np.random.seed(50)
            ran_vec = np.random.permutation(x_ep.shape[0])
            self.x_ep, self.tasks_ep, self.phenos_ep, self.towards_ep, self.leg_ep = slice_by_mask(
                ran_vec, x_ep, tasks_ep, phenos_ep, towards_ep, leg_ep
            )

            # Slice data for fitting base
            self.x_base, self.tasks_base = x[0:self.fit_samples_num, ], tasks[0:self.fit_samples_num, ]
            self.phenos_base, self.towards_base = phenos[0:self.fit_samples_num, ], towards[0:self.fit_samples_num, ]
            self.leg_bas = leg[0:self.fit_samples_num, ]

    def _define_model_inputs(self):
        # Convert labels to vector + from numpy to tensor
        towards_ep2d = expand1darr(self.towards_ep.astype(np.int), 3, self.model_container.seq_dim)
        towards_base2d = expand1darr(self.towards_base.astype(np.int), 3, self.model_container.seq_dim)
        x_ep, x_base, towards_ep2d_tensor, towards_base2d_tensor = numpy2tensor(
            self.model_container.device, self.x_ep, self.x_base, towards_ep2d, towards_base2d
        )

        ep_input = (x_ep, towards_ep2d_tensor)
        base_input = (x_base, towards_base2d_tensor)
        return ep_input, base_input

    def _forward_pass(self):

        # Convert labels to vector + from numpy to tensor
        ep_input, base_input = self._define_model_inputs()

        # Forward pass
        self.recon_motion_ep, self.pose_z_seq_ep, self.recon_pose_z_seq_ep, self.motion_z_ep = \
            self.model_container._forward_pass(*ep_input)

        self.recon_motion_base, self.pose_z_seq_base, self.recon_pose_z_seq_base, self.motion_z_base = \
            self.model_container._forward_pass(*base_input)

        # Convert output to numpy
        self.recon_motion_ep, self.pose_z_seq_ep, self.recon_pose_z_seq_ep, self.motion_z_ep = tensor2numpy(
            self.recon_motion_ep, self.pose_z_seq_ep, self.recon_pose_z_seq_ep, self.motion_z_ep
        )
        self.recon_motion_base, self.pose_z_seq_base, self.recon_pose_z_seq_base, self.motion_z_base = tensor2numpy(
            self.recon_motion_base, self.pose_z_seq_base, self.recon_pose_z_seq_base, self.motion_z_base
        )

    def _fit_and_transform_umap(self):

        motion_z_umapper = umap.UMAP(n_neighbors=15,
                                     n_components=2,
                                     min_dist=0.1,
                                     metric="euclidean")
        motion_z_umapper.fit(self.motion_z_ep)

        self.motion_z_ep_umap = motion_z_umapper.transform(self.motion_z_ep)

    def _save_for_interactive_plot(self):

        # Save arrays
        df = pd.DataFrame({"ori_motion": list(self.x_ep),
                           "recon_motion": list(self.recon_motion_ep),
                           "motion_z_umap": list(self.motion_z_ep_umap),
                           "phenotype": list(self.phenos_ep),
                           "task": list(self.tasks_ep),
                           "direction": list(self.towards_ep)
                           })
        write_df_pickle(df, os.path.join(self.save_data_dir, self.df_save_fn))

    def _draw_corresponding_videos(self):
        # Define and make directories
        save_videos_dir = os.path.join(self.save_data_dir, "videos", self.vid_dirname)
        os.makedirs(save_videos_dir, exist_ok=True)

        # Loop for each video
        num_vid = self.x_ep.shape[0]
        for vid_idx in range(num_vid):
            print("\rWriting vid {}/{}".format(vid_idx, num_vid), end="", flush=True)
            vwriter = skv.FFmpegWriter(os.path.join(save_videos_dir, "%s_%d.mp4" % (self.vid_dirname, vid_idx)))

            skepaint = SkeletonPainter(x=self.vid_draw_data[:, vid_idx, 0:25, :],
                                       y=self.vid_draw_data[:, vid_idx, 25:, :],
                                       texts=self.vid_draw_labels)
            for frame in skepaint.draw_multiple_skeletons():
                vwriter.writeFrame(frame)
            vwriter.close()

    def _define_draw_data(self):
        self.vid_draw_labels = ["Ori", "Recon"]
        self.vid_draw_data = np.stack([self.x_ep, self.recon_motion_ep])


class LatentSpaceSaver_CondDirectTask(LatentSpaceSaver_CondDirect):
    """
    Working with scripts.STVAE_run.CtaskSVAEmodel, with
        (1) direction conditionals,
        (2) task conditionals

    """

    def _define_model_inputs(self):
        # Convert labels to vector + from numpy to tensor
        towards_ep2d = expand1darr(self.towards_ep.astype(np.int), 3, self.model_container.seq_dim)
        towards_base2d = expand1darr(self.towards_base.astype(np.int), 3, self.model_container.seq_dim)
        tasks_ep2d = expand1darr(self.tasks_ep.astype(np.int), 8, self.model_container.seq_dim)
        tasks_base2d = expand1darr(self.tasks_base.astype(np.int), 8, self.model_container.seq_dim)

        x_ep, x_base, towards_ep2d, towards_base2d, tasks_ep2d, tasks_base2d = numpy2tensor(
            self.model_container.device, self.x_ep, self.x_base, towards_ep2d, towards_base2d, tasks_ep2d, tasks_base2d
        )

        # Conatenate between towards and tasks conditionals
        cond_ep = torch.cat((towards_ep2d, tasks_ep2d), dim=1)
        cond_base = torch.cat((towards_base2d, tasks_base2d), dim=1)

        ep_input = (x_ep, cond_ep)
        base_input = (x_base, cond_base)
        return ep_input, base_input
