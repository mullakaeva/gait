from .utils import read_preprocessed_keypoints, fullfile, idx2task, idx2task_dict, idx2pheno, pool_points, \
    write_df_pickle
from .keypoints_format import openpose_body_draw_sequence, excluded_points, draw_seq_col_indexes
from glob import glob
import numpy as np
import skvideo.io as skv
import skimage.io as ski
import pandas as pd
from skimage.color import rgba2rgb
import torch
import matplotlib.pyplot as plt
import os
import umap


def build_frame_2by2(*args):
    h, w, _ = args[0].shape
    output_arr = np.zeros((h * 2, w * 2, 3))
    iter_idx, max_iter_idx = 0, len(args) - 1
    for i in range(2):
        for j in range(2):
            if iter_idx <= max_iter_idx:
                output_arr[h * i: h * (i + 1), w * j: w * (j + 1), :] = args[iter_idx]
            iter_idx += 1

    return output_arr


def build_frame_2by3(*args):
    """
    Combine 6 two-dimensional frames with identical shape to a single array (2x3 grid). Starting left to right, top to bottom.

    Parameters
    ----------
    args : numpy.darray
        input frames, each with (h, w, c) shape. Number of arguments <= 6

    Returns
    -------
    output_arr : numpy.darray
        Array combining the <=6 input frames.
    """
    h, w, _ = args[0].shape
    output_arr = np.zeros((h * 2, w * 3, 3))
    iter_idx, max_iter_idx = 0, len(args) - 1
    for i in range(2):
        for j in range(3):
            if iter_idx <= max_iter_idx:
                output_arr[h * i: h * (i + 1), w * j: w * (j + 1), :] = args[iter_idx]
            iter_idx += 1
    return output_arr

def draw_skeleton_new(ax, x, y, linewidth=1):
    for seq_indexes in draw_seq_col_indexes:
        ax.plot(x[seq_indexes[0]], y[seq_indexes[0]], c=seq_indexes[1], linewidth=linewidth)
    return ax


def draw_skeleton(ax, x, y, linewidth=1):
    side_dict = {
        "m": "0",
        "l": "r",
        "r": "b"
    }
    for start, end, side in openpose_body_draw_sequence:
        ax.plot(x[[start, end]], y[[start, end]], c=side_dict[side], linewidth=linewidth)
    return ax


def plot2arr_skeleton(x, y, title, x_lim=(-0.6, 0.6), y_lim=(-0.6, 0.6), show_axis=True):
    fig, ax = plt.subplots()
    ax.scatter(np.delete(x, excluded_points), np.delete(y, excluded_points))
    ax = draw_skeleton(ax, x, y)
    fig.suptitle(title)
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[1], y_lim[0])
    if show_axis != True:
        ax.axis("off")
    fig.tight_layout()
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def gen_single_vid_two_skeleton_motion(x, recon, save_vid_path, x_lim=(-0.5, 0.5), y_lim=(-0.5, 0.5)):
    """

    Parameters
    ----------
    x : numpy.darray
        With shape (50, seq_length=128)
    recon : numpy.darray
        With shape (50, seq_length=128)
    save_vid_path : str
        Path of video which is to be produced

    Returns
    -------
    None

    """
    vwrtier = skv.FFmpegWriter(save_vid_path)
    for t in range(x.shape[1]):
        draw_x = plot2arr_skeleton(x=x[0:25, t], y=x[25:, t], title="", x_lim=x_lim, y_lim=y_lim, show_axis=False)
        draw_recon = plot2arr_skeleton(x=recon[0:25, t], y=recon[25:, t], title="", x_lim=x_lim, y_lim=y_lim,
                                       show_axis=False)
        final_frame = np.concatenate([draw_x[:, 150:530, :], draw_recon[:, 150:530, :]], axis=1)
        vwrtier.writeFrame(final_frame)
    vwrtier.close()


def plot2arr_skeleton_multiple_samples(x, y, motion_z_umap, labels, title, x_lim=(-0.6, 0.6), y_lim=(-0.6, 0.6),
                                       label_type="task"):
    """

    Parameters
    ----------
    x : numpy.darray
        With shape = (num_samples, 25)
    y : numpy.darray
        With shape = (num_samples, 25)
    motion_z_umap : numpy.darray
        With shape = (num_samples, 2)
    labels : numpy.darray
        With shape = (num_samples, )
    title
    x_lim
    y_lim
    label_type : str
        Either "task" or "pheno"
    Returns
    -------

    """
    if label_type == "task":
        num_classes = 8
        idx_func = idx2task
    elif label_type == "pheno":
        num_classes = 13
        idx_func = idx2pheno

    fig, ax = plt.subplots(figsize=(12, 8))
    num_samples = x.shape[0]
    for i in range(num_samples):
        ax = draw_skeleton(ax, x[i,], y[i,])

    # Scatter color spots for indicating the labels
    im_space = ax.scatter(motion_z_umap[:, 0], motion_z_umap[:, 1], c=labels, cmap="hsv")
    cbar = plt.colorbar(im_space)
    cbar.set_ticks([x for x in range(num_classes)])
    cbar.set_ticklabels([idx_func(x) for x in range(num_classes)])

    fig.suptitle(title)
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[1], y_lim[0])
    fig.tight_layout()
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot_latent_space_with_labels(z_space, z_labels, title, x_lim=None, y_lim=None, alpha=0.5, target_scatter=None,
                                  figsize=(6.4, 4.8), label_type="task"):
    if label_type == "task":
        num_class = 8
        idx_func = idx2task
    elif label_type == "pheno":
        num_class = 13
        idx_func = idx2pheno

    fig, ax = plt.subplots(figsize=figsize)

    # Scatter all vectors
    im_space = ax.scatter(z_space[:, 0], z_space[:, 1], c=z_labels, cmap="hsv", marker=".", alpha=alpha)
    cbar = plt.colorbar(im_space)
    cbar.set_ticks([x for x in range(num_class)])
    cbar.set_ticklabels([idx_func(x) for x in range(num_class)])

    # Draw a specific scatter point
    if target_scatter is not None:
        ax.scatter(target_scatter[0], target_scatter[1], c="k", marker="x", s=400)

    # Title, limits and drawing
    if x_lim is not None:
        ax.set_xlim(x_lim[0], x_lim[1])
    if y_lim is not None:
        ax.set_ylim(y_lim[0], y_lim[1])
    fig.suptitle(title)
    fig.canvas.draw()

    # Convert to numpy array
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return data


def plot_umap_with_labels(z, labels, title, z_base=None, alphas=[0.1, 0.25], label_type="task"):
    """

    Parameters
    ----------
    z : numpy.darray
        With shape (n_samples, n_features=2). Only the first 2 features will be used.
    labels : numpy.darray
        With shape (n_samples,) values ranged  [0, 7] (integer or rounded float)
    title : string
        Tittle of the plot
    alphas : iterable
        [alpha_this_class, alpha_other_classes]
    label_type : str
        Either "task" or "pheno"

    Returns
    -------

    """

    if label_type == "task":
        figure_grid = (2, 4)
        num_classes = 8
        idx_func = idx2task
    elif label_type == "pheno":
        figure_grid = (4, 4)
        num_classes = 13
        idx_func = idx2pheno

    fig, ax = plt.subplots(figure_grid[0], figure_grid[1], figsize=(14, 7))
    ax = ax.ravel()
    for class_idx in range(figure_grid[0] * figure_grid[1]):
        if class_idx < num_classes:
            if z_base is not None:
                ax[class_idx].scatter(z_base[:, 0], z_base[:, 1], c="0.1", marker=".", alpha=alphas[1])
            embed_this_class = z[labels.astype(int) == class_idx, :]
            embed_other_classes = z[labels.astype(int) != class_idx, :]
            ax[class_idx].scatter(embed_other_classes[:, 0], embed_other_classes[:, 1], c="0.1", marker=".",
                                  alpha=alphas[1])
            ax[class_idx].scatter(embed_this_class[:, 0], embed_this_class[:, 1], c="r", marker=".", alpha=alphas[0])
            ax[class_idx].set_title("{}".format(idx_func(class_idx)))
            ax[class_idx].axis("off")
        else:
            ax[class_idx].axis("off")

    # Title, limits and drawing
    fig.suptitle(title)
    fig.canvas.draw()

    # Convert to numpy array
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return data


def gen_motion_space_scatter_animation(recon_motion, motion_z_umap, labels, kernel_size,
                                       translation_scaling,
                                       skeleton_scaling=1):
    """

    Parameters
    ----------
    recon_motion : numpy.darray
        With shape = (n_samples, 50, seq)
    motion_z_umap : numpy.darray
        With shape = (n_samples, 2)
    labels : numpy.darray
        With shape = (n_samples, )
    translation_scaling : int or float
    skeleton_scaling : int or float
    kernel_size : int or float

    Returns
    -------

    """

    # Normalization and pooling
    motion_z_umap_scaled = (motion_z_umap - np.mean(motion_z_umap, axis=0)) * translation_scaling
    motion_z_umap_pooled, indexes = pool_points(motion_z_umap_scaled, kernel_size=kernel_size)

    # Select pooled samples
    recon_motion_pooled = recon_motion[indexes,]
    labels_pooled = labels[indexes,]

    # Scaling and assignments
    recon_motion_pooled = recon_motion_pooled * skeleton_scaling
    recon_motion_pooled[:, 0:25, :] += motion_z_umap_pooled[:, 0].reshape(-1, 1, 1)
    recon_motion_pooled[:, 25:, :] += motion_z_umap_pooled[:, 1].reshape(-1, 1, 1)
    return recon_motion_pooled, motion_z_umap_pooled, labels_pooled


class LatentSpaceVideoVisualizer:
    def __init__(self, model_identifier, save_vid_dir, seq_length=128):
        self.pose_umapper, self.motion_z_umapper = None, None
        self.save_vid_dir = save_vid_dir
        self.model_identifier = model_identifier
        self.seq_length = seq_length

    def fit_umap(self, pose_z_seq, motion_z, num_samples_pose_z_seq=128):
        motion_z_fit = motion_z.cpu().detach().numpy()  # (n_samples, motion_latents_dim)
        pose_z_seq_fit = pose_z_seq.cpu().detach().numpy()

        if num_samples_pose_z_seq is not None:
            pose_z_seq_fit = pose_z_seq_fit[0: num_samples_pose_z_seq, ]

        pose_z_flat_fit = self.seq2flat(pose_z_seq_fit)

        self.pose_umapper = umap.UMAP(n_neighbors=15,
                                      n_components=2,
                                      min_dist=0.1,
                                      metric="euclidean")
        self.motion_z_umapper = umap.UMAP(n_neighbors=15,
                                          n_components=2,
                                          min_dist=0.1,
                                          metric="euclidean")

        self.pose_umapper.fit(pose_z_flat_fit)
        self.motion_z_umapper.fit(motion_z_fit)

    def gen_umap_plot(self, motion_z_umap, pose_z_flat_umap, labels, labels_flat, mode, label_type, test_acc,
                      pose_dim=16,
                      motion_dim=128,
                      motion_z_umap_base=None):

        umap_plot_pose_arr = plot_umap_with_labels(pose_z_flat_umap, labels_flat,
                                                   title="Pose: {} | test acc: {} \nModel: {}".format(
                                                       pose_dim, test_acc,
                                                       self.model_identifier), label_type=label_type)
        umap_plot_motion_arr = plot_umap_with_labels(motion_z_umap, labels,
                                                     title="Motion: {} | test acc: {}\nModel: {}".format(
                                                         motion_dim, test_acc,
                                                         self.model_identifier),
                                                     alphas=[0.35, 0.1], label_type=label_type,
                                                     z_base=motion_z_umap_base)

        ski.imsave(
            os.path.join(self.save_vid_dir, "{}_{}_UmapPose_{}.png".format(mode, label_type, self.model_identifier)),
            umap_plot_pose_arr)
        ski.imsave(
            os.path.join(self.save_vid_dir, "{}_{}_UmapMotion_{}.png".format(mode, label_type, self.model_identifier)),
            umap_plot_motion_arr)

    def gen_reconstruction_vid(self, x, recon_motion, motion_z_umap, pose_z_flat_umap, recon_pose_z_flat_umap,
                               pose_z_seq_shape, labels,
                               labels_flat, pred_labels, mode, label_type="task", sample_num=10):
        if label_type == "task":
            idx_func = idx2task
        elif label_type == "pheno":
            idx_func = idx2pheno

        # Draw videos
        for sample_idx in range(sample_num):

            save_vid_path = os.path.join(self.save_vid_dir,
                                         "{}_{}_ReconVid-{}_{}.mp4".format(mode, label_type, sample_idx,
                                                                           self.model_identifier))
            vwriter = skv.FFmpegWriter(save_vid_path)

            draw_motion_latents = plot_latent_space_with_labels(motion_z_umap[:, 0:2], labels,
                                                                title="Motion latents",
                                                                target_scatter=motion_z_umap[sample_idx, 0:2],
                                                                alpha=0.5,
                                                                label_type=label_type)
            pose_z_umap_flat2seq = self.flat2seq(pose_z_flat_umap, pose_z_seq_shape)
            recon_pose_z_flat_umap_flat2seq = self.flat2seq(recon_pose_z_flat_umap, pose_z_seq_shape)

            # Draw input & output skeleton for every time step
            for t in range(self.seq_length):
                time = t / 25
                print("\rNow writing %s Recon_sample-%d | time-%0.4fs" % (mode, sample_idx, time), flush=True, end="")
                draw_arr_in = plot2arr_skeleton(x=x[sample_idx, 0:25, t],
                                                y=x[sample_idx, 25:, t],
                                                title="%s %d | %s | GT = %s" % (
                                                    mode, sample_idx, self.model_identifier,
                                                    idx_func(labels[sample_idx]))
                                                )

                draw_arr_out = plot2arr_skeleton(x=recon_motion[sample_idx, 0:25, t],
                                                 y=recon_motion[sample_idx, 25:, t],
                                                 title=" Recon %s %d | Pred = %s" % (
                                                     mode, sample_idx,
                                                     idx_func(pred_labels[sample_idx]))
                                                 )

                draw_pose_latents = plot_latent_space_with_labels(pose_z_flat_umap, labels_flat,
                                                                  title="pose latent",
                                                                  alpha=0.2,
                                                                  target_scatter=pose_z_umap_flat2seq[sample_idx, 0:2,
                                                                                 t],
                                                                  label_type=label_type)
                draw_recon_pose_latents = plot_latent_space_with_labels(recon_pose_z_flat_umap, labels_flat,
                                                                        title="pose latent (Recon)",
                                                                        alpha=0.2,
                                                                        target_scatter=recon_pose_z_flat_umap_flat2seq[
                                                                                       sample_idx, 0:2,
                                                                                       t],
                                                                        label_type=label_type)
                output_frame = build_frame_2by3(draw_arr_in, draw_arr_out, draw_motion_latents, draw_pose_latents
                                                , draw_recon_pose_latents)
                vwriter.writeFrame(output_frame)
                plt.close()
            print()
            vwriter.close()
        pass

    def gen_latent_space_animation(self, recon_motion, motion_z_umap, labels, mode, label_type="task"):

        vreader_latent_motion_space = skv.FFmpegWriter(
            os.path.join(self.save_vid_dir, "{}_{}_latent_motion_space.mp4".format(mode, label_type)))

        recon_pooled, motion_z_umap_pooled, labels_pooled = gen_motion_space_scatter_animation(
            recon_motion=recon_motion,
            motion_z_umap=motion_z_umap,
            labels=labels,
            kernel_size=0.5,
            translation_scaling=1,
            skeleton_scaling=0.6)
        min_x, max_x = np.min(recon_pooled[:, 0:25, :]), np.max(recon_pooled[:, 0:25, :])
        min_y, max_y = np.min(recon_pooled[:, 25:, :]), np.max(recon_pooled[:, 25:, :])

        for t in range(self.seq_length):
            print("\rDrawing latent motion space {}/{}".format(t, self.seq_length), flush=True, end="")

            latent_motion_arr = plot2arr_skeleton_multiple_samples(x=recon_pooled[:, 0:25, t],
                                                                   y=recon_pooled[:, 25:, t],
                                                                   motion_z_umap=motion_z_umap_pooled,
                                                                   labels=labels_pooled,
                                                                   title="Latent Motion Space",
                                                                   x_lim=(min_x, max_x),
                                                                   y_lim=(min_y, max_y),
                                                                   label_type=label_type)
            vreader_latent_motion_space.writeFrame(latent_motion_arr)

        print()
        vreader_latent_motion_space.close()

    def visualization_wrapper(self, x, recon_motion, labels, pred_labels, motion_z_umap, pose_z_seq, recon_pose_z_seq,
                              test_acc, mode, plotting_mode=[True, True, True], num_samples_pose_z_seq=128,
                              sample_num=10, label_type="task", motion_z_base=None, save_vis_data=None):

        print("Visulizing in mode {} and label {}".format(mode, label_type))

        # Convert to numpy
        x = x.cpu().detach().numpy()
        recon_motion = recon_motion.cpu().detach().numpy()
        if motion_z_base is not None:
            motion_z_base = motion_z_base.cpu().detach().numpy()
        pose_z_seq = pose_z_seq.cpu().detach().numpy()
        recon_pose_z_seq = recon_pose_z_seq.cpu().detach().numpy()
        if label_type == "task":
            pred_labels = pred_labels.cpu().detach().numpy()
            pred_labels = np.argmax(pred_labels, axis=1)

        if num_samples_pose_z_seq is not None:
            pose_z_seq = pose_z_seq[0: num_samples_pose_z_seq, ]
            recon_pose_z_seq = recon_pose_z_seq[0: num_samples_pose_z_seq, ]

        # Changing shapes
        pose_z_flat, recon_pose_z_flat = self.seq2flat(pose_z_seq), self.seq2flat(recon_pose_z_seq)
        labels_flat = np.repeat(labels[0:pose_z_seq.shape[0], np.newaxis], self.seq_length, axis=1)
        labels_flat = labels_flat.reshape(-1)

        # Convert to umap space
        if motion_z_base is not None:
            motion_z_umap_base = self.motion_z_umapper.transform(motion_z_base)
        else:
            motion_z_umap_base = None

        pose_z_flat_umap = self.pose_umapper.transform(pose_z_flat)
        recon_pose_z_flat_umap = self.pose_umapper.transform(recon_pose_z_flat)

        # Visualization modules
        if plotting_mode[0]:
            self.gen_umap_plot(motion_z_umap=motion_z_umap,
                               pose_z_flat_umap=pose_z_flat_umap,
                               labels=labels, labels_flat=labels_flat,
                               mode=mode, test_acc=test_acc, label_type=label_type,
                               motion_z_umap_base=motion_z_umap_base)
        if plotting_mode[1]:
            self.gen_reconstruction_vid(x=x, recon_motion=recon_motion, motion_z_umap=motion_z_umap,
                                        pose_z_flat_umap=pose_z_flat_umap,
                                        recon_pose_z_flat_umap=recon_pose_z_flat_umap,
                                        pose_z_seq_shape=pose_z_seq.shape, labels=labels, labels_flat=labels_flat,
                                        pred_labels=pred_labels,
                                        mode=mode, sample_num=sample_num, label_type=label_type)
        if plotting_mode[2]:
            self.gen_latent_space_animation(recon_motion=recon_motion, motion_z_umap=motion_z_umap,
                                            labels=labels, mode=mode, label_type=label_type)

    @staticmethod
    def seq2flat(x_seq):
        """
        Convert (a, b, c) to (a, c, b), then (a * c, b)
        Parameters
        ----------
        x_seq : numpy.darray
            With shape (a, b, c), i.e. (num_frames, latent_dimension, seq_length)
        Returns
        -------
        x_flat : numpy.darray
            With shape (a * c, b)
        """
        return np.transpose(x_seq, (0, 2, 1)).reshape(x_seq.shape[0] * x_seq.shape[2], -1)

    @staticmethod
    def flat2seq(x_flat, x_seq_shape):
        """
        Convert (a * c, b) to (a, c, b), then (a, b, c)

        Parameters
        ----------
        x_flat : numpy.darray
            With shape (a * c, b )
        x_seq_shape : tuple
            x_seq_shape = (a, b, c)

        Returns
        -------
        x_seq : numpy.darray
            With shape (a * c, b)

        """
        return np.transpose(x_flat.reshape(x_seq_shape[0], x_seq_shape[2], -1), (0, 2, 1))


class SkeletonPainter:
    def __init__(self, x, y, texts, sep_x=0.4, y_lim=[-0.6, 0.6]):
        """

        Parameters
        ----------
        x : numpy.darray
            It has shape (m, 25, 128). x-coordinates of the walking sequence of 25 joints
        y : numpy.darray
            It has shape (m, 25, 128). y-coordinates of ^
        texts : tuple or list or iterable
            It should has length m
        sep_x : float
            Separation between the centers of skeletons on x-axis
        y_lim : tuple or list

        """

        self.x, self.y = x, y
        self.texts = texts
        self.num_skeletons = self.x.shape[0]
        self.sep_x = sep_x
        self.translation_x_vec = np.array([self.sep_x * i for i in range(self.num_skeletons)]).reshape(-1, 1, 1)
        self.text_y = -0.5
        self.x_trans = self.x + self.translation_x_vec
        self.y_lim = [y_lim[1], y_lim[0]]

    def draw_multiple_skeletons(self):

        for t in range(128):
            fig, ax = plt.subplots()
            ax.set_ylim(self.y_lim[0], self.y_lim[1])
            ax.axis("off")
            for i in range(self.num_skeletons):
                ax = draw_skeleton_new(ax, self.x_trans[i, :, t], self.y[i, :, t])
                ax.text(self.translation_x_vec[i, 0, 0], self.text_y, "{}".format(self.texts[i]))

            fig.tight_layout()
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            yield data