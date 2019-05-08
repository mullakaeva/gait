import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import matplotlib.pyplot as plt
import skvideo.io as skv
from common.keypoints_format import openpose_body_draw_sequence
from .Model import VAE
from common.utils import RunningAverageMeter


def gen_spiral(a, b, speed, t_range, steps):
    t_start, t_end = t_range
    t = np.linspace(t_start, t_end, steps)
    theta = speed * t
    r = a + b * t
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def gen_lines(x_start=None, y_start=None, lim=(-2, 2), line_steps=500):
    if x_start is None:
        x = np.linspace(lim[0], lim[1], line_steps)
        y = np.zeros(x.shape) + y_start
    elif y_start is None:
        y = np.linspace(lim[0], lim[1], line_steps)
        x = np.zeros(y.shape) + x_start
    else:
        print("Only one, x_start or y_start is None")
        raise TypeError
    return x, y


def gen_paths(x_max, y_max, num_lines, num_steps=100):
    # Storage
    x_list = []
    y_list = []

    # Generate spiral
    # a, b = 0, -0.3
    # t_range = [0, 2 * np.pi]
    # speed = 7
    # steps_spiral = 500
    # x_spiral, y_spiral = gen_spiral(a, b, speed, t_range, steps_spiral)
    # x_list.append(x_spiral)
    # y_list.append(y_spiral)

    # Cross paths
    x_start_list = np.linspace(-x_max, x_max, num_lines)
    y_start_list = np.linspace(-y_max, y_max, num_lines)
    for x_start in x_start_list:
        x_temp, y_temp = gen_lines(x_start=x_start, lim=(-y_max, y_max), line_steps=num_steps)
        x_list.append(x_temp)
        y_list.append(y_temp)
    for y_start in y_start_list:
        x_temp, y_temp = gen_lines(y_start=y_start, lim=(-x_max, x_max), line_steps=num_steps)
        x_list.append(x_temp)
        y_list.append(y_temp)

    # Concatenate paths
    x = np.concatenate(x_list)
    y = np.concatenate(y_list)
    color_vals = np.linspace(0, 1, x.shape[0])
    return x, y, color_vals


class GaitVAEmodel:
    def __init__(self, data_gen,
                 input_dims=50,
                 latent_dims=2,
                 gpu=0,
                 step_lr_decay=0.8,
                 KLD_const=0.001,
                 save_chkpt_path=None):

        # Standard Init
        self.data_gen = data_gen
        self.input_dims = input_dims
        self.latent_dims = latent_dims
        self.device = torch.device('cuda:{}'.format(gpu))
        self.save_chkpt_path = save_chkpt_path

        # Learning rate decay, regularization_constant, loss
        self.step_lr_decay = step_lr_decay
        self.KLD_regularization_const = KLD_const
        self.weights_vec_loss = self._get_weights_vec_loss()
        self.loss_train_meter, self.loss_cv_meter = RunningAverageMeter(), RunningAverageMeter()

        # initialize model, params, optimizer
        self.model = VAE(input_dims=self.input_dims,
                         latent_dims=self.latent_dims).to(self.device)
        params = self.model.parameters()
        self.optimizer = optim.Adam(params, lr=0.001)

    def train(self, n_epochs=50):
        try:
            scheduler = StepLR(self.optimizer, step_size=1, gamma=self.step_lr_decay)
            for epoch in range(n_epochs):
                iter_idx = 0
                for (x, _), (x_test, _) in self.data_gen.iterator():
                    x = torch.from_numpy(x).float().to(self.device)
                    x_test = torch.from_numpy(x_test).float().to(self.device)
                    self.optimizer.zero_grad()

                    # Loss of CV set and training set
                    self.model.eval()
                    with torch.no_grad():
                        out_test, mu_test, logvar_test, z_test = self.model(x_test)
                        loss_test = self.loss_function(x_test, out_test, mu_test, logvar_test)
                        self.loss_cv_meter.update(loss_test.item())

                    # Training
                    self.model.train()
                    out, mu, logvar, z = self.model(x)
                    loss_train = self.loss_function(x, out, mu, logvar)
                    self.loss_train_meter.update(loss_train.item())

                    # Back-prop
                    loss_train.backward()
                    self.optimizer.step()

                    iter_idx += 1

                    # Print Info
                    print("\rEpoch %d/%d at iter %d/%d | Loss: %0.8f | CV Loss: %0.8f" % (epoch, n_epochs, iter_idx,
                                                                                          self.data_gen.num_rows / self.data_gen.m,
                                                                                          self.loss_train_meter.avg,
                                                                                          self.loss_cv_meter.avg)
                          , flush=True, end="")
                scheduler.step()
                self._save_model()

        except KeyboardInterrupt:
            self._save_model()

    def sample_from_latents(self, z_c):
        """

        Parameters
        ----------
        z_c : numpy.darray
            Conditioned latent encoding vectors, with shape (num_samples, num_latents + num_labels)

        Returns
        -------
        x : numpy.darray
            The reconstructed x-coordinates from latents,
            with shape (num_samples, num_keypoints, sequence_length), specifically by default, (512, 25, 128)
        y : numpy.darray
            Same as x, with it is the y-coordinates
        """
        z_c_tensor = torch.from_numpy(z_c).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model.decode(z_c_tensor)
        out_np = out.cpu().numpy()
        x = out_np[:, 0:self.data_gen.keyps_x_dims, ]
        y = out_np[:, self.data_gen.keyps_x_dims:self.data_gen.total_fea_dims, ]
        return x, y

    def load_model(self, load_chkpt_path):
        checkpoint = torch.load(load_chkpt_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss_train_meter = checkpoint['loss_train_meter']
        self.loss_cv_meter = checkpoint['loss_cv_meter']
        self.KLD_regularization_const = checkpoint['KLD_regularization_const']
        print('Loaded ckpt from {}'.format(load_chkpt_path))

    def _save_model(self):
        if self.save_chkpt_path is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss_train_meter': self.loss_train_meter,
                'loss_cv_meter': self.loss_cv_meter,
                "KLD_regularization_const" : self.KLD_regularization_const
            }, self.save_chkpt_path)
            print('Stored ckpt at {}'.format(self.save_chkpt_path))

    def _get_weights_vec_loss(self):
        weights_vec = torch.ones(1, 50).float().to(self.device)
        weights_vec[0, 19:25] = 2
        weights_vec[0, 19*2:] = 2
        return weights_vec

    def loss_function(self, x, pred, mu, logvar):
        img_loss = torch.sum(self.weights_vec_loss * ((x-pred)**2))
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = img_loss + self.KLD_regularization_const*KLD
        return loss


class GaitSingleSkeletonVAEvisualiser:
    def __init__(self, data_gen, load_model_path, save_vid_dir, latent_dims=2):
        # Hard coded stuff
        self.num_samples_pred = 5
        self.num_samples_latents = 3
        self.latents_dim, self.label_dim = 2, 0
        self.times = 128

        # Init
        self.data_gen = data_gen
        self.load_model_path = load_model_path
        self.save_vid_dir = save_vid_dir
        self.model_container = GaitVAEmodel(data_gen, latent_dims=latent_dims)
        self.model_container.load_model(self.load_model_path)

    def visualise_vid(self):
        # Init
        os.makedirs(self.save_vid_dir, exist_ok=True)
        (x_in, y_in), (x_out, y_out), labels, mu = self._get_pred_results()

        labels_expanded = np.repeat(labels.reshape(-1,1), self.times, axis=1)
        labels_flattened = labels_expanded.flatten()
        z_x, z_y = mu[:, 0, :], mu[:, 1, :]
        z_space_flattened = np.concatenate((z_x.flatten().reshape(-1, 1), z_y.flatten().reshape(-1,1)), axis=1)
        # labels ~ (m, )
        # mu ~ (m, 2, 128)

        KLD_const = self.model_container.KLD_regularization_const

        for sample_idx in range(self.num_samples_pred):

            save_vid_path = os.path.join(self.save_vid_dir,
                                         "Seq%d_KLD-%f.mp4"% (sample_idx,
                                                              KLD_const))
            vwriter = skv.FFmpegWriter(save_vid_path)
            for t in range(self.times):
                time = t / 25
                print("\rNow writing KLD-%f | Recon_sample-%d | time-%0.3fs" % (KLD_const, sample_idx, time),
                      flush=True, end="")
                title_in = "KLD-%f | Recon_sample-%d | time-%0.3fs" % (KLD_const, sample_idx, time)
                draw_arr_in = plot2arr_skeleton(x_in[sample_idx, :, t],
                                                y_in[sample_idx, :, t],
                                                title_in,
                                                x_lim=(-0.6, 0.6),
                                                y_lim=(0.6, -0.6))
                title_latent = "Latent"
                draw_arr_latent = plot2arr_latent_space(mu[sample_idx, :, :].T, t, title_latent,
                                                        x_lim=(-5, 5),
                                                        y_lim=(-5, 5),
                                                        z_info=(z_space_flattened, labels_flattened))

                draw_arr_out = plot2arr_skeleton(x_out[sample_idx, :, t],
                                                y_out[sample_idx, :, t],
                                                title_in,
                                                x_lim=(-0.6, 0.6),
                                                y_lim=(0.6, -0.6))
                h, w = draw_arr_in.shape[0], draw_arr_in.shape[1]
                output_arr = np.zeros((h * 2, w * 2, 3))
                output_arr[0:h, 0:w, :] = draw_arr_in
                output_arr[h:h * 2, 0:w, :] = draw_arr_out
                output_arr[0:h, w:w * 2, :] = draw_arr_latent
                vwriter.writeFrame(output_arr)
            print()
            vwriter.close()

    def visualise_latent_space(self):
        # Randomly sample the data to visualze the latent-label distribution
        z_space, labels_space = self._sample_collapsed_data()
        x_max, y_max = np.quantile(np.abs(z_space[:, 0]), 0.99), np.quantile(np.abs(z_space[:, 1]), 0.99)
        num_scale = int(np.mean([x_max,y_max]))+1
        x_skeleton, y_skeleton, latents, _ = self._get_latents_results(x_max=x_max,
                                                                       y_max=y_max,
                                                                       num_lines=num_scale*4,
                                                                       num_steps=200)
        num_sample = x_skeleton.shape[0]
        save_vid_path = os.path.join(self.save_vid_dir,
                                     "visualize_latent_space_%f.mp4" % self.model_container.KLD_regularization_const)
        vwriter = skv.FFmpegWriter(save_vid_path)
        for sample_idx in range(num_sample):
            print("\rDrawing %d/%d" % (sample_idx, num_sample), flush=True, end="")
            title = "Time = %0.4f" % (sample_idx / 25)
            ske_arr = plot2arr_skeleton(x_skeleton[sample_idx, :],
                                        y_skeleton[sample_idx, :],
                                        title)
            lataent_arr = plot2arr_latent_space(latents, sample_idx, title,
                                                x_lim=(-x_max, x_max),
                                                y_lim=(-y_max, y_max),
                                                z_info=(z_space, labels_space))

            output_arr = np.concatenate((ske_arr, lataent_arr), axis=1)
            vwriter.writeFrame(output_arr)
        vwriter.close()

    def _get_pred_results(self):
        for (x_train, labels_train), (in_test, labels_test) in self.data_gen.iterator():
            # Flatten data
            n_samples, n_times = x_train.shape[0], x_train.shape[2]
            in_test = np.transpose(x_train, (0, 2, 1))
            in_test = in_test[:, :, 0:50].reshape(-1, 50)

            # Forward pass
            self.model_container.model.eval()
            with torch.no_grad():
                in_test, labels_test = torch.from_numpy(in_test).float().to(self.model_container.device), \
                                       torch.from_numpy(labels_test).float().to(self.model_container.device)
                out_test, mu, logvar, z = self.model_container.model(in_test)
                # loss = total_loss(in_test, out_test, mu, logvar)
            break

        # Unflatten data
        in_test_np, out_test_np = in_test.cpu().numpy(), out_test.cpu().numpy()
        in_test_np, out_test_np = in_test_np.reshape(n_samples, n_times, 50), out_test_np.reshape(n_samples, n_times,
                                                                                                  50)
        in_test_np, out_test_np = np.transpose(in_test_np, (0, 2, 1)), np.transpose(out_test_np, (0, 2, 1))
        mu_np = z.cpu().numpy().reshape(n_samples, n_times, -1)
        mu_np = np.transpose(mu_np, (0, 2, 1))
        x_in, y_in = in_test_np[:, 0:25, :], in_test_np[:, 25:50, :]
        x_out, y_out = out_test_np[:, 0:25, :], out_test_np[:, 25:50, :]
        return (x_in, y_in), (x_out, y_out), labels_train, mu_np

    def _get_latents_results(self, x_max, y_max, num_lines, num_steps):
        x_paths, y_paths, color_vals = gen_paths(x_max, y_max, num_lines, num_steps)
        z_paths = np.vstack((x_paths, y_paths)).T  # (num_samples, 2)
        # z = torch.from_numpy(z_paths).to(self.model_container.device)
        x_tensor, y_tensor = self.model_container.sample_from_latents(z_paths)

        return x_tensor, y_tensor, z_paths, color_vals

    def _sample_collapsed_data(self):
        for (data_train, labels_train), (data_test, labels_test) in self.data_gen.iterator():
            # Forward pass
            self.model_container.model.eval()
            with torch.no_grad():
                in_test = torch.from_numpy(data_train).float().to(self.model_container.device)
                out_test, mu, logvar, z = self.model_container.model(in_test)
            break
        z = z.cpu().numpy()
        return z, labels_train


class GaitSingleSkeletonVAEvisualiserCollapsed(GaitSingleSkeletonVAEvisualiser):
    def visualise_vid(self):
        # Init
        os.makedirs(self.save_vid_dir, exist_ok=True)

        # Visualise in-out pair
        for sample_num in range(self.num_samples_pred):
            (x_in, y_in), (x_out, y_out) = self._get_pred_results()

            save_vid_path = os.path.join(self.save_vid_dir, "Recon-%d.mp4" % (sample_num))
            vwriter = skv.FFmpegWriter(save_vid_path)
            for t in range(self.times):
                time = t / 25
                print("\rNow writing Recon_sample-%d | time-%0.4fs" % (sample_num, time), flush=True, end="")
                title_in = "Input: %d | Time = %0.4fs" % (sample_num, time)
                draw_arr_in = plot2arr_skeleton(x_in, y_in, 0, t, title_in)
                title_out = "Output: %d | Time = %0.4fs" % (sample_num, time)
                draw_arr_out = plot2arr_skeleton(x_out, y_out, 0, t, title_out)

                draw_arr_recon = np.concatenate((draw_arr_in, draw_arr_out), axis=1)
                vwriter.writeFrame(draw_arr_recon)
            print()
            vwriter.close()

    def _get_pred_results(self):
        for (data_train, data_test) in self.data_gen.iterator():
            data = data_train[0:self.times, ]

            # Forward pass
            self.model_container.model.eval()
            with torch.no_grad():
                in_test = torch.from_numpy(data).float().to(self.model_container.device)
                out_test, mu, logvar, z = self.model_container.model(in_test)
                loss = self.model_container.loss_function(in_test, out_test, mu, logvar)
            break

        out_test_np = out_test.cpu().numpy()
        x_in, y_in = np.transpose(data[:, :, 0:25], (1, 2, 0)), np.transpose(data[:, :, 25:], (1, 2, 0))
        x_out, y_out = np.transpose(out_test_np[:, :, 0:25], (1, 2, 0)), np.transpose(out_test_np[:, :, 25:], (1, 2, 0))
        return (x_in, y_in), (x_out, y_out)


def plot2arr_skeleton(x, y, title, x_lim=(-0.6, 0.6), y_lim=(0.6, -0.6)):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax = draw_skeleton(ax, x, y)
    fig.suptitle(title)
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
    fig.tight_layout()
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot2arr_latents(z, title, x_lim=(-1, 1), y_lim=(-1, 1)):
    fig, ax = plt.subplots()
    ax.scatter(z[0], z[1], marker="x")
    fig.suptitle(title)
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
    fig.tight_layout()
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot2arr_latent_space(z, sample_idx, title, x_lim=(-1, 1), y_lim=(-1, 1), z_info=None):
    """
    Parameters
    ----------
    z : numpy.darray
        Latent vectors with shape (num, 2).
    title : str
        Title of the plot
    x_lim : tuple
        Limits of the x_axix
    y_lim : tuple
        Limits of the y_axix

    Returns
    -------
    data : numpy.darray
        Output drawn array
    """
    fig, ax = plt.subplots()
    # Scatter all vectors
    if z_info is not None:
        z_space, z_labels = z_info
        im_space = ax.scatter(z_space[:, 0], z_space[:, 1], c=z_labels, cmap="hsv", marker=".", alpha=0.5)
        fig.colorbar(im_space)

    # Scatter current vectors
    ax.scatter(z[sample_idx, 0], z[sample_idx, 1], c="k", marker="x")

    # Title, limits and drawing
    fig.suptitle(title)
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
    fig.tight_layout()
    fig.canvas.draw()

    # Data
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def draw_skeleton(ax, x, y):
    side_dict = {
        "m": "k",
        "l": "r",
        "r": "b"
    }
    for start, end, side in openpose_body_draw_sequence:
        ax.plot(x[[start, end]], y[[start, end]], c=side_dict[side])
    return ax
