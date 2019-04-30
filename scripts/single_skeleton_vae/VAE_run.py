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


class GaitVAEmodel:
    def __init__(self, data_gen,
                 input_dims=50,
                 latent_dims=2,
                 gpu=0,
                 step_lr_decay=0.8,
                 save_chkpt_path=None):

        # Standard Init
        self.data_gen = data_gen
        self.input_dims = input_dims
        self.latent_dims = latent_dims
        self.device = torch.device('cuda:{}'.format(gpu))
        self.save_chkpt_path = save_chkpt_path

        # Learning rate decay and loss
        self.step_lr_decay = step_lr_decay
        self.recon_loss = self._set_up_loss_func()
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
                for (x, x_test) in self.data_gen.iterator():
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
        x = out_np[:, 0:self.data_gen.keyps_x_dims, :]
        y = out_np[:, self.data_gen.keyps_x_dims:self.data_gen.total_fea_dims, :]
        return x, y

    def load_model(self, load_chkpt_path):
        checkpoint = torch.load(load_chkpt_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss_train_meter = checkpoint['loss_train_meter']
        self.loss_cv_meter = checkpoint['loss_cv_meter']
        print('Loaded ckpt from {}'.format(load_chkpt_path))

    def _save_model(self):
        if self.save_chkpt_path is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss_train_meter': self.loss_train_meter,
                'loss_cv_meter': self.loss_cv_meter
            }, self.save_chkpt_path)
            print('Stored ckpt at {}'.format(self.save_chkpt_path))

    def _set_up_loss_func(self):
        return nn.MSELoss(reduction="sum")

    def loss_function(self, x, pred, mu, logvar):
        img_loss = self.recon_loss(x, pred)
        # return img_loss
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = img_loss + KLD
        return loss


class GaitSingleSkeletonVAEvisualiser:
    def __init__(self, data_gen, load_model_path, save_vid_dir, latent_dims=2):
        # Hard coded stuff
        self.num_samples_pred = 2
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

        # Visualise in-out pair
        for label_num in range(8):
            label_mask = np.argmax(labels, axis=1) == label_num
            x_in_labelled, y_in_labelled = x_in[label_mask,], y_in[label_mask,]
            x_out_labelled, y_out_labelled = x_out[label_mask,], y_out[label_mask,]
            mu_labelled = mu[label_mask,]
            for sample_num in range(self.num_samples_pred):
                save_vid_path = os.path.join(self.save_vid_dir, "Label-%d_Recon-%d.mp4" % (label_num, sample_num))
                vwriter = skv.FFmpegWriter(save_vid_path)
                for t in range(self.times):
                    time = t / 25
                    print("\rNow writing label-%d | Recon_sample-%d | time-%0.4fs" % (label_num, sample_num, time),
                          flush=True, end="")

                    mse = 0.5 * np.mean(
                        np.square(x_in_labelled - x_out_labelled) + np.square((y_in_labelled - y_out_labelled)))

                    title_in = "Input: %d | label: %d | Time = %0.4fs" % (sample_num, label_num, time)
                    draw_arr_in = plot2arr_skeleton(x_in_labelled, y_in_labelled, sample_num, t, title_in,
                                                    x_lim=(-0.6, 0.6), y_lim=(0.6, -0.6))
                    title_latent = "Latents | mse = %f" % mse
                    draw_arr_latent = plot2arr_latents(mu_labelled, sample_num, t, title_latent)
                    title_out = "Output: %d | label: %d | Time = %0.4fs" % (sample_num, label_num, time)
                    draw_arr_out = plot2arr_skeleton(x_out_labelled, y_out_labelled, sample_num, t, title_out,
                                                     x_lim=(-0.6, 0.6), y_lim=(0.6, -0.6))
                    # title_out = "Zoomed: %d | label: %d | Time = %0.4fs" % (sample_num, label_num, time)
                    # draw_arr_zoomed = plot2arr_skeleton(x_out_labelled, y_out_labelled, sample_num, t, title_out,
                    #                                  x_lim=(0.4, 0.6))
                    h, w = draw_arr_in.shape[0], draw_arr_in.shape[1]
                    output_arr = np.zeros((h * 2, w * 2, 3))
                    output_arr[0:h, 0:w, :] = draw_arr_in
                    output_arr[h:h * 2, 0:w, :] = draw_arr_out
                    output_arr[0:h, w:w * 2, :] = draw_arr_latent
                    # output_arr[h:h * 2, w:w * 2, :] = draw_arr_zoomed
                    vwriter.writeFrame(output_arr)
                print()
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
        mu_np = mu.cpu().numpy().reshape(n_samples, n_times, -1)
        mu_np = np.transpose(mu_np, (0, 2, 1))
        x_in, y_in = in_test_np[:, 0:25, :], in_test_np[:, 25:50, :]
        x_out, y_out = out_test_np[:, 0:25, :], out_test_np[:, 25:50, :]
        return (x_in, y_in), (x_out, y_out), labels_train, mu_np

    def _get_latents_results(self):
        z_sampled = np.random.normal(0, 1, (self.num_samples_latents * self.label_dim, self.latents_dim))
        cond_vec = np.zeros((self.num_samples_latents * self.label_dim, self.label_dim))
        z_c = np.concatenate((z_sampled, cond_vec), axis=1)
        for sample_idx in range(self.num_samples_latents):
            for cond in range(self.label_dim):
                sample_idx_each = (sample_idx * self.label_dim) + cond
                z_c[sample_idx_each, self.latents_dim + cond] = 1
        x, y = self.model_container.sample_from_latents(z_c)
        return x, y, z_c


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
        # import pdb
        # pdb.set_trace()

        out_test_np = out_test.cpu().numpy()
        x_in, y_in = np.transpose(data[:, :, 0:25], (1, 2, 0)), np.transpose(data[:, :, 25:], (1, 2, 0))
        x_out, y_out = np.transpose(out_test_np[:, :, 0:25], (1, 2, 0)), np.transpose(out_test_np[:, :, 25:], (1, 2, 0))
        return (x_in, y_in), (x_out, y_out)


def plot2arr_skeleton(x, y, sample_num, t, title, x_lim=(-0.6, 0.6), y_lim=(0.6, -0.6)):
    fig, ax = plt.subplots()
    ax.scatter(x[sample_num, :, t], y[sample_num, :, t])
    ax = draw_skeleton(ax, x, y, sample_num, t)
    fig.suptitle(title)
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
    fig.tight_layout()
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot2arr_latents(z, sample_num, t, title, x_lim=(-1, 1), y_lim=(-1, 1)):
    fig, ax = plt.subplots()
    ax.scatter(z[sample_num, 0, t], z[sample_num, 1, t], marker="x")
    fig.suptitle(title)
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
    fig.tight_layout()
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def draw_skeleton(ax, x, y, sample_num, t):
    for start, end in openpose_body_draw_sequence:
        ax.plot(x[sample_num, [start, end], t], y[sample_num, [start, end], t])
    return ax
