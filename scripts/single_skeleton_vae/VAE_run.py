import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import os
import matplotlib.pyplot as plt
import skvideo.io as skv
from sklearn.decomposition import PCA
from common.visualisation import plot2arr_skeleton, plot_latent_space_with_labels, build_frame_4by4
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
                 kld=None,
                 dropout_p=0,
                 init_lr=0.001,
                 lr_milestones=[50, 100, 150],
                 lr_decay_gamma=0.1,
                 save_chkpt_path=None,
                 data_gen_type="single"):

        # Others
        self.epoch = 0

        # Standard Init
        self.data_gen = data_gen
        self.input_dims = input_dims
        self.latent_dims = latent_dims
        self.dropout_p = dropout_p
        self.kld = kld
        self.data_gen_type = data_gen_type
        self.device = torch.device('cuda:{}'.format(gpu))
        self.save_chkpt_path = save_chkpt_path

        # Weigted vector for loss
        self.weights_vec = self._get_weights_vec()

        # initialize model, params, optimizer, loss
        self.model = VAE(input_dims=self.input_dims,
                         latent_dims=self.latent_dims,
                         kld=self.kld,
                         dropout_p=self.dropout_p).to(self.device)
        params = self.model.parameters()
        self.loss_meter = {
            "train_recon": RunningAverageMeter(),
            "cv_recon": RunningAverageMeter(),
            "train_kld": RunningAverageMeter(),
            "cv_kld": RunningAverageMeter()
        }
        self.loss_recorder = {
            "train_recon": [], "cv_recon": [], "train_kld": [], "cv_kld": []
        }
        self.optimizer = optim.Adam(params, lr=init_lr)
        self.lr_scheduler = MultiStepLR(self.optimizer, milestones=lr_milestones, gamma=lr_decay_gamma)

    def train(self, n_epochs=50):
        try:
            for epoch in range(n_epochs):
                iter_idx = 0
                for (x, _), (x_test, _) in self.data_gen.iterator():
                    x = torch.from_numpy(x).float().to(self.device)
                    x_test = torch.from_numpy(x_test).float().to(self.device)
                    self.optimizer.zero_grad()

                    # CV set
                    self.model.eval()
                    with torch.no_grad():
                        out_test, mu_test, logvar_test, z_test = self.model(x_test)
                        loss_cv, (recon_loss_cv, kld_loss_cv) = self.loss_function(x_test, out_test, mu_test,
                                                                                   logvar_test)
                        self.loss_meter["cv_recon"].update(recon_loss_cv.item())
                        self.loss_meter["cv_kld"].update(kld_loss_cv.item())

                    # Train set
                    self.model.train()
                    out, mu, logvar, z = self.model(x)
                    loss, (recon_loss, kld_loss) = self.loss_function(x, out, mu, logvar)
                    self.loss_meter["train_recon"].update(recon_loss.item())
                    self.loss_meter["train_kld"].update(kld_loss.item())

                    # Back-prop
                    loss.backward()
                    self.optimizer.step()
                    iter_idx += 1

                    # Print Info
                    print("\rEpoch %d/%d at iter %d/%d | Loss: (%0.8f, %0.8f) | CV Loss: (%0.8f, %0.8f)" % (
                        self.epoch,
                        n_epochs,
                        iter_idx,
                        self.data_gen.num_rows / self.data_gen.m,
                        self.loss_meter["train_recon"].avg,
                        self.loss_meter["train_kld"].avg,
                        self.loss_meter["cv_recon"].avg,
                        self.loss_meter["cv_kld"].avg
                    ),
                          flush=True, end="")
                print()
                # save (overwrite) model file every epoch
                self._save_model()
                self._plot_loss()
                self.lr_scheduler.step(epoch=self.epoch)

        except KeyboardInterrupt as e:
            self._save_model()
            raise e

    def sample_from_latents(self, z_c):
        """

        Parameters
        ----------
        z_c : numpy.darray
            Conditioned latent encoding vectors, with shape (num_samples, num_latents)

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
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.loss_meter = checkpoint['loss_meter']
        self.loss_recorder = checkpoint['loss_recorder']
        self.loss_recorder = checkpoint['loss_recorder']
        self.kld = checkpoint['kld']
        self.dropout_p = checkpoint['dropout_p']
        self.epoch = len(self.loss_recorder["train_recon"])
        print('Loaded ckpt from {}'.format(load_chkpt_path))

    def _save_model(self):
        if self.save_chkpt_path is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'loss_meter': self.loss_meter,
                'loss_recorder': self.loss_recorder,
                'kld': self.kld,
                'dropout_p': self.dropout_p
            }, self.save_chkpt_path)
            print('Stored ckpt at {}'.format(self.save_chkpt_path))

    def loss_function(self, x, pred, mu, logvar):
        # Set KLD loss
        if self.kld is None:
            kld_multiplier = 0
        elif isinstance(self.kld, list):
            kld_multiplier = self._get_kld_multiplier(self.kld[0], self.kld[1], self.kld[2])
        elif isinstance(self.kld, int):
            kld_multiplier = self.kld
        kld_loss_indicator = (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
        kld_loss = kld_multiplier * kld_loss_indicator

        # Set recon loss
        recon_loss = torch.mean(self.weights_vec * ((x - pred) ** 2))
        # Combine different losses
        loss = recon_loss + kld_loss
        return loss, (recon_loss, kld_loss_indicator)

    def _plot_loss(self):

        def sliding_plot(epoch_windows, ax1, ax2):
            x_length = np.linspace(self.epoch - epoch_windows, self.epoch - 1, epoch_windows)
            # Retrieve loss
            y_train_recon = self.loss_recorder["train_recon"][self.epoch - epoch_windows:]
            y_cv_recon = self.loss_recorder["cv_recon"][self.epoch - epoch_windows:]
            y_train_kld = self.loss_recorder["train_kld"][self.epoch - epoch_windows:]
            y_cv_kld = self.loss_recorder["cv_kld"][self.epoch - epoch_windows:]

            # Plot training loss
            ax1.plot(x_length, y_train_recon, c="b")
            ax1_kld = ax1.twinx()
            ax1_kld.plot(x_length, y_train_kld, c="r")
            ax1.set_ylabel("train_recon")
            ax1_kld.set_ylabel("train_kld")

            # Plot cv loss
            ax2.plot(x_length, y_cv_recon, c="b")
            ax2_kld = ax2.twinx()
            ax2_kld.plot(x_length, y_cv_kld, c="r")
            ax2.set_ylabel("cv_recon")
            ax2_kld.set_ylabel("cv_kld")


        epoch_windows_original = 150
        epoch_windows_zoomed = 20

        # Recording loss history
        self.loss_recorder["train_recon"].append(self.loss_meter["train_recon"].avg)
        self.loss_recorder["train_kld"].append(self.loss_meter["train_kld"].avg)
        self.loss_recorder["cv_recon"].append(self.loss_meter["cv_recon"].avg)
        self.loss_recorder["cv_kld"].append(self.loss_meter["cv_kld"].avg)

        self.epoch = len(self.loss_recorder["train_recon"])
        fig, ax = plt.subplots(2, 2, figsize=(12, 12))

        # Restrict to show only recent epochs
        if self.epoch > epoch_windows_original:
            sliding_plot(epoch_windows_original, ax[0, 0], ax[1, 0])
        else:
            sliding_plot(self.epoch, ax[0, 0], ax[1, 0])

        if self.epoch > epoch_windows_zoomed:
            sliding_plot(epoch_windows_zoomed, ax[0, 1], ax[1, 1])
        else:
            ax[0, 1].axis("off")
            ax[1, 1].axis("off")
        fig.suptitle(os.path.splitext(os.path.split(self.save_chkpt_path)[1])[0])
        plt.savefig(self.save_chkpt_path + ".png", dpi=300)

    def _get_weights_vec(self):
        if self.data_gen_type == "temporal":
            weights_vec = self.data_gen.weighting_vec.copy().reshape(1, -1, 1)
        elif self.data_gen_type == "single":
            weights_vec = self.data_gen.weighting_vec.copy().reshape(1, -1)

        weights_vec = torch.from_numpy(weights_vec).float().to(self.device)
        return weights_vec

    def _get_kld_multiplier(self, start, end, const):
        if self.epoch < start:
            return 0
        elif (self.epoch >= start) and (self.epoch < end):
            return const * ((self.epoch - start)/(end - start))
        elif self.epoch >= end:
            return const

class GaitSingleSkeletonVAEvisualiser:
    def __init__(self, data_gen, load_model_path, save_vid_dir, latent_dims=2, input_dims=50, kld=None,
                 dropout_p=0,
                 model_identifier="",
                 data_gen_type="single"):
        # Hard coded stuff
        self.num_samples_pred = 5
        self.num_samples_latents = 3
        self.latents_dim = latent_dims
        self.times = 128

        # Init
        self.data_gen = data_gen
        self.load_model_path = load_model_path
        self.save_vid_dir = save_vid_dir
        self.model_identifier = model_identifier
        self.model_container = GaitVAEmodel(data_gen,
                                            input_dims=input_dims,
                                            latent_dims=latent_dims,
                                            kld=kld,
                                            dropout_p=dropout_p,
                                            data_gen_type=data_gen_type)
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

        for sample_idx in range(self.num_samples_pred):

            save_vid_path = os.path.join(self.save_vid_dir,
                                         "Sample%d_%s.mp4"% (sample_idx,
                                                             self.model_identifier))
            vwriter = skv.FFmpegWriter(save_vid_path)
            for t in range(self.times):
                time = t / 25
                print("\rNow writing %s | Sample-%d | time-%0.3fs" % (self.model_identifier, sample_idx, time),
                      flush=True, end="")
                # Plot and draw input
                title_in = "%s | Input-%d | time-%0.3fs" % (self.model_identifier, sample_idx, time)
                draw_arr_in = plot2arr_skeleton(x_in[sample_idx, :, t],
                                                y_in[sample_idx, :, t],
                                                title_in,
                                                x_lim=(-0.6, 0.6),
                                                y_lim=(0.6, -0.6))
                # Plot and draw latent
                title_latent = "%s | Latent" % self.model_identifier

                draw_arr_latent = plot_latent_space_with_labels(z_space_flattened, labels_flattened, title_latent,
                                                                target_scatter=mu[sample_idx, :, t])

                # Plot and draw output
                title_out = "%s | Output-%d | time-%0.3fs" % (self.model_identifier, sample_idx, time)
                draw_arr_out = plot2arr_skeleton(x_out[sample_idx, :, t],
                                                y_out[sample_idx, :, t],
                                                title_out,
                                                x_lim=(-0.6, 0.6),
                                                y_lim=(0.6, -0.6))
                # Build video frame
                output_arr = build_frame_4by4([draw_arr_in, draw_arr_out, draw_arr_latent])
                plt.close()
                vwriter.writeFrame(output_arr)
            print()
            vwriter.close()

    def visualise_latent_space(self):
        # Randomly sample the data to visualze the latent-label distribution
        z_space, labels_space = self._sample_collapsed_data()
        x_max, y_max = np.quantile(np.abs(z_space[:, 0]), 0.995), np.quantile(np.abs(z_space[:, 1]), 0.995)
        num_scale = int(np.mean([x_max,y_max]))+1

        # Sample data from the grid lines of the latent space
        x_skeleton, y_skeleton, latents, _ = self._get_latents_results(x_max=x_max,
                                                                       y_max=y_max,
                                                                       num_lines=num_scale*4,
                                                                       num_steps=200)
        num_sample = x_skeleton.shape[0]
        save_vid_path = os.path.join(self.save_vid_dir,
                                     "latent_space_%s.mp4" % self.model_identifier)
        vwriter = skv.FFmpegWriter(save_vid_path)
        for sample_idx in range(num_sample):
            print("\rDrawing %d/%d" % (sample_idx, num_sample), flush=True, end="")
            title = "%s | Time = %0.4f" % (self.model_identifier, sample_idx / 25)
            ske_arr = plot2arr_skeleton(x_skeleton[sample_idx, :],
                                        y_skeleton[sample_idx, :],
                                        title)
            latents_arr = plot_latent_space_with_labels(z_space, labels_space, title, x_lim=(-x_max, x_max),
                                                        y_lim=(-y_max, y_max), alpha=0.5,
                                                        target_scatter=latents[sample_idx, ])

            output_arr = np.concatenate((ske_arr, latents_arr), axis=1)
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

    def _get_latents_results(self, x_max, y_max, num_lines, num_steps):
        x_paths, y_paths, color_vals = gen_paths(x_max, y_max, num_lines, num_steps)
        z_paths_temp = np.vstack((x_paths, y_paths)).T  # (num_samples, 2)

        # Pad with 0's if latent dimension > 2
        if self.latents_dim > 2:
            z_paths = np.zeros((z_paths_temp.shape[0], self.latents_dim))
            z_paths[:, 0:2] = z_paths_temp
        else:
            z_paths = z_paths_temp
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
