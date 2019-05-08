import torch
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
import skvideo.io as skv
import skimage.io as ski
from common.keypoints_format import openpose_body_draw_sequence
from .Model import TemporalVAE
from common.utils import RunningAverageMeter


class GaitTVAEmodel:
    def __init__(self, data_gen,
                 input_channels=50,
                 hidden_channels=512,
                 latent_dims=2,
                 gpu=0,
                 KLD_const=0.001,
                 velo_const=0.001,
                 save_chkpt_path=None):

        # Network dimensions related
        self.data_gen = data_gen
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.latent_dims = latent_dims
        self.device = torch.device('cuda:{}'.format(gpu))
        self.sequence_length = self.data_gen.n

        # Loss related
        self.KLD_regularization_const = KLD_const
        self.velo_regularization_const = velo_const
        self.weights_vec_loss = self._get_weights_vec_loss()
        self.loss_train_meter, self.loss_cv_meter = RunningAverageMeter(), RunningAverageMeter()

        # Model initialization
        self.save_chkpt_path = save_chkpt_path
        self.model = TemporalVAE(n_channels=self.input_channels,
                                 L=self.sequence_length,
                                 hidden_channels=self.hidden_channels,
                                 latent_dims=self.latent_dims).to(self.device)
        params = self.model.parameters()
        self.optimizer = optim.Adam(params, lr=0.001)

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
                        loss_test = self.loss_function(x_test, out_test, mu_test, logvar_test)
                        self.loss_cv_meter.update(loss_test.item())

                    # Train set
                    self.model.train()
                    out, mu, logvar, z = self.model(x)
                    loss = self.loss_function(x, out, mu, logvar)
                    self.loss_train_meter.update(loss.item())

                    # Back-prop
                    loss.backward()
                    self.optimizer.step()
                    iter_idx += 1

                    # Print Info
                    print("Epoch %d/%d at iter %d/%d | Loss: %0.4f | CV Loss: %0.4f" % (epoch, n_epochs, iter_idx,
                                                                                        self.data_gen.num_rows / self.data_gen.m,
                                                                                        self.loss_train_meter.avg,
                                                                                        self.loss_cv_meter.avg)
                          )
                # save (overwrite) model file every epoch
                self._save_model()

        except KeyboardInterrupt as e:
            self._save_model()
            raise e

    def sample_from_latents(self, z):
        """

        Parameters
        ----------
        z : numpy.darray
            Latent encoding vectors, with shape (num_samples, num_latents)

        Returns
        -------
        x : numpy.darray
            The reconstructed x-coordinates from latents,
            with shape (num_samples, num_keypoints, sequence_length), e.g. by default, (512, 25, 128)
        y : numpy.darray
            Same as x, except it is the y-coordinates
        """
        z_tensor = torch.from_numpy(z).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model.decode(z_tensor)
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
        self.KLD_regularization_const = checkpoint['KLD_regularization_const']
        self.velo_regularization_const = checkpoint['velo_regularization_const']
        print('Loaded ckpt from {}'.format(load_chkpt_path))

    def _save_model(self):
        if self.save_chkpt_path is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss_train_meter': self.loss_train_meter,
                'loss_cv_meter': self.loss_cv_meter,
                'KLD_regularization_const': self.KLD_regularization_const,
                'velo_regularization_const': self.velo_regularization_const
            }, self.save_chkpt_path)
            print('Stored ckpt at {}'.format(self.save_chkpt_path))

    def _get_weights_vec_loss(self):
        weights_vec = torch.ones(1, 50, 1).float().to(self.device)
        weights_vec[0, 19:25, 0] = 2
        weights_vec[0, 19 * 2:, 0] = 2
        return weights_vec

    def _temporal_smoothen_loss(self, pred):

        pad_vec = torch.zeros((pred.shape[0], 50, 1)).float().to(self.device)
        pred_padded = torch.cat((pred, pad_vec), 2) # Concat along axis=2
        pred_velo = pred_padded[:, :, 1:] - pred
        velo_loss = torch.sum(pred_velo ** 2)
        return velo_loss

    def loss_function(self, x, pred, mu, logvar):
        img_loss = torch.sum(self.weights_vec_loss * ((x - pred) ** 2))
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        velo_loss = self._temporal_smoothen_loss(pred)
        loss = img_loss + self.KLD_regularization_const * KLD + self.velo_regularization_const * velo_loss
        return loss


class GaitCVAEvisualiser:
    def __init__(self, data_gen, load_model_path, save_vid_dir):
        # Hard coded stuff
        self.num_samples_pred = 5
        self.num_samples_latents = 3
        self.latents_dim = 2
        self.times = 128

        # Init paths
        self.save_vid_dir = save_vid_dir
        os.makedirs(self.save_vid_dir, exist_ok=True)
        self.load_model_path = load_model_path

        # Init data & model
        self.data_gen = data_gen
        self.model_container = GaitTVAEmodel(data_gen)
        self.model_container.load_model(self.load_model_path)

    def visualise_random_reconstruction_label_clusters(self, sample_num):
        (x_in, y_in), (x_out, y_out), labels, z = self._get_pred_results()

        hyper_params_title = "KLD = %f | Velo = %f" % (self.model_container.KLD_regularization_const,
                                                       self.model_container.velo_regularization_const)
        print("visualize: %s" % hyper_params_title)
        kld_identifier = -np.log10(self.model_container.KLD_regularization_const)
        velo_identifier = -np.log10(self.model_container.velo_regularization_const)

        # Visualise reconstruction
        for sample_idx in range(sample_num):
            save_vid_path = os.path.join(self.save_vid_dir,
                                         "Recon%d_KLD-%f_Velo-%f.mp4" % (sample_idx,
                                                                         kld_identifier,
                                                                         velo_identifier))
            vwriter = skv.FFmpegWriter(save_vid_path)

            # Draw input & output skeleton for every time step
            for t in range(self.times):
                time = t / 25
                print("\rNow writing Recon_sample-%d | time-%0.4fs" % (sample_idx, time), flush=True, end="")
                draw_arr_in = plot2arr_skeleton(x=x_in[sample_idx, :, t],
                                                y=y_in[sample_idx, :, t],
                                                title="%d | " % sample_idx + hyper_params_title
                                                )

                draw_arr_out = plot2arr_skeleton(x=x_out[sample_idx, :, t],
                                                 y=y_out[sample_idx, :, t],
                                                 title="%d | " % sample_idx + hyper_params_title
                                                 )
                draw_arr_recon = np.concatenate((draw_arr_in, draw_arr_out), axis=1)
                vwriter.writeFrame(draw_arr_recon)
            print()
            vwriter.close()

        # Visualise label clusters images
        save_img_path = os.path.join(self.save_vid_dir,
                                     "cluster_KLD-%f_Velo-%f.png" % (kld_identifier,
                                                                     velo_identifier))
        draw_clusters = plot_latent_labels_cluster(z, labels, hyper_params_title)
        ski.imsave(save_img_path, draw_clusters)

    def _get_pred_results(self):
        for (x_train, labels_train), (_, _) in self.data_gen.iterator():
            self.model_container.model.eval()
            with torch.no_grad():
                x_train = torch.from_numpy(x_train).float().to(self.model_container.device)
                out_test, _, _, z = self.model_container.model(x_train)
            break
        x_train_np, out_test_np, z = x_train.cpu().numpy(), out_test.cpu().numpy(), z.cpu().numpy()
        x_in, y_in = x_train_np[:, 0:25, :], x_train_np[:, 25:50, :]
        x_out, y_out = out_test_np[:, 0:25, :], out_test_np[:, 25:50, :]
        return (x_in, y_in), (x_out, y_out), labels_train, z


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


def plot_latent_labels_cluster(z_space, z_labels, title, x_lim=None, y_lim=None):
    fig, ax = plt.subplots()

    # Scatter all vectors
    im_space = ax.scatter(z_space[:, 0], z_space[:, 1], c=z_labels, cmap="hsv", marker=".", alpha=0.5)
    fig.colorbar(im_space)

    # Set limits of axes
    if x_lim is None:
        x_max = np.quantile(np.abs(z_space[:, 0]), 0.98)
        x_lim = (-x_max, x_max)

    if y_lim is None:
        y_max = np.quantile(np.abs(z_space[:, 1]), 0.98)
        y_lim = (-y_max, y_max)

    # Title, limits and drawing
    fig.suptitle(title)
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
    fig.tight_layout()
    fig.canvas.draw()

    # Convert to numpy array
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
