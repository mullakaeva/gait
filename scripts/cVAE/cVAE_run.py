import torch
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
import skvideo.io as skv
from common.keypoints_format import openpose_body_draw_sequence
from .Model import cVAE, total_loss


class GaitCVAEmodel:
    def __init__(self, data_gen, input_channels=58, hidden_channels=1024, latent_dims=8, gpu=0, save_chkpt_path=None):
        self.data_gen = data_gen
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.latent_dims = latent_dims
        self.device = torch.device('cuda:{}'.format(gpu))
        self.save_chkpt_path = save_chkpt_path
        self.sequence_length, self.label_dims = self.data_gen.n, self.data_gen.label_dims
        self.model = cVAE(n_channels=self.input_channels,
                          L=self.sequence_length,
                          hidden_channels=self.hidden_channels,
                          latent_dims=self.latent_dims,
                          label_dims=self.label_dims).to(self.device)
        params = self.model.parameters()
        self.optimizer = optim.Adam(params, lr=0.001)

    def train(self, n_epochs=50):
        try:
            for epoch in range(n_epochs):
                iter_idx = 0
                for (x, labels), (x_test, labels_test) in self.data_gen.iterator():
                    x, labels = torch.from_numpy(x).float().to(self.device), torch.from_numpy(labels).float().to(self.device)
                    x_test, labels_test = torch.from_numpy(x_test).float().to(self.device), torch.from_numpy(
                        labels_test).float().to(self.device)
                    self.optimizer.zero_grad()

                    # CV set
                    self.model.eval()
                    with torch.no_grad():
                        out_test, mu_test, logvar_test, z_test = self.model.forward(x_test, labels_test)
                        loss_test = total_loss(x_test, out_test, mu_test, logvar_test)

                    # Train set
                    self.model.train()
                    out, mu, logvar, z = self.model.forward(x, labels)
                    loss = total_loss(x, out, mu, logvar)

                    # Back-prop
                    loss.backward()
                    self.optimizer.step()
                    iter_idx += 1
                    # Print Info
                    print("Epoch %d/%d at iter %d/%d | Loss: %0.4f | CV Loss: %0.4f" % (epoch, n_epochs, iter_idx,
                                                                                        self.data_gen.num_rows / self.data_gen.m,
                                                                                        loss,
                                                                                        loss_test)
                          )
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
        print('Loaded ckpt from {}'.format(load_chkpt_path))

    def _save_model(self):
        if self.save_chkpt_path is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, self.save_chkpt_path)
            print('Stored ckpt at {}'.format(self.save_chkpt_path))


class GaitCVAEvisualiser:
    def __init__(self, data_gen, load_model_path, save_vid_dir):
        # Hard coded stuff
        self.num_samples_pred = 5
        self.num_samples_latents = 3
        self.latents_dim, self.label_dim = 8, 8
        self.times = 128
        # Init
        self.data_gen = data_gen
        self.load_model_path = load_model_path
        self.save_vid_dir = save_vid_dir
        self.model_container = GaitCVAEmodel(data_gen)
        self.model_container.load_model(self.load_model_path)

    def visualise_vid(self):
        # Init
        os.makedirs(self.save_vid_dir, exist_ok=True)
        (x_in, y_in), (x_out, y_out) = self._get_pred_results()
        x_latents, y_latents, z_c = self._get_latents_results()

        # Visualise in-out pair
        for sample_num in range(self.num_samples_pred):
            save_vid_path = os.path.join(self.save_vid_dir, "Recon-%d.mp4" % (sample_num))
            vwriter = skv.FFmpegWriter(save_vid_path)
            for t in range(self.times):
                time = t/25
                print("\rNow writing Recon_sample-%d | time-%0.4fs" % (sample_num, time), flush=True,end="")
                title_in = "Input: %d | Time = %0.4fs" % (sample_num, time)
                draw_arr_in = plot2arr(x_in, y_in, sample_num, t, title_in)
                title_out = "Output: %d | Time = %0.4fs" % (sample_num, time)
                draw_arr_out = plot2arr(x_out, y_out, sample_num, t, title_out)

                draw_arr_recon = np.concatenate((draw_arr_in, draw_arr_out), axis = 1)
                vwriter.writeFrame(draw_arr_recon)
            print()
            vwriter.close()

        # Visualise sampled vid from latents
        for sample_num in range(z_c.shape[0]):
            cond = np.argmax(z_c[sample_num, self.latents_dim:])
            save_vid_path = os.path.join(self.save_vid_dir, "Sample-%d_Cond-%d.mp4" % (sample_num, cond))
            vwriter = skv.FFmpegWriter(save_vid_path)
            for t in range(self.times):
                time = t/25
                print("\rNow writing | Cond-%d | Sample-%d | time-%0.4fs" % (cond, sample_num, time), flush=True, end="")
                title = "Sample: %d | Cond: %d | Time = %0.4fs" % (sample_num, cond, time)
                draw_arr_latents = plot2arr(x_latents, y_latents, sample_num, t, title)
                vwriter.writeFrame(draw_arr_latents)
            print()
            vwriter.close()


    def _get_pred_results(self):
        for (x_train, labels_train), (in_test, labels_test) in self.data_gen.iterator():
            self.model_container.model.eval()
            with torch.no_grad():
                in_test, labels_test = torch.from_numpy(in_test).float().to(self.model_container.device), \
                                      torch.from_numpy(labels_test).float().to(self.model_container.device)
                out_test, _, _, _ = self.model_container.model.forward(in_test, labels_test)
            break
        in_test_np, out_test_np = in_test.cpu().numpy(), out_test.cpu().numpy()
        x_in, y_in = in_test_np[:, 0:25, :], in_test_np[:, 25:50, :]
        x_out, y_out = out_test_np[:, 0:25, :], out_test_np[:, 25:50, :]
        return (x_in, y_in), (x_out, y_out)

    def _get_latents_results(self):
        z_sampled = np.random.normal(0, 1, (self.num_samples_latents*self.label_dim, self.latents_dim))
        cond_vec = np.zeros((self.num_samples_latents*self.label_dim, self.label_dim))
        z_c = np.concatenate((z_sampled, cond_vec), axis=1)
        for sample_idx in range(self.num_samples_latents):
            for cond in range(self.label_dim):
                sample_idx_each = (sample_idx * self.label_dim) + cond
                z_c[sample_idx_each, self.latents_dim + cond] = 1
        x, y = self.model_container.sample_from_latents(z_c)
        return x, y, z_c


def plot2arr(x, y, sample_num, t, title):
    fig, ax = plt.subplots()
    ax.scatter(x[sample_num, :, t], y[sample_num, :, t])
    ax = draw_skeleton(ax, x, y, sample_num, t)
    fig.suptitle(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    fig.tight_layout()
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def draw_skeleton(ax, x, y, sample_num, t):
    for start, end in openpose_body_draw_sequence:
        ax.plot(x[sample_num, [start, end], t], y[sample_num, [start, end], t])
    return ax