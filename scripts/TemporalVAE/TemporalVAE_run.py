import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import os
import matplotlib.pyplot as plt
import skvideo.io as skv
import skimage.io as ski
from common.keypoints_format import plot2arr_skeleton
from .Model_t128 import TemporalVAE
from common.utils import RunningAverageMeter, gaitclass
import umap
from sklearn.decomposition import PCA


class GaitTVAEmodel:
    def __init__(self, data_gen,
                 input_channels=50,
                 hidden_channels=512,
                 latent_dims=2,
                 gpu=0,
                 kld=None,
                 dropout_p=0,
                 init_lr=0.001,
                 lr_milestones=[50, 100, 150],
                 lr_decay_gamma=0.1,
                 save_chkpt_path=None):
        # Others
        self.epoch = 0

        # Network dimensions related
        self.data_gen = data_gen
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.latent_dims = latent_dims
        self.device = torch.device('cuda:{}'.format(gpu))
        self.sequence_length = self.data_gen.n

        # Loss related
        self.kld = kld
        self.dropout_p = dropout_p
        self.weights_vec = self._get_weights_vec()
        self.loss_train_meter, self.loss_cv_meter = RunningAverageMeter(), RunningAverageMeter()
        self.loss_recorder = {"train": [], "cv": []}

        # Model initialization
        self.save_chkpt_path = save_chkpt_path
        self.model = TemporalVAE(n_channels=self.input_channels,
                                 L=self.sequence_length,
                                 hidden_channels=self.hidden_channels,
                                 latent_dims=self.latent_dims,
                                 dropout_p=self.dropout_p).to(self.device)
        params = self.model.parameters()
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
                        loss_test, kld_loss_test = self.loss_function(x_test, out_test, mu_test, logvar_test)
                        self.loss_cv_meter.update(loss_test.item())

                    # Train set
                    self.model.train()
                    out, mu, logvar, z = self.model(x)
                    loss, kld_loss = self.loss_function(x, out, mu, logvar)
                    self.loss_train_meter.update(loss.item())

                    # Back-prop
                    loss.backward()
                    self.optimizer.step()
                    iter_idx += 1

                    # Print Info
                    print("\rEpoch %d/%d at iter %d/%d | Loss: %0.8f | CV Loss: %0.8f" % (self.epoch,
                                                                                          n_epochs,
                                                                                          iter_idx,
                                                                                          self.data_gen.num_rows / self.data_gen.m,
                                                                                          self.loss_train_meter.avg,
                                                                                          self.loss_cv_meter.avg),
                          flush=True, end="")
                print()
                # save (overwrite) model file every epoch
                self._save_model()
                self._plot_loss()
                self.lr_scheduler.step(epoch=self.epoch)

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
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.loss_train_meter = checkpoint['loss_train_meter']
        self.loss_cv_meter = checkpoint['loss_cv_meter']
        self.loss_recorder = checkpoint['loss_recorder']
        self.kld = checkpoint['kld']
        self.dropout_p = checkpoint['dropout_p']
        self.epoch = len(self.loss_recorder["train"])
        print('Loaded ckpt from {}'.format(load_chkpt_path))

    def _save_model(self):
        if self.save_chkpt_path is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'loss_train_meter': self.loss_train_meter,
                'loss_cv_meter': self.loss_cv_meter,
                'loss_recorder': self.loss_recorder,
                'kld': self.kld,
                'dropout_p': self.dropout_p
            }, self.save_chkpt_path)
            print('Stored ckpt at {}'.format(self.save_chkpt_path))

    def _get_weights_vec(self):
        weights_vec = self.data_gen.weighting_vec.copy().reshape(1, -1, 1)
        weights_vec = torch.from_numpy(weights_vec).float().to(self.device)
        return weights_vec

    def _temporal_smoothen_loss(self, pred):

        pad_vec = torch.zeros((pred.shape[0], 50, 1)).float().to(self.device)
        pred_padded = torch.cat((pred, pad_vec), 2)  # Concat along axis=2
        pred_velo = pred_padded[:, :, 1:] - pred
        velo_loss = torch.sum(pred_velo ** 2)
        return velo_loss

    def loss_function(self, x, pred, mu, logvar):
        # Set KLD
        if self.kld is None:
            kld_loss = 0
        else:
            kld_loss = self.kld * (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
        # Set recon loss
        recon_loss = torch.mean(self.weights_vec * ((x - pred) ** 2))
        # Combine different losses
        loss = recon_loss + kld_loss
        return loss, kld_loss

    def _plot_loss(self):

        def sliding_plot(epoch_windows, ax1, ax2):
            x_length = np.linspace(self.epoch - epoch_windows, self.epoch - 1, epoch_windows)
            y_train = self.loss_recorder["train"][self.epoch - epoch_windows:]
            y_cv = self.loss_recorder["cv"][self.epoch - epoch_windows:]

            y_min_train, y_max_train = np.min(y_train), np.max(y_train)
            y_min_cv, y_max_cv = np.min(y_cv), np.max(y_cv)

            ax1.plot(x_length, y_train, label="train")
            ax2.plot(x_length, y_cv, label="cv")
            ax1.set_ylim(y_min_train, y_max_train)
            ax2.set_ylim(y_min_cv, y_max_cv)

        epoch_windows_original = 100
        epoch_windows_zoomed = 20

        self.loss_recorder["train"].append(self.loss_train_meter.avg)
        self.loss_recorder["cv"].append(self.loss_cv_meter.avg)
        self.epoch = len(self.loss_recorder["train"])
        fig, ax = plt.subplots(2, 2, figsize=(12, 12))

        # Restrict to show only recent epochs
        if self.epoch > epoch_windows_original:
            sliding_plot(epoch_windows_original, ax[0, 0], ax[1, 0])
        else:
            ax[0, 0].plot(self.loss_recorder["train"], label="train")
            ax[1, 0].plot(self.loss_recorder["cv"], label="cv")

        if self.epoch > epoch_windows_zoomed:
            sliding_plot(epoch_windows_zoomed, ax[0, 1], ax[1, 1])
        else:
            ax[0, 1].axis("off")
            ax[1, 1].axis("off")
        fig.suptitle(os.path.splitext(os.path.split(self.save_chkpt_path)[1])[0])
        plt.savefig(self.save_chkpt_path + ".png", dpi=300)


class GaitCVAEvisualiser:
    def __init__(self, data_gen, load_model_path, save_vid_dir,
                 hidden_channels=512,
                 latent_dims=2,
                 kld=None,
                 dropout_p=0,
                 init_lr=0.001,
                 lr_milestones=[50, 100, 150],
                 lr_decay_gamma=0.1,
                 model_identifier=""):
        # Hard coded stuff
        self.num_samples_pred = 5
        self.num_samples_latents = 3
        self.latents_dim = 2
        self.times = data_gen.n

        # Init paths
        self.save_vid_dir = save_vid_dir
        os.makedirs(self.save_vid_dir, exist_ok=True)
        self.load_model_path = load_model_path

        # Init data & model
        self.data_gen = data_gen
        self.model_container = GaitTVAEmodel(data_gen,
                                             hidden_channels=hidden_channels,
                                             latent_dims=latent_dims,
                                             kld=kld,
                                             dropout_p=dropout_p,
                                             init_lr=init_lr,
                                             lr_milestones=lr_milestones,
                                             lr_decay_gamma=lr_decay_gamma)
        self.model_container.load_model(self.load_model_path)
        self.model_identifier = model_identifier

    def visualise_random_reconstruction_label_clusters(self, sample_num):
        (x_in, y_in), (x_out, y_out), labels, z = self._get_pred_results()

        print("visualize: %s" % self.model_identifier)

        # Visualise reconstruction
        for sample_idx in range(sample_num):
            save_vid_path = os.path.join(self.save_vid_dir,
                                         "Recon%d_%s.mp4" % (sample_idx, self.model_identifier))

            vwriter = skv.FFmpegWriter(save_vid_path)

            # Draw input & output skeleton for every time step
            for t in range(self.times):
                time = t / 25
                print("\rNow writing Recon_sample-%d | time-%0.4fs" % (sample_idx, time), flush=True, end="")
                draw_arr_in = plot2arr_skeleton(x=x_in[sample_idx, :, t],
                                                y=y_in[sample_idx, :, t],
                                                title="%d | " % sample_idx + self.model_identifier
                                                )

                draw_arr_out = plot2arr_skeleton(x=x_out[sample_idx, :, t],
                                                 y=y_out[sample_idx, :, t],
                                                 title="%d | " % sample_idx + self.model_identifier
                                                 )
                draw_arr_recon = np.concatenate((draw_arr_in, draw_arr_out), axis=1)
                vwriter.writeFrame(draw_arr_recon)
            print()
            vwriter.close()

        # Visualise label clusters images
        save_img_path = os.path.join(self.save_vid_dir,
                                     "cluster_%s.png" % (self.model_identifier))

        draw_clusters = plot_latent_labels_cluster(z, labels, self.model_identifier)
        ski.imsave(save_img_path, draw_clusters)

    def visualize_umap_embedding(self,
                                 n_neighs=[5, 15, 50],
                                 min_dists=[0.001, 0.1, 0.5],
                                 metrics=["euclidean"],
                                 pca_enableds=[True, False]):

        z, labels = self._get_all_latents_results()
        # pca_model = PCA(n_components=50)
        # z_pca = pca_model.fit_transform(z)
        save_img_dir = os.path.join(self.save_vid_dir, "umap")
        # Loop for creating umap
        for neigh in n_neighs:
            for min_dist in min_dists:
                for metric in metrics:
                    for pca in pca_enableds:
                        umap_identifier = "PCA-{}_neigh-{}_dist-{}_metric-{}".format(pca, neigh, min_dist, metric)

                        save_img1 = os.path.join(save_img_dir,
                                                 self.model_identifier + "_type1_" + umap_identifier + ".png")
                        save_img2 = os.path.join(save_img_dir,
                                                 self.model_identifier + "_type2_" + umap_identifier + ".png")
                        print("Visualizing U-map | {}".format(umap_identifier))

                        # U-map embedding
                        if pca:
                            embedding = umap.UMAP(n_neighbors=neigh,
                                                  min_dist=min_dist,
                                                  metric=metric).fit_transform(z_pca)
                        else:
                            embedding = umap.UMAP(n_neighbors=neigh,
                                                  min_dist=min_dist,
                                                  metric=metric).fit_transform(z)

                        # Output image
                        title = self.model_identifier + "\n{}".format(umap_identifier)
                        draw_clusters_sep = plot_umap_with_labels(embedding, labels, title)
                        plt.savefig(save_img1, dpi=300)
                        draw_clusters_sin = plot_latent_labels_cluster(embedding, labels, title, alpha=0.2)
                        plt.savefig(save_img2, dpi=300)
                        plt.close()

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

    def _get_all_latents_results(self):
        z_list = []
        labels_list = []
        for (x_train, labels_train), (_, _) in self.data_gen.iterator():
            self.model_container.model.eval()
            with torch.no_grad():
                x_train = torch.from_numpy(x_train).float().to(self.model_container.device)
                out_test, _, _, z = self.model_container.model(x_train)
                z_list.append(z.cpu().numpy())
                labels_list.append(labels_train)
        all_z = np.vstack(z_list)
        all_labels = np.concatenate(labels_list)
        return all_z, all_labels


def plot_latent_labels_cluster(z_space, z_labels, title, x_lim=None, y_lim=None, alpha=0.5):
    fig, ax = plt.subplots(figsize=(12, 12))

    # Scatter all vectors
    im_space = ax.scatter(z_space[:, 0], z_space[:, 1], c=z_labels, cmap="hsv", marker=".", alpha=alpha)
    fig.colorbar(im_space)

    # Set limits of axes
    # if x_lim is None:
    #     x_max = np.quantile(np.abs(z_space[:, 0]), 0.995)
    #     x_lim = (-x_max, x_max)
    #
    # if y_lim is None:
    #     y_max = np.quantile(np.abs(z_space[:, 1]), 0.995)
    #     y_lim = (-y_max, y_max)

    # Title, limits and drawing
    fig.suptitle(title)
    # ax.set_xlim(x_lim[0], x_lim[1])
    # ax.set_ylim(y_lim[0], y_lim[1])
    fig.canvas.draw()

    # Convert to numpy array
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return data


def plot_umap_with_labels(z, labels, title):
    fig, ax = plt.subplots(2, 4, figsize=(14, 7))
    ax = ax.ravel()
    for class_idx in range(8):
        embed_this_class = z[labels.astype(int) == class_idx, :]
        embed_other_classes = z[labels.astype(int) != class_idx, :]
        ax[class_idx].scatter(embed_other_classes[:, 0], embed_other_classes[:, 1], c="0.1", marker=".", alpha=0.25)
        ax[class_idx].scatter(embed_this_class[:, 0], embed_this_class[:, 1], c="r", marker=".", alpha=0.1)
        ax[class_idx].set_title("{}".format(gaitclass(class_idx)))
        ax[class_idx].axis("off")

    # Title, limits and drawing
    fig.suptitle(title)
    fig.canvas.draw()

    # Convert to numpy array
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return data
