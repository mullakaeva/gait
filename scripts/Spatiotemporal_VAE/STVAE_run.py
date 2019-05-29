import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn import CrossEntropyLoss
import numpy as np
import os
import matplotlib.pyplot as plt
import pprint
from common.utils import MeterAssembly
from .Model import SpatioTemporalVAE
from common.visualisation import plot2arr_skeleton, plot_latent_space_with_labels, plot_umap_with_labels, \
    build_frame_4by4, plot_pca_explained_var

from sklearn.decomposition import PCA
import skvideo.io as skv


class STVAEmodel:
    def __init__(self,
                 data_gen,
                 fea_dim=50,
                 seq_dim=128,
                 posenet_latent_dim=10,
                 posenet_dropout_p=0,
                 posenet_kld=None,
                 motionnet_latent_dim=25,
                 motionnet_hidden_dim=512,
                 motionnet_dropout_p=0,
                 motionnet_kld=None,
                 pose_latent_gradient=0,
                 recon_gradient=0,
                 classification_weight=0,
                 gpu=0,
                 init_lr=0.001,
                 lr_milestones=[50, 100, 150],
                 lr_decay_gamma=0.1,
                 save_chkpt_path=None,
                 load_chkpt_path=None):

        # Others
        self.epoch = 0
        self.device = torch.device('cuda:{}'.format(gpu))
        self.save_chkpt_path = save_chkpt_path
        self.load_chkpt_path = load_chkpt_path

        # Load parameters
        self.data_gen = data_gen
        self.fea_dim = fea_dim
        self.seq_dim = seq_dim
        self.init_lr = init_lr
        self.lr_milestones = lr_milestones
        self.lr_decay_gamma = lr_decay_gamma
        self.posenet_latent_dim = posenet_latent_dim
        self.posenet_dropout_p = posenet_dropout_p
        self.motionnet_latent_dim = motionnet_latent_dim
        self.motionnet_dropout_p = motionnet_dropout_p
        self.motionnet_hidden_dim = motionnet_hidden_dim
        self.pose_latent_gradient = pose_latent_gradient
        self.recon_gradient = recon_gradient
        self.classification_weight = classification_weight
        self.posenet_kld = posenet_kld
        self.motionnet_kld = motionnet_kld
        self.posenet_kld_bool = False if self.posenet_kld is None else True
        self.motionnet_kld_bool = False if self.motionnet_kld is None else True

        self.loss_meter = MeterAssembly(
            "train_total_loss",
            "train_recon",
            "train_pose_kld",
            "train_motion_kld",
            "train_recon_grad",
            "train_latent_grad",
            "train_acc",
            "test_total_loss",
            "test_recon",
            "test_pose_kld",
            "test_motion_kld",
            "test_recon_grad",
            "test_latent_grad",
            "test_acc"
        )
        self.weights_vec = self._get_weights_vec()
        self.class_criterion = CrossEntropyLoss()
        # Initialize model, params, optimizer, loss
        if load_chkpt_path is None:
            self.model, self.optimizer, self.lr_scheduler = self._model_initialization()
        else:
            self.model, self.optimizer, self.lr_scheduler = self._load_model()

    def train(self, n_epochs=50):
        try:
            for epoch in range(n_epochs):
                iter_idx = 0
                for (x, labels, _), (x_test, labels_test, _) in self.data_gen.iterator():
                    # Convert numpy to torch.tensor
                    x = torch.from_numpy(x).float().to(self.device)
                    x_test = torch.from_numpy(x_test).float().to(self.device)
                    labels = torch.from_numpy(labels).long().to(self.device)
                    labels_test = torch.from_numpy(labels_test).long().to(self.device)

                    # Clear optimizer's previous gradients
                    self.optimizer.zero_grad()

                    # CV set
                    self.model.eval()
                    with torch.no_grad():
                        recon_motion_t, pred_labels_t, pose_stats_t, motion_stats_t = self.model(x_test)
                        recon_info_t, class_info_t = (x_test, recon_motion_t), (labels_test, pred_labels_t)
                        loss_t, (
                            recon_t, posekld_t, motionkld_t, recongrad_t, latentgrad_t, acc_t) = self.loss_function(
                            recon_info_t,
                            class_info_t,
                            pose_stats_t,
                            motion_stats_t
                        )
                        self.loss_meter.update_meters(
                            test_total_loss=loss_t.item(),
                            test_recon=recon_t.item(),
                            test_pose_kld=posekld_t.item(),
                            test_motion_kld=motionkld_t.item(),
                            test_recon_grad=recongrad_t.item(),
                            test_latent_grad=latentgrad_t.item(),
                            test_acc=acc_t
                        )

                    # Train set
                    self.model.train()
                    recon_motion, pred_labels, pose_stats, motion_stats = self.model(x)
                    recon_info, class_info = (x, recon_motion), (labels, pred_labels)
                    loss, (recon, posekld, motionkld, recongrad, latentgrad, acc) = self.loss_function(recon_info,
                                                                                                       class_info,
                                                                                                       pose_stats,
                                                                                                       motion_stats)
                    self.loss_meter.update_meters(
                        train_total_loss=loss.item(),
                        train_recon=recon.item(),
                        train_pose_kld=posekld.item(),
                        train_motion_kld=motionkld.item(),
                        train_recon_grad=recongrad.item(),
                        train_latent_grad=latentgrad.item(),
                        train_acc=acc
                    )

                    # Back-prop
                    loss.backward()
                    self.optimizer.step()
                    iter_idx += 1

                    # Print Info
                    print("\rEpoch %d/%d at iter %d/%d | loss = %0.8f, %0.8f | acc = %0.3f, %0.3f" % (
                        self.epoch,
                        n_epochs,
                        iter_idx,
                        self.data_gen.num_rows / self.data_gen.m,
                        self.loss_meter.get_meter_avg()["train_total_loss"],
                        self.loss_meter.get_meter_avg()["test_total_loss"],
                        acc, acc_t
                    ), flush=True, end=""
                          )

                # Print losses and update recorders
                print()
                pprint.pprint(self.loss_meter.get_meter_avg())
                self.loss_meter.update_recorders()
                self.epoch = len(self.loss_meter.get_recorders()["train_total_loss"])
                self.lr_scheduler.step(epoch=self.epoch)

                # save (overwrite) model file every epoch
                self._save_model()
                self._plot_loss()

        except KeyboardInterrupt as e:
            self._save_model()
            raise e

    def _load_model(self):
        checkpoint = torch.load(self.load_chkpt_path)
        print('Loaded ckpt from {}'.format(self.load_chkpt_path))
        # Attributes for model initialization
        self.loss_meter = checkpoint['loss_meter']
        self.epoch = len(self.loss_meter.get_recorders()["train_total_loss"])
        self.fea_dim = checkpoint['fea_dim']
        self.seq_dim = checkpoint['seq_dim']
        self.init_lr = checkpoint['init_lr']
        self.lr_milestones = checkpoint['lr_milestones']
        self.lr_decay_gamma = checkpoint['lr_decay_gamma']
        self.posenet_latent_dim = checkpoint['posenet_latent_dim']
        self.posenet_dropout_p = checkpoint['posenet_dropout_p']
        self.motionnet_latent_dim = checkpoint['motionnet_latent_dim']
        self.motionnet_dropout_p = checkpoint['motionnet_dropout_p']
        self.motionnet_hidden_dim = checkpoint['motionnet_hidden_dim']
        self.pose_latent_gradient = checkpoint['pose_latent_gradient']
        self.recon_gradient = checkpoint['recon_gradient']
        self.classification_weight = checkpoint['classification_weight']
        self.posenet_kld = checkpoint['posenet_kld']
        self.motionnet_kld = checkpoint['motionnet_kld']
        self.posenet_kld_bool = checkpoint['posenet_kld_bool']
        self.motionnet_kld_bool = checkpoint['motionnet_kld_bool']

        # Model initialization
        model, optimizer, lr_scheduler = self._model_initialization()
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        return model, optimizer, lr_scheduler

    def _save_model(self):
        if self.save_chkpt_path is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'loss_meter': self.loss_meter,
                'fea_dim': self.fea_dim,
                'seq_dim': self.seq_dim,
                'init_lr': self.init_lr,
                'lr_milestones': self.lr_milestones,
                'lr_decay_gamma': self.lr_decay_gamma,
                'posenet_latent_dim': self.posenet_latent_dim,
                'posenet_dropout_p': self.posenet_dropout_p,
                'motionnet_latent_dim': self.motionnet_latent_dim,
                'motionnet_dropout_p': self.motionnet_dropout_p,
                'motionnet_hidden_dim': self.motionnet_hidden_dim,
                'pose_latent_gradient': self.pose_latent_gradient,
                'recon_gradient': self.recon_gradient,
                'classification_weight': self.classification_weight,
                'posenet_kld': self.posenet_kld,
                'motionnet_kld': self.motionnet_kld,
                'posenet_kld_bool': self.posenet_kld_bool,
                'motionnet_kld_bool': self.motionnet_kld_bool
            }, self.save_chkpt_path)

            print('Stored ckpt at {}'.format(self.save_chkpt_path))

    def loss_function(self, recon_info, class_info, pose_stats, motion_stats):
        x, recon_motion = recon_info
        labels, pred_labels = class_info
        pose_z_seq, pose_mu, pose_logvar = pose_stats
        motion_z, motion_mu, motion_logvar = motion_stats

        # Posenet kld
        posenet_kld_multiplier = self._get_kld_multiplier(self.posenet_kld)
        posenet_kld_loss_indicator = -0.5 * torch.mean(1 + pose_logvar - pose_mu.pow(2) - pose_logvar.exp())
        posenet_kld_loss = posenet_kld_multiplier * posenet_kld_loss_indicator

        # Motionnet kld
        motionnet_kld_multiplier = self._get_kld_multiplier(self.motionnet_kld)
        motionnet_kld_loss_indicator = -0.5 * torch.mean(1 + motion_logvar - motion_mu.pow(2) - motion_logvar.exp())
        motionnet_kld_loss = motionnet_kld_multiplier * motionnet_kld_loss_indicator

        # Recon loss
        recon_loss = torch.mean(self.weights_vec * ((x - recon_motion) ** 2))

        # Gradient loss
        recon_grad_loss_indicator = torch.mean(self.weights_vec * self._calc_gradient(recon_motion))
        pose_latent_grad_loss_indicator = torch.mean(self._calc_gradient(pose_z_seq))
        recon_grad_loss = self.recon_gradient * recon_grad_loss_indicator
        pose_latent_grad_loss = self.pose_latent_gradient * pose_latent_grad_loss_indicator

        # Classification loss
        class_loss_indicator, acc = self._get_classification_acc(pred_labels, labels)
        class_loss = self.classification_weight * class_loss_indicator

        # Combine different losses
        loss = recon_loss + posenet_kld_loss + motionnet_kld_loss + recon_grad_loss + pose_latent_grad_loss + class_loss

        return loss, (recon_loss, posenet_kld_loss_indicator, motionnet_kld_loss_indicator, recon_grad_loss_indicator,
                      pose_latent_grad_loss_indicator, acc)

    def _model_initialization(self):
        model = SpatioTemporalVAE(
            fea_dim=self.fea_dim,
            seq_dim=self.seq_dim,
            posenet_latent_dim=self.posenet_latent_dim,
            posenet_dropout_p=self.posenet_dropout_p,
            posenet_kld=self.posenet_kld_bool,
            motionnet_latent_dim=self.motionnet_latent_dim,
            motionnet_hidden_dim=self.motionnet_hidden_dim,
            motionnet_dropout_p=self.motionnet_dropout_p,
            motionnet_kld=self.motionnet_kld_bool
        ).to(self.device)
        params = model.parameters()
        optimizer = optim.Adam(params, lr=self.init_lr)
        lr_scheduler = MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=self.lr_decay_gamma)
        return model, optimizer, lr_scheduler

    def _plot_loss(self):
        '''
            "train_recon",
            "train_pose_kld",
            "train_motion_kld",
            "train_recon_grad",
            "train_latent_grad",
            "train_acc",
        '''

        def plot_ax_train_test(ax, x_length, windows, recorders, key_suffix, train_ylabel, test_ylabel):
            ax_tw = ax.twinx()
            ax.plot(x_length, recorders["train_" + key_suffix][windows:], c="b")
            ax_tw.plot(x_length, recorders["test_" + key_suffix][windows:], c="r")
            ax.set_ylabel(train_ylabel)
            ax_tw.set_ylabel(test_ylabel)

        def sliding_plot(epoch_windows, axes, recorders):
            windows = self.epoch - epoch_windows
            x_length = np.linspace(windows, self.epoch - 1, epoch_windows)

            plot_ax_train_test(axes[0, 0], x_length, windows, recorders, "recon", "Train Recon MSE", "")
            plot_ax_train_test(axes[1, 0], x_length, windows, recorders, "pose_kld", "Train pose_kld", "")
            plot_ax_train_test(axes[2, 0], x_length, windows, recorders, "motion_kld", "Train motion_kld", "")
            plot_ax_train_test(axes[0, 1], x_length, windows, recorders, "recon_grad", "", "Test recon_grad")
            plot_ax_train_test(axes[1, 1], x_length, windows, recorders, "latent_grad", "", "Test latent_grad")
            plot_ax_train_test(axes[2, 1], x_length, windows, recorders, "acc", "", "Test acc")

        epoch_windows = 100
        recorders = self.loss_meter.get_recorders()
        fig, ax = plt.subplots(3, 2, figsize=(16, 8))

        # Restrict to show only recent epochs
        if self.epoch > epoch_windows:
            sliding_plot(epoch_windows, ax, recorders)
        else:
            sliding_plot(self.epoch, ax, recorders)

        fig.suptitle(os.path.splitext(os.path.split(self.save_chkpt_path)[1])[0])
        plt.savefig(self.save_chkpt_path + ".png", dpi=300)

    def _get_weights_vec(self):
        weights_vec = self.data_gen.weighting_vec.copy().reshape(1, -1, 1)
        weights_vec = torch.from_numpy(weights_vec).float().to(self.device)
        return weights_vec

    def _get_kld_multiplier(self, kld_arg):
        # Set KLD loss
        if kld_arg is None:
            kld_multiplier = 0
        elif isinstance(kld_arg, list):
            start, end, const = kld_arg[0], kld_arg[1], kld_arg[2]
            if self.epoch < start:
                kld_multiplier = 0
            elif (self.epoch >= start) and (self.epoch < end):
                kld_multiplier = const * ((self.epoch - start) / (end - start))
            elif self.epoch >= end:
                kld_multiplier = const
        elif isinstance(kld_arg, int) or isinstance(kld_arg, float):
            kld_multiplier = kld_arg
        return kld_multiplier

    def _calc_gradient(self, x):
        grad = (x[:, :, 0:127] - x[:, :, 1:]) ** 2
        return grad

    def _get_classification_acc(self, pred_labels, labels):
        class_loss_indicator = self.class_criterion(pred_labels, labels)
        pred_labels_np, labels_np = pred_labels.cpu().detach().numpy(), labels.cpu().detach().numpy()
        acc = np.mean(np.argmax(pred_labels_np, axis=1) == labels_np) * 100
        return class_loss_indicator, acc

    def save_model_losses_data(self, save_vid_dir, model_identifier):
        import pandas as pd
        loss_data = self.loss_meter.get_recorders()
        df_losses = pd.DataFrame(loss_data)
        df_losses.to_csv(os.path.join(save_vid_dir, "results_data", "loss_{}.csv".format(model_identifier)))

    def vis_reconstruction(self, data_gen, sample_num, save_vid_dir, model_identifier):
        # Refresh data generator
        self.data_gen = data_gen

        # Get data from data generator's first loop
        for (x, labels, _), (_, _, _) in self.data_gen.iterator():
            x = torch.from_numpy(x).float().to(self.device)
            break

        # Forward pass
        self.model.eval()
        with torch.no_grad():
            recon_motion, pred_labels, (pose_z_seq, pose_mu, pose_logvar), (
                motion_z, motion_mu, motion_logvar) = self.model(x)

        # Convert to numpy
        x = x.cpu().detach().numpy()
        recon_motion = recon_motion.cpu().detach().numpy()
        motion_z = motion_z.cpu().detach().numpy()
        pose_z_seq = pose_z_seq.cpu().detach().numpy()
        seq_length = x.shape[2]

        # Flatten pose latent

        pose_z_flat = np.transpose(pose_z_seq, (0, 2, 1)).reshape(pose_z_seq.shape[0] * pose_z_seq.shape[2], -1)
        labels_flat = np.zeros((labels.shape[0], seq_length))
        labels_flat[:, :] = labels.reshape(labels.shape[0], 1)
        labels_flat = labels_flat.reshape(-1)

        # PCA: Fit, transform and plot
        pose_pcafitter = PCA()
        motion_pcafitter = PCA()
        pose_z_flat = pose_pcafitter.fit_transform(pose_z_flat)
        motion_z = motion_pcafitter.fit_transform(motion_z)
        pose_z_seq_back = np.transpose(pose_z_flat.reshape(pose_z_seq.shape[0], pose_z_seq.shape[2], -1), (0, 2, 1))
        plot_pca_explained_var([pose_pcafitter, motion_pcafitter],
                               "Pose: {} | Motion: {}\nModel: {}".format(pose_z_seq.shape[1], motion_z.shape[1],
                                                                         model_identifier),
                               save_path=os.path.join(save_vid_dir, "results_data",
                                                      "PCA_{}.png".format(model_identifier)))


        # Draw videos
        for sample_idx in range(sample_num):

            save_vid_path = os.path.join(save_vid_dir, "ReconVid-{}_{}.mp4".format(sample_idx, model_identifier))
            vwriter = skv.FFmpegWriter(save_vid_path)

            draw_motion_latents = plot_latent_space_with_labels(motion_z[:, 0:2], labels, "Motion latents",
                                                                target_scatter=motion_z[sample_idx, 0:2], alpha=1)

            # Draw input & output skeleton for every time step
            for t in range(seq_length):
                time = t / 25
                print("\rNow writing Recon_sample-%d | time-%0.4fs" % (sample_idx, time), flush=True, end="")
                draw_arr_in = plot2arr_skeleton(x=x[sample_idx, 0:25, t],
                                                y=x[sample_idx, 25:, t],
                                                title="%d | " % sample_idx + model_identifier
                                                )

                draw_arr_out = plot2arr_skeleton(x=recon_motion[sample_idx, 0:25, t],
                                                 y=recon_motion[sample_idx, 25:, t],
                                                 title=" Recon %d | label %d " % (sample_idx, labels[sample_idx])
                                                 )

                draw_pose_latents = plot_latent_space_with_labels(pose_z_flat[0:10000, 0:2], labels_flat[0:10000],
                                                                  title="pose latent",
                                                                  alpha=0.3,
                                                                  target_scatter=pose_z_seq_back[sample_idx, 0:2, t])

                output_frame = build_frame_4by4([draw_arr_in, draw_arr_out, draw_motion_latents, draw_pose_latents])
                vwriter.writeFrame(output_frame)
            print()
            vwriter.close()
