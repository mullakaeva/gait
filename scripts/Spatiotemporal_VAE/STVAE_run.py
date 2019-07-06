import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn import CrossEntropyLoss

import numpy as np
import os
import matplotlib.pyplot as plt
import pprint
import re
from glob import glob
from common.utils import MeterAssembly, RunningAverageMeter, dict2json
from common.visualisation import LatentSpaceVideoVisualizer, save_vis_data_for_interactiveplot
from .Model import SpatioTemporalVAE
from .GraphModel import GraphSpatioTemporalVAE


class STVAEmodel:
    def __init__(self,
                 data_gen,
                 fea_dim=50,
                 seq_dim=128,
                 model_type="normal",
                 posenet_latent_dim=10,
                 posenet_dropout_p=0,
                 posenet_kld=None,
                 motionnet_latent_dim=25,
                 motionnet_hidden_dim=512,
                 motionnet_dropout_p=0,
                 motionnet_kld=None,
                 recon_weight=1,
                 pose_latent_gradient=0,
                 recon_gradient=0,
                 classification_weight=0,
                 rmse_weighting_startepoch=None,
                 latent_recon_loss=None,  # None = disabled
                 recon_loss_power=2,  # Can also be list, e.g. [2, 4, 50]
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
        self.model_type = model_type
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
        self.recon_weight = recon_weight
        self.pose_latent_gradient = pose_latent_gradient
        self.recon_gradient = recon_gradient
        self.classification_weight = classification_weight
        self.posenet_kld = posenet_kld
        self.motionnet_kld = motionnet_kld
        self.posenet_kld_bool = False if self.posenet_kld is None else True
        self.motionnet_kld_bool = False if self.motionnet_kld is None else True
        self.rmse_weighting_startepoch = rmse_weighting_startepoch
        self.rmse_weighting_vec, self.rmse_weighting_vec_meter = self._initilize_rmse_weighting_vec()
        self.latent_recon_loss = latent_recon_loss
        self.recon_loss_power = recon_loss_power

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
        self.class_criterion = CrossEntropyLoss(reduction="none")
        # Initialize model, params, optimizer, loss
        if load_chkpt_path is None:
            self.model, self.optimizer, self.lr_scheduler = self._model_initialization()
        else:
            self.model, self.optimizer, self.lr_scheduler = self._load_model()
        # self._save_model()  # Enabled only for renewing newly introduced hyper-parameters

    def train(self, n_epochs=50):
        try:
            for epoch in range(n_epochs):
                iter_idx = 0

                for train_data, test_data in self.data_gen.iterator():
                    x, nan_masks, labels, labels_mask, _, _ = train_data
                    x_test, nan_masks_test, labels_test, labels_mask_test, _, _ = test_data

                for (x, nan_masks, labels, labels_mask), (
                        x_test, nan_masks_test, labels_test, labels_mask_test) in self.data_gen.iterator():
                    # Convert numpy to torch.tensor
                    x = torch.from_numpy(x).float().to(self.device)
                    x_test = torch.from_numpy(x_test).float().to(self.device)
                    labels = torch.from_numpy(labels).long().to(self.device)
                    labels_test = torch.from_numpy(labels_test).long().to(self.device)
                    labels_mask = torch.from_numpy(labels_mask * 1 + 1e-5).float().to(self.device)
                    labels_mask_test = torch.from_numpy(labels_mask_test * 1 + 1e-5).float().to(self.device)
                    nan_masks = torch.from_numpy(nan_masks * 1 + 1e-5).float().to(self.device)
                    nan_masks_test = torch.from_numpy(nan_masks_test * 1 + 1e-5).float().to(self.device)

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
                            motion_stats_t,
                            nan_masks_test,
                            labels_mask_test
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
                                                                                                       motion_stats,
                                                                                                       nan_masks,
                                                                                                       labels_mask)

                    # Running average of RMSE weighting
                    if (self.rmse_weighting_startepoch is not None) and (self.epoch == self.rmse_weighting_startepoch):
                        squared_diff = nan_masks((recon_motion - x) ** 2)  # (n_samples, 50, 128)
                        self._update_rmse_weighting_vec(squared_diff)

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

                # Assign the average RMSE_weight to weighting vector
                if (self.rmse_weighting_startepoch is not None) and (self.epoch == self.rmse_weighting_startepoch):
                    self.rmse_weighting_vec = self.rmse_weighting_vec_meter.avg.clone()

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
        self.recon_weight = checkpoint['recon_weight']
        self.pose_latent_gradient = checkpoint['pose_latent_gradient']
        self.recon_gradient = checkpoint['recon_gradient']
        self.classification_weight = checkpoint['classification_weight']
        self.posenet_kld = checkpoint['posenet_kld']
        self.motionnet_kld = checkpoint['motionnet_kld']
        self.posenet_kld_bool = checkpoint['posenet_kld_bool']
        self.motionnet_kld_bool = checkpoint['motionnet_kld_bool']
        self.rmse_weighting_startepoch = checkpoint['rmse_weighting_startepoch']
        self.rmse_weighting_vec_meter = checkpoint['rmse_weighting_vec_meter']
        self.rmse_weighting_vec = checkpoint['rmse_weighting_vec']
        self.latent_recon_loss = checkpoint['latent_recon_loss']
        self.recon_loss_power = checkpoint['recon_loss_power']

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
                'recon_weight': self.recon_weight,
                'pose_latent_gradient': self.pose_latent_gradient,
                'recon_gradient': self.recon_gradient,
                'classification_weight': self.classification_weight,
                'posenet_kld': self.posenet_kld,
                'motionnet_kld': self.motionnet_kld,
                'posenet_kld_bool': self.posenet_kld_bool,
                'motionnet_kld_bool': self.motionnet_kld_bool,
                'rmse_weighting_startepoch': self.rmse_weighting_startepoch,
                'rmse_weighting_vec_meter': self.rmse_weighting_vec_meter,
                'rmse_weighting_vec': self.rmse_weighting_vec,
                'latent_recon_loss': self.latent_recon_loss,
                'recon_loss_power': self.recon_loss_power
            }, self.save_chkpt_path)

            print('Stored ckpt at {}'.format(self.save_chkpt_path))

    def loss_function(self, recon_info, class_info, pose_stats, motion_stats, nan_masks, label_masks):
        x, recon_motion = recon_info
        labels, pred_labels = class_info
        pose_z_seq, recon_pose_z_seq, pose_mu, pose_logvar = pose_stats
        motion_z, motion_mu, motion_logvar = motion_stats

        # Posenet kld
        posenet_kld_multiplier = self._get_interval_multiplier(self.posenet_kld)
        posenet_kld_loss_indicator = -0.5 * torch.mean(1 + pose_logvar - pose_mu.pow(2) - pose_logvar.exp())
        posenet_kld_loss = posenet_kld_multiplier * posenet_kld_loss_indicator

        # Motionnet kld
        motionnet_kld_multiplier = self._get_interval_multiplier(self.motionnet_kld)
        motionnet_kld_loss_indicator = -0.5 * torch.mean(1 + motion_logvar - motion_mu.pow(2) - motion_logvar.exp())
        motionnet_kld_loss = motionnet_kld_multiplier * motionnet_kld_loss_indicator

        # Recon loss
        diff = x - recon_motion
        recon_loss_indicator = torch.sum(self.rmse_weighting_vec * nan_masks * (diff ** 2))  # For evaluation
        power = self._get_step_multiplier(self.recon_loss_power)
        power_diff = torch.sum(self.rmse_weighting_vec * nan_masks * torch.pow(diff, power))
        recon_loss = self.recon_weight * power_diff  # For error propagation

        # Latent recon loss
        squared_pose_z_seq = ((pose_z_seq - recon_pose_z_seq) ** 2)
        recon_latent_loss_indicator = torch.mean(squared_pose_z_seq)
        recon_latent_loss = 0 if self.latent_recon_loss is None else self.latent_recon_loss * recon_latent_loss_indicator

        # Gradient loss
        nan_mask_negibour_sum = self._calc_gradient_sum(nan_masks)
        gradient_mask = (nan_mask_negibour_sum == 2).float()  # If the adjacent entries are both 1
        recon_grad_loss_indicator = torch.mean(gradient_mask * self._calc_gradient(recon_motion))
        pose_latent_grad_loss_indicator = torch.mean(self._calc_gradient(pose_z_seq))
        recon_grad_loss = self.recon_gradient * recon_grad_loss_indicator
        pose_latent_grad_loss = self.pose_latent_gradient * pose_latent_grad_loss_indicator

        # Classification loss
        class_loss_indicator, acc = self._get_classification_acc(pred_labels, labels, label_masks)
        class_loss = self.classification_weight * class_loss_indicator

        # Combine different losses
        ## KLD has to be set to 0 manually if it is turned off, otherwise it is not numerically stable
        motionnet_kld_loss = 0 if self.motionnet_kld is None else motionnet_kld_loss
        posenet_kld_loss = 0 if self.posenet_kld is None else posenet_kld_loss
        loss = recon_loss + posenet_kld_loss + motionnet_kld_loss + recon_grad_loss + pose_latent_grad_loss + recon_latent_loss + class_loss

        return loss, (
            recon_loss_indicator, posenet_kld_loss_indicator, motionnet_kld_loss_indicator, recon_grad_loss_indicator,
            pose_latent_grad_loss_indicator, acc)

    def _model_initialization(self):
        if self.model_type == "graph":
            model_class = GraphSpatioTemporalVAE
        elif self.model_type == "normal":
            model_class = SpatioTemporalVAE
        else:
            print('Enter either "graph" or "normal" as the argument of model_type')
        model = model_class(
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

    def _initilize_rmse_weighting_vec(self):
        unnormalized = torch.ones(self.data_gen.batch_shape).float().to(self.device)
        normalized = torch.mean(unnormalized, dim=0, keepdim=True) / torch.sum(unnormalized)
        mean_rmse_meter = RunningAverageMeter()
        mean_rmse_meter.update(normalized)
        return normalized, mean_rmse_meter

    def _update_rmse_weighting_vec(self, squared_diff):
        normalized = torch.mean(squared_diff, dim=0, keepdim=True) / torch.sum(squared_diff)
        self.rmse_weighting_vec_meter.update(normalized)

    def _get_interval_multiplier(self, quantity_arg):
        """

        Parameters
        ----------
        quantity_arg : int or float or list
            Multiplier. If list, e.g. [50, 100, 0.1], then the function returns 0 before self.epoch < 50,
            the returned value linearly increases from 0 to 0.1 between 50th and 100th epoch, and remains as 0.1 after self.epoch > 100
        Returns
        -------
        quantity_multiplier : int or float

        """

        if quantity_arg is None:
            quantity_multiplier = 0
        elif isinstance(quantity_arg, list):
            start, end, const = quantity_arg[0], quantity_arg[1], quantity_arg[2]
            if self.epoch < start:
                quantity_multiplier = 0
            elif (self.epoch >= start) and (self.epoch < end):
                quantity_multiplier = const * ((self.epoch - start) / (end - start))
            elif self.epoch >= end:
                quantity_multiplier = const
        elif isinstance(quantity_arg, int) or isinstance(quantity_arg, float):
            quantity_multiplier = quantity_arg
        return quantity_multiplier

    def _get_step_multiplier(self, quantity_arg):
        """

        Parameters
        ----------
        quantity_arg : int or float or list
            if list, e.g. [2, 4, 50], it returns 2 before 50th epoch, and returns 4 at or after 50th epoch
            if int  or float, e.g. 2, it always returns 2
        Returns
        -------
        quantity_multiplier : int or float
        """
        quantity_muliiplier = None
        if isinstance(quantity_arg, int) or isinstance(quantity_arg, float):
            quantity_muliiplier = quantity_arg
        elif isinstance(quantity_arg, list) and len(quantity_arg) == 3:
            if self.epoch < quantity_arg[2]:
                quantity_muliiplier = quantity_arg[0]
            elif self.epoch >= quantity_arg[2]:
                quantity_muliiplier = quantity_arg[1]

        return quantity_muliiplier

    @staticmethod
    def _calc_gradient(x):
        grad = torch.abs(x[:, :, 0:127] - x[:, :, 1:])
        return grad

    @staticmethod
    def _calc_gradient_sum(x):
        grad = x[:, :, 0:127] + x[:, :, 1:]
        return grad

    def _get_classification_acc(self, pred_labels, labels, label_masks):
        class_loss_indicator_vec = self.class_criterion(pred_labels, labels)
        # class_loss_indicator = torch.mean(class_loss_indicator_vec)
        class_loss_indicator = torch.mean(label_masks * class_loss_indicator_vec)
        if class_loss_indicator is None:
            import pdb
            pdb.set_trace()
        pred_labels_np, labels_np = pred_labels.cpu().detach().numpy(), labels.cpu().detach().numpy()
        label_masks_np = label_masks.cpu().detach().numpy()
        acc = np.mean(np.argmax(pred_labels_np[label_masks_np > 0.5,], axis=1) == labels_np[label_masks_np > 0.5]) * 100
        return class_loss_indicator, acc

    def save_model_losses_data(self, project_dir, model_identifier):
        import pandas as pd
        loss_data = self.loss_meter.get_recorders()
        df_losses = pd.DataFrame(loss_data)
        df_losses.to_csv(os.path.join(project_dir, "vis", model_identifier, "loss_{}.csv".format(model_identifier)))

    def vis_reconstruction(self, data_gen, vid_sample_num, project_dir, model_identifier):
        plotting_mode = [True, True, True]
        # Define paths
        save_vid_dir = os.path.join(project_dir, "vis", model_identifier)
        if os.path.isdir(save_vid_dir) is not True:
            os.makedirs(save_vid_dir)

        # Refresh data generator
        self.data_gen = data_gen

        x_train_phenos_list, tasks_train_list, phenos_train_list = [], [], []

        # Get data from data generator's first loop
        for train_data, test_data in self.data_gen.iterator():
            x_train_fit, x_masks_train, tasks_train, task_masks_train, phenos_train, pheno_masks_train = train_data

            masks_train = (task_masks_train == 1) & (pheno_masks_train == 1)

            x_train, tasks_train, phenos_train = x_train_fit[masks_train,].copy(), tasks_train[masks_train,], \
                                                 phenos_train[
                                                     masks_train]

            # Produce phenos of equal/similar amounts
            uniphenos, phenos_counts = np.unique(phenos_train, return_counts=True)
            max_counts = np.sort(phenos_counts)[3]
            # max_counts = 100
            for pheno_idx in range(13):
                x_train_each_pheno = x_train[phenos_train == pheno_idx,]
                tasks_train_each_pheno = tasks_train[phenos_train == pheno_idx,]
                phenos_train_each_pheno = phenos_train[phenos_train == pheno_idx,]

                x_train_phenos_list.append(x_train_each_pheno[0:max_counts, ])
                tasks_train_list.append(tasks_train_each_pheno[0:max_counts, ])
                phenos_train_list.append(phenos_train_each_pheno[0:max_counts, ])

            x_equal_pheno = np.vstack(x_train_phenos_list)
            tasks_equal_pheno = np.concatenate(tasks_train_list)
            phenos_equal_pheno = np.concatenate(phenos_train_list)
            np.random.seed(50)
            ran_vec = np.random.permutation(x_equal_pheno.shape[0])
            x_equal_pheno, tasks_equal_pheno, phenos_equal_pheno = x_equal_pheno[ran_vec,], tasks_equal_pheno[ran_vec,], \
                                                                   phenos_equal_pheno[ran_vec,]

            # Produce base points
            x_base, tasks_base, phenos_base = x_train[0:4096, ], tasks_train[0:4096, ], phenos_train[0:4096, ]

            break

        # Convert to tensor
        x_equal_pheno = torch.from_numpy(x_equal_pheno).float().to(self.device)
        x_base = torch.from_numpy(x_base).float().to(self.device)

        # Forward pass
        self.model.eval()
        with torch.no_grad():
            # For projection
            recon_motion, pred_tasks, pose_z_info, motion_z_info = self.model(x_equal_pheno)
            pose_z_seq, recon_pose_z_seq, _, _ = pose_z_info
            motion_z, _, _ = motion_z_info

            # Base
            recon_motion_base, pred_tasks_base, pose_z_info_base, motion_z_info_base = self.model(x_base)
            pose_z_seq_base, recon_pose_z_seq_base, _, _ = pose_z_info_base
            motion_z_base, _, _ = motion_z_info_base

        # Fit umap embedding with
        vis = LatentSpaceVideoVisualizer(model_identifier=model_identifier, save_vid_dir=save_vid_dir)
        vis.fit_umap(pose_z_seq=pose_z_seq_base, motion_z=motion_z_base)

        # Transform motion_z to motion_z_umap
        motion_z = motion_z.cpu().detach().numpy()
        motion_z_base = motion_z_base.cpu().detach().numpy()
        motion_z_umap = vis.motion_z_umapper.transform(motion_z)
        motion_z_base_umap = vis.motion_z_umapper.transform(motion_z_base)
        save_vis_data_for_interactiveplot(x=x_base.cpu().detach().numpy(),
                                          recon=recon_motion_base.cpu().detach().numpy(),
                                          motion_z_umap=motion_z_base_umap,
                                          pheno_labels=phenos_base,
                                          tasks_labels=tasks_base,
                                          save_data_dir="/mnt/JupyterNotebook/interactive_latent_exploration/data")


        return

        # "pheno" labels
        vis.visualization_wrapper(x=x_equal_pheno, recon_motion=recon_motion, labels=phenos_equal_pheno,
                                  pred_labels=phenos_equal_pheno,
                                  motion_z_umap=motion_z_umap, pose_z_seq=pose_z_seq,
                                  recon_pose_z_seq=recon_pose_z_seq,
                                  test_acc=self.loss_meter.get_meter_avg()["test_acc"], mode="train", sample_num=25,
                                  label_type="pheno", plotting_mode=plotting_mode, motion_z_base=motion_z_base)
        vis.visualization_wrapper(x=x_equal_pheno, recon_motion=recon_motion, labels=tasks_equal_pheno,
                                  pred_labels=pred_tasks,
                                  motion_z_umap=motion_z_umap, pose_z_seq=pose_z_seq,
                                  recon_pose_z_seq=recon_pose_z_seq,
                                  test_acc=self.loss_meter.get_meter_avg()["test_acc"], mode="train", sample_num=25,
                                  label_type="task", plotting_mode=plotting_mode, motion_z_base=motion_z_base)

    def evaluate_all_models(self, data_gen, project_dir, model_list, draw_vid=False):

        def cale_additional_stasts(x, recon_motion, nan_masks):
            x_np, recon_motion_np, nan_masks_np = x.cpu().detach().numpy(), recon_motion.cpu().detach().numpy(), nan_masks.cpu().detach().numpy()
            nan_masks_np[nan_masks_np == 0] = np.nan  # convert 0 to nan, s.t. they can be masked
            masked_sq_diff = nan_masks_np * ((x_np - recon_motion_np) ** 2)
            joint_RSE = np.nanmean(np.nanmean(masked_sq_diff, axis=2), axis=0)
            rmse_q99 = np.nanquantile(np.nanmean(np.nanmean(masked_sq_diff, axis=2), axis=1), 0.99)
            return rmse_q99, joint_RSE, np.nanmean(masked_sq_diff)

        # Define the paths of the set of models to be evaluated
        print("Entering eval mode. Now models will be reloaded")
        model_subset = True if isinstance(model_list, list) and len(model_list) > 0 else False
        models_dir = os.path.join(project_dir, "model_chkpt")
        model_paths = []
        if model_subset:
            print("A subset of models are selected =\n", model_list)
            for model_name in model_subset:
                model_paths.append(os.path.join(models_dir, "ckpt_%s.pth" % model_name))
        else:
            print("Searching for all models in {}".format(models_dir))
            model_paths = glob(os.path.join(models_dir, "ckpt*.pth"))
        eval_results_dir = os.path.join(project_dir, "evaluation")
        if os.path.isdir(eval_results_dir) is not True:
            os.makedirs(eval_results_dir)

        # Refresh data generator
        self.data_gen = data_gen

        for idx, model_path in enumerate(model_paths):

            # Load model
            self.load_chkpt_path = model_path
            self.model, self.optimizer, self.lr_scheduler = self._load_model()
            self.model.eval()
            model_file_name = os.path.split(model_path)[1]
            model_identifier = re.match("ckpt_(.*?).pth", model_file_name).group(1)
            save_vid_dir = os.path.join(project_dir, "vis", model_identifier)
            if os.path.isdir(save_vid_dir) is not True:
                os.makedirs(save_vid_dir)
            print("Now evalulating model {}\n".format(model_identifier))
            batch_idx = 1

            # Quantities to evaluate
            eval_meter = MeterAssembly(
                "train_RMSE",
                "train_q99_RMSE",
                "train_joint_RSE",
                "train_recon_grad",
                "train_latent_grad",
                "train_acc",
                "test_RMSE",
                "test_q99_RMSE",
                "test_joint_RSE",
                "test_recon_grad",
                "test_latent_grad",
                "test_acc"
            )

            # Load batches of training/testing data
            for (x, nan_masks, labels, labels_mask), (
                    x_test, nan_masks_test, labels_test, labels_mask_test) in self.data_gen.iterator():
                print("\r Model {}/{} Current progress : {}/{}".format(idx, len(model_paths), batch_idx,
                                                                       self.data_gen.num_rows / self.data_gen.m),
                      flush=True, end="")

                # Convert numpy to torch.tensor
                x = torch.from_numpy(x).float().to(self.device)
                x_test = torch.from_numpy(x_test).float().to(self.device)
                labels = torch.from_numpy(labels).long().to(self.device)
                labels_test = torch.from_numpy(labels_test).long().to(self.device)
                nan_masks = torch.from_numpy(nan_masks * 1).float().to(self.device)
                nan_masks_test = torch.from_numpy(nan_masks_test * 1).float().to(self.device)

                # Forward pass
                with torch.no_grad():
                    recon_motion_t, pred_labels_t, pose_stats_t, motion_stats_t = self.model(x_test)
                    recon_info_t, class_info_t = (x_test, recon_motion_t), (labels_test, pred_labels_t)
                    loss_t, (
                        recon_t, posekld_t, motionkld_t, recongrad_t, latentgrad_t, acc_t) = self.loss_function(
                        recon_info_t,
                        class_info_t,
                        pose_stats_t,
                        motion_stats_t,
                        nan_masks_test
                    )
                    recon_motion, pred_labels, pose_stats, motion_stats = self.model(x)
                    recon_info, class_info = (x, recon_motion), (labels, pred_labels)
                    loss, (recon, posekld, motionkld, recongrad, latentgrad, acc) = self.loss_function(recon_info,
                                                                                                       class_info,
                                                                                                       pose_stats,
                                                                                                       motion_stats,
                                                                                                       nan_masks)

                # draw videos if enabled
                if draw_vid and (batch_idx == 1):
                    (pose_z_seq, recon_pose_z_seq, pose_mu, pose_logvar) = pose_stats
                    (pose_z_seq_test, recon_pose_z_seq_test, pose_mu_test, pose_logvar_test) = pose_stats_t
                    (motion_z, motion_mu, motion_logvar) = motion_stats
                    (motion_z_test, motion_mu_test, motion_logvar_test) = motion_stats_t

                    # Videos and Umap plots for Train data
                    gen_videos(x=x, recon_motion=recon_motion, motion_z=motion_z, pose_z_seq=pose_z_seq,
                               recon_pose_z_seq=recon_pose_z_seq, labels=labels.cpu().detach().numpy(),
                               pred_labels=pred_labels, test_acc=acc.item(),
                               sample_num=1,
                               save_vid_dir=save_vid_dir, model_identifier=model_identifier, mode="train")

                    # Videos and Umap plots for Test data
                    gen_videos(x=x_test, recon_motion=recon_motion_t, motion_z=motion_z_test,
                               pose_z_seq=pose_z_seq_test,
                               recon_pose_z_seq=recon_pose_z_seq_test,
                               labels=labels_test.cpu().detach().numpy(),
                               pred_labels=pred_labels_t, test_acc=acc_t.item(),
                               sample_num=1,
                               save_vid_dir=save_vid_dir, model_identifier=model_identifier, mode="test")

                rmse_q99, joint_RSE, rmse = cale_additional_stasts(x, recon_motion, nan_masks)
                rmse_q99_t, joint_RSE_t, rmse_t = cale_additional_stasts(x_test, recon_motion_t, nan_masks_test)

                # Add quantity of interest to lists
                eval_meter.append_recorders(
                    train_RMSE=rmse.item(),
                    train_q99_RMSE=rmse_q99,
                    train_joint_RSE=joint_RSE,
                    train_recon_grad=recongrad.item(),
                    train_latent_grad=latentgrad.item(),
                    train_acc=acc.item(),
                    test_RMSE=rmse_t.item(),
                    test_q99_RMSE=rmse_q99_t,
                    test_joint_RSE=joint_RSE_t,
                    test_recon_grad=recongrad_t.item(),
                    test_latent_grad=latentgrad_t.item(),
                    test_acc=acc_t.item()
                )
                batch_idx += 1
            print("\nEvaluating...")

            eval_dict = dict()
            for key in eval_meter.recorder.keys():
                arr_np = np.array(eval_meter.recorder[key])
                mean_val = np.nanmean(arr_np, axis=0)
                eval_dict[key] = mean_val.tolist()
            dict2json(os.path.join(eval_results_dir, "{}.json".format(model_file_name)), eval_dict)
