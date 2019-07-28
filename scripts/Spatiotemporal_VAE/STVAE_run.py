import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn import CrossEntropyLoss

import numpy as np
import os
import matplotlib.pyplot as plt
import pprint
import pandas as pd
import umap
import re
from glob import glob
from common.utils import MeterAssembly, RunningAverageMeter, dict2json, tensor2numpy, numpy2tensor, load_df_pickle, \
    write_df_pickle, expand1darr
from common.visualisation import LatentSpaceVideoVisualizer
from common.data_preparation import prepare_data_for_concatenated_latent
from .Model import SpatioTemporalVAE
from .ConditionalModel import ConditionalSpatioTemporalVAE


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

                    train_converted, test_converted = self._convert_input_data(train_data, test_data)
                    x, nan_masks, labels, labels_mask = train_converted
                    x_test, labels_test, labels_mask_test, nan_masks_test = test_converted

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
        if self.model_type == "conditional":
            model_class = ConditionalSpatioTemporalVAE
        elif self.model_type == "normal":
            model_class = SpatioTemporalVAE
        else:
            print('Enter either "conditional" or "normal" as the argument of model_type')
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

    def _convert_input_data(self, train_data, test_data):
        # Unfolding
        x, nan_masks, tasks, tasks_mask, _, _, _ = train_data
        x_test, nan_masks_test, tasks_test, tasks_mask_test, _, _, _ = test_data

        # Convert numpy to torch.tensor
        x, x_test = numpy2tensor(self.device, x, x_test)
        tasks = torch.from_numpy(tasks).long().to(self.device)
        tasks_test = torch.from_numpy(tasks_test).long().to(self.device)
        tasks_mask = torch.from_numpy(tasks_mask * 1 + 1e-5).float().to(self.device)
        tasks_mask_test = torch.from_numpy(tasks_mask_test * 1 + 1e-5).float().to(self.device)
        nan_masks = torch.from_numpy(nan_masks * 1 + 1e-5).float().to(self.device)
        nan_masks_test = torch.from_numpy(nan_masks_test * 1 + 1e-5).float().to(self.device)

        return (x, nan_masks, tasks, tasks_mask), (x_test, tasks_test, tasks_mask_test, nan_masks_test)

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

    def _forward_pass(self, x):
        self.model.eval()
        with torch.no_grad():
            # For projection
            recon_motion, pred_tasks, pose_z_info, motion_z_info = self.model(x)
            pose_z_seq, recon_pose_z_seq, _, _ = pose_z_info
            motion_z, _, _ = motion_z_info
        return recon_motion, pose_z_seq, recon_pose_z_seq, motion_z

    def save_for_concatenated_latent_vis(self, df_path, save_data_dir):

        # Load data
        df_shuffled = prepare_data_for_concatenated_latent(df_path, equal_phenos=False)
        df_shuffled = df_shuffled.sample(frac=1, random_state=60).reset_index(drop=True)

        # Forward pass and add column
        x = np.asarray(list(df_shuffled["features"]))

        motion_z_list = []
        batch = 512
        batch_times = int(x.shape[0] / batch)
        for i in range(batch_times + 1):
            if i < batch_times:
                x_each = numpy2tensor(self.device, x[i * batch: (i + 1) * batch, ])[0]
            else:
                x_each = numpy2tensor(self.device, x[i * batch:, ])[0]
            _, _, _, motion_z_batch = self._forward_pass(x_each)
            motion_z_batch = tensor2numpy(motion_z_batch)[0]
            motion_z_list.append(motion_z_batch)

        motion_z = np.vstack(motion_z_list)
        df_shuffled["motion_z"] = list(motion_z)

        # Calc grand means for each tasks
        print("Calculating grand means")
        task_means = dict()
        for task_idx in range(8):
            mask = df_shuffled["tasks"] == task_idx
            task_means[task_idx] = np.mean(np.asarray(list(df_shuffled[mask].motion_z)), axis=0)

        # Calc means for each tasks in each patients
        print("Calculating patient's task's mean")
        all_patient_ids = np.unique(df_shuffled["idpatients"])
        num_patient_ids = all_patient_ids.shape[0]
        patient_id_list, features_list, phenos_list = [], [], []

        for p_idx in range(num_patient_ids):
            print("\rpatient {}/{}".format(p_idx, num_patient_ids), flush=True, end="")
            patient_id = all_patient_ids[p_idx]
            patient_mask = df_shuffled["idpatients"] == patient_id
            unique_tasks = np.unique(df_shuffled[patient_mask]["tasks"])
            unique_phenos = np.unique(np.concatenate(list(df_shuffled[patient_mask]["phenos"])))
            task_vec_list = []

            for task_idx in range(8):

                if task_idx not in unique_tasks:
                    task_vec_list.append(task_means[task_idx])
                else:
                    mask = (df_shuffled["idpatients"] == patient_id) & (df_shuffled["tasks"] == task_idx)
                    patient_task_mean = np.mean(np.asarray(list(df_shuffled[mask]["motion_z"])), axis=0)
                    task_vec_list.append(patient_task_mean)
            task_vec = np.concatenate(task_vec_list)

            patient_id_list.append(patient_id)
            features_list.append(task_vec)
            phenos_list.append(unique_phenos)

        df_concat = pd.DataFrame({"patient_id": patient_id_list, "fingerprint": features_list,
                                  "phenos": phenos_list})
        print("Umapping")
        self.fingerprint_umapper = umap.UMAP(n_neighbors=15,
                                             n_components=2,
                                             min_dist=0.1,
                                             metric="euclidean")
        fingerprint_z = self.fingerprint_umapper.fit_transform(np.asarray(list(df_concat["fingerprint"])))
        df_concat["fingerprint_z"] = list(fingerprint_z)
        print("Saving")
        write_df_pickle(df_concat, os.path.join(save_data_dir, "concat_fingerprint.pickle"))


class CSTVAEmodel(STVAEmodel):
    def __init__(self,
                 data_gen,
                 fea_dim=50,
                 seq_dim=128,
                 conditional_label_dim=0,  # New
                 model_type="conditional",
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

        self.conditional_label_dim = conditional_label_dim
        super(CSTVAEmodel, self).__init__(
            data_gen=data_gen,
            fea_dim=fea_dim,
            seq_dim=seq_dim,
            model_type=model_type,
            posenet_latent_dim=posenet_latent_dim,
            posenet_dropout_p=posenet_dropout_p,
            posenet_kld=posenet_kld,
            motionnet_latent_dim=motionnet_latent_dim,
            motionnet_hidden_dim=motionnet_hidden_dim,
            motionnet_dropout_p=motionnet_dropout_p,
            motionnet_kld=motionnet_kld,
            recon_weight=recon_weight,
            pose_latent_gradient=pose_latent_gradient,
            recon_gradient=recon_gradient,
            classification_weight=classification_weight,
            rmse_weighting_startepoch=rmse_weighting_startepoch,
            latent_recon_loss=latent_recon_loss,
            recon_loss_power=recon_loss_power,
            gpu=gpu,
            init_lr=init_lr,
            lr_milestones=lr_milestones,
            lr_decay_gamma=lr_decay_gamma,
            save_chkpt_path=save_chkpt_path,
            load_chkpt_path=load_chkpt_path,
        )

    def _model_initialization(self):
        if self.model_type == "conditional":
            model_class = ConditionalSpatioTemporalVAE
        elif self.model_type == "normal":
            model_class = SpatioTemporalVAE
        else:
            print('Enter either "conditional" or "normal" as the argument of model_type')
        model = model_class(
            fea_dim=self.fea_dim,
            seq_dim=self.seq_dim,
            posenet_latent_dim=self.posenet_latent_dim,
            posenet_dropout_p=self.posenet_dropout_p,
            posenet_kld=self.posenet_kld_bool,
            motionnet_latent_dim=self.motionnet_latent_dim,
            motionnet_hidden_dim=self.motionnet_hidden_dim,
            motionnet_dropout_p=self.motionnet_dropout_p,
            motionnet_kld=self.motionnet_kld_bool,
            conditional_label_dim=self.conditional_label_dim
        ).to(self.device)
        params = model.parameters()
        optimizer = optim.Adam(params, lr=self.init_lr)
        lr_scheduler = MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=self.lr_decay_gamma)
        return model, optimizer, lr_scheduler

    def train(self, n_epochs=50):
        try:
            for epoch in range(n_epochs):
                iter_idx = 0

                for train_data, test_data in self.data_gen.iterator():

                    train_converted, test_converted = self._convert_input_data(train_data, test_data)
                    x, nan_masks, labels, labels_mask, cond = train_converted
                    x_test, nan_masks_test, labels_test, labels_mask_test, cond_test = test_converted

                    # Clear optimizer's previous gradients
                    self.optimizer.zero_grad()

                    # CV set
                    self.model.eval()
                    with torch.no_grad():
                        recon_motion_t, pred_labels_t, pose_stats_t, motion_stats_t = self.model(x_test, cond_test)
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
                    recon_motion, pred_labels, pose_stats, motion_stats = self.model(x, cond)
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
        self.conditional_label_dim = checkpoint['conditional_label_dim']  # New
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
                'conditional_label_dim': self.conditional_label_dim,  # New
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

    def _forward_pass(self, x, towards2d):
        self.model.eval()
        with torch.no_grad():
            # For projection
            towards2d = numpy2tensor(self.device, towards2d)[0]
            recon_motion, pred_tasks, pose_z_info, motion_z_info = self.model(x, towards2d)
            pose_z_seq, recon_pose_z_seq, _, _ = pose_z_info
            motion_z, _, _ = motion_z_info
        return recon_motion, pose_z_seq, recon_pose_z_seq, motion_z

    def _convert_input_data(self, train_data, test_data):
        x, nan_masks, tasks, tasks_mask, _, _, towards = train_data
        x_test, nan_masks_test, tasks_test, tasks_mask_test, _, _, towards_test = test_data

        # Convert numpy to torch.tensor
        x, x_test = numpy2tensor(self.device, x, x_test)
        tasks = torch.from_numpy(tasks).long().to(self.device)
        tasks_test = torch.from_numpy(tasks_test).long().to(self.device)
        tasks_mask = torch.from_numpy(tasks_mask * 1 + 1e-5).float().to(self.device)
        tasks_mask_test = torch.from_numpy(tasks_mask_test * 1 + 1e-5).float().to(self.device)
        nan_masks = torch.from_numpy(nan_masks * 1 + 1e-5).float().to(self.device)
        nan_masks_test = torch.from_numpy(nan_masks_test * 1 + 1e-5).float().to(self.device)
        towards, towards_test = numpy2tensor(self.device,
                                             expand1darr(towards.astype(np.int64), 3, self.seq_dim),
                                             expand1darr(towards_test.astype(np.int64), 3, self.seq_dim)
                                             )
        return (x, nan_masks, tasks, tasks_mask, towards), (
            x_test, nan_masks_test, tasks_test, tasks_mask_test, towards_test)

    def save_for_concatenated_latent_vis(self, df_path, save_data_dir):

        # Load data
        df_shuffled = prepare_data_for_concatenated_latent(df_path, equal_phenos=False)
        df_shuffled = df_shuffled.sample(frac=1, random_state=60).reset_index(drop=True)

        # Forward pass and add column
        x = np.asarray(list(df_shuffled["features"]))
        directions = np.asarray(list(df_shuffled["directions"]))

        motion_z_list = []
        batch = 512
        batch_times = int(x.shape[0] / batch)
        for i in range(batch_times + 1):
            if i < batch_times:
                x_each = numpy2tensor(self.device, x[i * batch: (i + 1) * batch, ])[0]
                direction_each = directions[i * batch: (i + 1) * batch, ]
            else:
                x_each = numpy2tensor(self.device, x[i * batch:, ])[0]
                direction_each = directions[i * batch:, ]

            _, _, _, motion_z_batch = self._forward_pass(x_each, direction_each)
            motion_z_batch = tensor2numpy(motion_z_batch)[0]
            motion_z_list.append(motion_z_batch)

        motion_z = np.vstack(motion_z_list)
        df_shuffled["motion_z"] = list(motion_z)
        del df_shuffled["features"]  # The "features" column is not needed anymore after forward pass
        df_aver = df_shuffled.groupby("avg_idx", as_index=False).apply(
            np.mean)  # Average across avg_idx. They are split from the same video.
        del df_aver["avg_idx"]

        # Calc grand means for each tasks
        print("Calculating grand means")
        task_means = dict()
        for task_idx in range(8):
            mask = df_aver["tasks"] == task_idx
            task_means[task_idx] = np.mean(np.asarray(list(df_aver[mask].motion_z)), axis=0)

        # Calc means for each tasks in each patients
        print("Calculating patient's task's mean")
        all_patient_ids = np.unique(df_aver["idpatients"])
        num_patient_ids = all_patient_ids.shape[0]
        patient_id_list, features_list, phenos_list = [], [], []

        for p_idx in range(num_patient_ids):
            print("\rpatient {}/{}".format(p_idx, num_patient_ids), flush=True, end="")
            patient_id = all_patient_ids[p_idx]
            patient_mask = df_aver["idpatients"] == patient_id
            unique_tasks = np.unique(df_aver[patient_mask]["tasks"])
            unique_phenos = np.unique(np.concatenate(list(df_aver[patient_mask]["phenos"])))
            task_vec_list = []

            for task_idx in range(8):

                if task_idx not in unique_tasks:
                    task_vec_list.append(task_means[task_idx])
                else:
                    mask = (df_aver["idpatients"] == patient_id) & (df_aver["tasks"] == task_idx)
                    patient_task_mean = np.mean(np.asarray(list(df_aver[mask]["motion_z"])), axis=0)
                    task_vec_list.append(patient_task_mean)
            task_vec = np.concatenate(task_vec_list)

            patient_id_list.append(patient_id)
            features_list.append(task_vec)
            phenos_list.append(unique_phenos)

        df_concat = pd.DataFrame({"patient_id": patient_id_list, "fingerprint": features_list,
                                  "phenos": phenos_list})
        print("Umapping")
        self.fingerprint_umapper = umap.UMAP(n_neighbors=15,
                                             n_components=2,
                                             min_dist=0.1,
                                             metric="euclidean")
        fingerprint_z = self.fingerprint_umapper.fit_transform(np.asarray(list(df_concat["fingerprint"])))
        df_concat["fingerprint_z"] = list(fingerprint_z)
        print("Saving")
        write_df_pickle(df_concat, os.path.join(save_data_dir, "concat_fingerprint_CB-K-0.0001.pickle"))


class CtaskSVAEmodel(CSTVAEmodel):
    def _convert_input_data(self, train_data, test_data):
        x, nan_masks, tasks, tasks_mask, _, _, towards = train_data
        x_test, nan_masks_test, tasks_test, tasks_mask_test, _, _, towards_test = test_data

        # Select those data with task labels
        train_mask = tasks_mask == True
        test_mask = tasks_mask_test == True
        x, nan_masks, tasks, tasks_mask, towards = x[train_mask], nan_masks[train_mask], tasks[train_mask], tasks_mask[
            train_mask], towards[train_mask]
        x_test, nan_masks_test, tasks_test, tasks_mask_test, towards_test = x_test[test_mask], nan_masks_test[
            test_mask], tasks_test[test_mask], tasks_mask_test[test_mask], towards_test[test_mask]

        # Convert numpy to torch.tensor
        x, x_test = numpy2tensor(self.device, x, x_test)

        tasks_input = torch.from_numpy(tasks).long().to(self.device)
        tasks_test_input = torch.from_numpy(tasks_test).long().to(self.device)
        tasks_mask_input = torch.from_numpy(tasks_mask * 1 + 1e-5).float().to(self.device)
        tasks_mask_test_input = torch.from_numpy(tasks_mask_test * 1 + 1e-5).float().to(self.device)

        nan_masks = torch.from_numpy(nan_masks * 1 + 1e-5).float().to(self.device)
        nan_masks_test = torch.from_numpy(nan_masks_test * 1 + 1e-5).float().to(self.device)
        towards, towards_test = numpy2tensor(self.device,
                                             expand1darr(towards.astype(np.int64), 3, self.seq_dim),
                                             expand1darr(towards_test.astype(np.int64), 3, self.seq_dim)
                                             )

        # concatenate between task labels and direction labels
        tasks_onehot, tasks_test_onehot = numpy2tensor(self.device,
                                                       expand1darr(tasks.astype(np.int64), 8, self.seq_dim),
                                                       expand1darr(tasks_test.astype(np.int64), 8, self.seq_dim)
                                                       )

        cond = torch.cat([towards, tasks_onehot], dim=1)
        cond_test = torch.cat([towards_test, tasks_test_onehot], dim=1)

        return (x, nan_masks, tasks_input, tasks_mask_input, cond), (
            x_test, nan_masks_test, tasks_test_input, tasks_mask_test_input, cond_test)
