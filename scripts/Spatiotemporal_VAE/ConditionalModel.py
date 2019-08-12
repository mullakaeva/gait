from .Model import SpatioTemporalVAE, PoseVAE, MotionVAE, Unsqueeze, ClassNet, pose_block
from common.utils import TensorAssigner, TensorAssignerDouble, numpy_bool_index_select
import torch
import torch.nn as nn
import numpy as np


class ConditionalSpatioTemporalVAE(SpatioTemporalVAE):
    def __init__(self,
                 fea_dim=50,
                 seq_dim=128,
                 posenet_latent_dim=10,
                 posenet_dropout_p=0,
                 posenet_kld=True,
                 motionnet_latent_dim=25,
                 motionnet_hidden_dim=512,
                 motionnet_dropout_p=0,
                 motionnet_kld=True,
                 conditional_label_dim=0,
                 device=None
                 ):
        """
        This network takes input with shape (m, fea_dim, seq_dim), and reconstructs it, where m is number of samples.
        This network also does classification with the motion's latents. Number of classes is hard-coded as 8 (see below)

        Parameters
        ----------
        fea_dim : int
        seq_dim : int
        posenet_latent_dim : int
        posenet_dropout_p : float
        posenet_kld : bool
        motionnet_latent_dim : int
        motionnet_hidden_dim : int
        motionnet_dropout_p : float
        motionnet_kld : bool
        conditional_label_dim : int
            0 if conditional VAE is disabled. >0 specify the dimension of labels that will be concatenated to features
        """
        super(ConditionalSpatioTemporalVAE, self).__init__(
            fea_dim=fea_dim,
            seq_dim=seq_dim,
            posenet_latent_dim=posenet_latent_dim,
            posenet_dropout_p=posenet_dropout_p,
            posenet_kld=posenet_kld,
            motionnet_latent_dim=motionnet_latent_dim,
            motionnet_hidden_dim=motionnet_hidden_dim,
            motionnet_dropout_p=motionnet_dropout_p,
            motionnet_kld=motionnet_kld,
            device=device
        )
        self.conditional_label_dim = conditional_label_dim
        self.pose_vae = ConditionalPoseVAE(fea_dim=self.fea_dim,
                                           latent_dim=self.posenet_latent_dim,
                                           conditional_label_dim=self.conditional_label_dim,
                                           kld=self.posenet_kld,
                                           dropout_p=self.posenet_kld,
                                           device=self.device)

        self.motion_vae = ConditionalMotionVAE(fea_dim=self.posenet_latent_dim,
                                               seq_dim=self.seq_dim,
                                               hidden_dim=self.motionnet_hidden_dim,
                                               latent_dim=self.motionnet_latent_dim,
                                               conditional_label_dim=self.conditional_label_dim,
                                               kld=self.motionnet_kld,
                                               dropout_p=self.motionnet_dropout_p,
                                               device=self.device)
        self.class_net = ConditionalClassNet(input_dim=self.motionnet_latent_dim,
                                             conditional_label_dim=self.conditional_label_dim,
                                             n_classes=self.n_classes,
                                             device=self.device)

    def forward(self, *inputs):
        """
        Parameters
        ----------
        x : torch.tensor
            With shape (m, fea, seq)
        labels : torch.tensor
            With shape (m, label_dim, seq)
        """
        x, labels = inputs

        (pose_z_seq, pose_mu, pose_logvar), (motion_z, motion_mu, motion_logvar) = self.encode(x, labels)

        recon_motion, recon_pose_z_seq, concat_motion_z = self.decode(
            motion_z, labels
        )  # Convert (m, motion_latent_dim+label_dim) to (m, fea, seq)
        pred_labels = self.class_net(concat_motion_z)  # Convert (m, motion_latent_dim+label_dim) to (m, n_classes)
        return recon_motion, pred_labels, (pose_z_seq, recon_pose_z_seq, pose_mu, pose_logvar), (
            motion_z, motion_mu, motion_logvar)

    def encode(self, x, labels):
        # Concatenation of input for encoding
        if self.conditional_label_dim > 0:
            concat_x = torch.cat([x, labels], dim=1)
        else:
            concat_x = x
        # Propagtion
        out = self.transpose_flatten(concat_x)  # Convert (m, fea+label_dim, seq) to (m * seq, fea+label_dim)
        pose_out = self.pose_encode(
            out)  # Convert (m * seq, fea+label_dim) to (m * seq, pose_latent_dim (or *2 if kld=True) )
        pose_z, pose_mu, pose_logvar = self.pose_bottoleneck(pose_out)  # all outputs (m * seq, pose_latent_dim)
        pose_z_seq = self.unflatten_transpose(pose_z)  # Convert (m * seq, pose_latent_dim) to (m, pose_latent_dim, seq)
        out = self.motion_encode(
            pose_z_seq)  # Convert (m, pose_latent_dim, seq) to (m, motion_latent_dim (or *2 if kld=True) )
        motion_z, motion_mu, motion_logvar = self.motion_bottoleneck(out)  # all outputs (m, motion_latent_dim)
        return (pose_z_seq, pose_mu, pose_logvar), (motion_z, motion_mu, motion_logvar)

    def decode(self, motion_z, labels):
        # Concatenation of latents for decoding
        if self.conditional_label_dim > 0:
            concat_motion_z = torch.cat([motion_z, labels[:, :, 0]], dim=1)
        else:
            concat_motion_z = motion_z

        recon_pose_z_seq = self.motion_decode(
            concat_motion_z)  # Convert (m, motion_latent_dim) to  (m, pose_latent_dim, seq)
        out = self.transpose_flatten(
            recon_pose_z_seq)  # Convert (m, pose_latent_dim, seq) to (m * seq, pose_latent_dim)
        out = self.pose_decode(out)  # Convert (m * seq, pose_latent_dim) to (m * seq, fea)
        recon_motion = self.unflatten_transpose(out)  # Convert (m * seq, fea) to (m, fea+, seq)
        return recon_motion, recon_pose_z_seq, concat_motion_z


class ConditionalPoseVAE(PoseVAE):
    def __init__(self, fea_dim, latent_dim, conditional_label_dim, kld, dropout_p, device=None):
        super(ConditionalPoseVAE, self).__init__(
            fea_dim=fea_dim,
            latent_dim=latent_dim,
            kld=kld,
            dropout_p=dropout_p,
            device=device
        )
        self.conditional_label_dim = conditional_label_dim
        self.first_layer = nn.Linear(self.fea_dim + self.conditional_label_dim, self.encode_units[0])


class ConditionalMotionVAE(MotionVAE):
    def __init__(self, fea_dim=50, seq_dim=128, hidden_dim=1024, latent_dim=8, conditional_label_dim=0, kld=False,
                 dropout_p=0, device=None):
        super(ConditionalMotionVAE, self).__init__(
            fea_dim=fea_dim,
            seq_dim=seq_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            kld=kld,
            dropout_p=dropout_p,
            device=device
        )
        self.conditional_label_dim = conditional_label_dim
        self.latents2de = nn.Sequential(
            Unsqueeze(),
            nn.ConvTranspose1d(self.latent_dim + self.conditional_label_dim,
                               hidden_dim,
                               kernel_size=self.decoding_kernels[0],
                               stride=self.decoding_strides[0])
        )


class ConditionalClassNet(ClassNet):
    def __init__(self, input_dim, conditional_label_dim, n_classes, device=None):
        self.conditional_label_dim = conditional_label_dim
        super(ConditionalClassNet, self).__init__(input_dim=input_dim,
                                                  n_classes=n_classes,
                                                  device=device)
        self.first_layer = nn.Linear(self.input_dim + self.conditional_label_dim,
                                     self.encode_units[0])


class IdentifyNet(nn.Module):
    def __init__(self, num_phenos, motion_z_dim=128,
                 task_dim=8, fingerprint_dim=1024, hidden_dim=1024 * 2, device=None):
        # init
        super(IdentifyNet, self).__init__()
        self.num_phenos = num_phenos
        self.task_dim = task_dim
        self.motion_z_dim = motion_z_dim
        self.fingerprint_dim = fingerprint_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device('cuda:0') if device is None else device

        # Network
        self.layer_in = nn.Sequential(*pose_block(input_channels=self.fingerprint_dim,
                                                  output_channels=self.hidden_dim))
        self.layer2 = nn.Sequential(*pose_block(input_channels=self.hidden_dim,
                                                output_channels=self.hidden_dim))
        self.layer_out = nn.Sequential(*pose_block(input_channels=self.hidden_dim,
                                                   output_channels=self.num_phenos))

    def forward(self, *inputs):

        motion_z, tasks, tasks_mask, patient_ids, phenos, phenos_mask = inputs
        fingerprint, uni_phenos = self._transform_to_patient_task_means(motion_z, tasks, tasks_mask, patient_ids, phenos, phenos_mask)

        out = self.layer_in(fingerprint)
        out = self.layer2(out)
        out = self.layer_out(out)

        return out, uni_phenos
    def _transform_to_patient_task_means(self, motion_z, tasks, tasks_mask, patient_ids, phenos, phenos_mask):

        # Masking
        true_mask = (tasks_mask == 1) & (np.isnan(patient_ids) == False) & (phenos_mask == 1)

        # Slicing
        sliced_z = numpy_bool_index_select(tensor_arr=motion_z, mask=true_mask, device=self.device)
        tasks = tasks[true_mask]
        phenos = phenos[true_mask]
        patient_ids = patient_ids[true_mask]

        # Labels
        uni_patients = np.unique(patient_ids)
        num_uni_patients = uni_patients.shape[0]

        # Map patient to pheno's label
        uni_patient_phenos = []
        phenos_labels = []
        for p_id in range(uni_patients.shape[0]):
            patient_index = np.where(patient_ids == patient_ids[p_id])[0]
            phenos_id_each = phenos[patient_index]
            uni_patient_phenos.append((p_id, phenos_id_each[0]))
            phenos_labels.append(phenos_id_each[0])
        # Enabling gradient record on slice assignment
        patient_assigner = TensorAssignerDouble(size=(num_uni_patients, self.task_dim, self.motion_z_dim),
                                                device=self.device)
        task_assigner = TensorAssigner(size=(self.task_dim, self.motion_z_dim), device=self.device)

        # Calc grand means
        for i in range(self.task_dim):
            average_tasks = torch.mean(numpy_bool_index_select(tensor_arr=sliced_z, mask=(tasks == i), device=self.device),
                                       dim=0)

            task_assigner.assign(i, average_tasks)
        aver_tasks_all = task_assigner.get_fingerprint()
        # Calc patient's task's means
        for p_id, phenos_id in uni_patient_phenos:

            for j in range(self.task_dim):
                patient_task_mask = (tasks == j) & (patient_ids == patient_ids[p_id])
                if np.sum(patient_task_mask) > 0:
                    average_patient_task = torch.mean(numpy_bool_index_select(tensor_arr=sliced_z, mask=(tasks == j),
                                                                              device=self.device),
                                                      dim=0)
                else:
                    average_patient_task = aver_tasks_all[j,]
                patient_assigner.assign(p_id, j, average_patient_task)
        # Reshape
        fingerprint = patient_assigner.get_fingerprint()
        fingerprint = fingerprint.reshape(num_uni_patients, -1)
        return fingerprint, np.asarray(phenos_labels)


class ConditionalIdentifySpatioTemporalVAE(ConditionalSpatioTemporalVAE):
    def __init__(self,
                 fea_dim=50,
                 seq_dim=128,
                 num_phenos=13,
                 posenet_latent_dim=10,
                 posenet_dropout_p=0,
                 posenet_kld=True,
                 motionnet_latent_dim=25,
                 motionnet_hidden_dim=512,
                 motionnet_dropout_p=0,
                 motionnet_kld=True,
                 conditional_label_dim=0,
                 device=None
                 ):
        super(ConditionalIdentifySpatioTemporalVAE, self).__init__(
            fea_dim=fea_dim,
            seq_dim=seq_dim,
            posenet_latent_dim=posenet_latent_dim,
            posenet_dropout_p=posenet_dropout_p,
            posenet_kld=posenet_kld,
            motionnet_latent_dim=motionnet_latent_dim,
            motionnet_hidden_dim=motionnet_hidden_dim,
            motionnet_dropout_p=motionnet_dropout_p,
            motionnet_kld=motionnet_kld,
            conditional_label_dim=conditional_label_dim,
            device=device
        )
        self.identify_net = IdentifyNet(
            num_phenos = num_phenos,
            motion_z_dim=self.motionnet_latent_dim,
            task_dim=8,
            fingerprint_dim=self.motionnet_latent_dim*8,
            hidden_dim=num_phenos*2,
            device=self.device
        )

    def forward(self, *inputs):

        x, labels, tasks_np, tasks_mask_np, patient_ids_np, phenos_np, phenos_mask_np = inputs
        (pose_z_seq, pose_mu, pose_logvar), (motion_z, motion_mu, motion_logvar) = self.encode(x, labels)

        recon_motion, recon_pose_z_seq, concat_motion_z = self.decode(
            motion_z, labels
        )  # Convert (m, motion_latent_dim+label_dim) to (m, fea, seq)
        pred_labels = self.class_net(concat_motion_z)  # Convert (m, motion_latent_dim+label_dim) to (m, n_classes)

        # Identification net
        pred_identify, labels_identify = self.identify_net(motion_z, tasks_np, tasks_mask_np, patient_ids_np, phenos_np, phenos_mask_np)

        return recon_motion, pred_labels, (pose_z_seq, recon_pose_z_seq, pose_mu, pose_logvar), (
            motion_z, motion_mu, motion_logvar), (pred_identify, labels_identify)