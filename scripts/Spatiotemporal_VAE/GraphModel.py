from .Model import *
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv

openpose_body_connection_scheme = (
    (0, 1),  # nose to neck
    (0, 15),  # nose to r_eye
    (0, 16),  # nose to l_eye
    (15, 17),  # r_eye to r_ear
    (16, 18),  # l_eye to l_ear
    (18, 1),  # l_ear to neck
    (17, 1),  # r_ear to neck
    (1, 5),  # neck to l_shoulder
    (5, 6),  # l_shoulder to l_elbow
    (6, 7),  # l_elbow to l_wrist
    (1, 2),  # neck to r_shoulder
    (2, 3),  # r_shoulder to r_elbow
    (3, 4),  # r_elbow to r_wrist
    (1, 8),  # neck to hip_centre
    (8, 9),  # hip_centre to r_hip
    (9, 10),  # r_hip to r_knee
    (10, 11),  # r_knee to r_ankle
    (11, 24),  # r_ankle to r_heel
    (11, 22),  # r_ankle to r_bigtoe
    (22, 23),  # r_bigtoe to r_smalltoe
    (8, 12),  # hip_centre to l_hip
    (12, 13),  # l_hip to l_knee
    (13, 14),  # l_knee to l_ankle
    (14, 21),  # l_ankle to l_heel
    (14, 19),  # l_ankle to l_bigtoe
    (19, 20)  # l_bigtoe to l_small toe
)


def construct_edge_indices():
    """

    Returns
    -------
    edge_index : torch.tensor
        Tensor with shape (2, num_edges), with dtype=long. Pre-defined (2, 50)
    """
    num_edges = len(openpose_body_connection_scheme)
    edge_index = torch.zeros((2, num_edges * 2)).long()
    for idx, edge_tuple in enumerate(openpose_body_connection_scheme):
        edge_np = np.array(edge_tuple)
        edge_np_reverse = np.array(edge_tuple)[::-1]

        edge_tensor = torch.from_numpy(np.array(edge_np)).long()
        edge_tensor_reverse = torch.from_numpy(np.array(edge_np_reverse)).long()

        edge_index[:, idx] = edge_tensor
        edge_index[:, num_edges + idx] = edge_tensor_reverse
    return edge_index


def construct_batch_edge_index(x, edge_index):
    """

    Parameters
    ----------
    x : torch.tensor
        Shape = (m, num_nodes, node_fea)
    edge_index : torch.tensor
        Shape = (2, num_edges), data type = long
    Returns
    -------
    x_flat : torch.tensor
        Shape = (m * num_nodes, node_fea)
    all_edge_index : torch.tensor
        Shape = (2, num_edges * m), data type = long
    """

    single_num_edges = edge_index.shape[1]
    num_samples = x.shape[0]

    # Flatten x
    x_flat = x.view(-1, x.shape[2])

    # Repeat Edge index
    all_edge_index = torch.zeros((2, num_samples * single_num_edges))
    all_edge_index[:, 0:single_num_edges] = edge_index.clone()

    for i in range(num_samples):
        new_edge_index = (edge_index + 25 * i)
        all_edge_index[:, i * single_num_edges: (i + 1) * single_num_edges] = new_edge_index
    return x_flat, all_edge_index.long().to(x.device)


def activation_block(output_channels,
                     dropout_p=0):
    bn_layer = nn.BatchNorm1d(output_channels, track_running_stats=False)
    relu_layer = nn.ReLU()
    droput_layer = nn.Dropout(dropout_p)
    block_list = [
        bn_layer,
        relu_layer,
        droput_layer
    ]
    return block_list


class GraphSpatioTemporalVAE(nn.Module):
    def __init__(self,
                 fea_dim=25,
                 seq_dim=128,
                 posenet_latent_dim=10,
                 posenet_dropout_p=0,
                 posenet_kld=True,
                 motionnet_latent_dim=25,
                 motionnet_hidden_dim=512,
                 motionnet_dropout_p=0,
                 motionnet_kld=True,
                 device=None
                 ):

        # # Loading parameters
        # Data dimension
        self.fea_dim = fea_dim
        self.seq_dim = seq_dim
        self.n_classes = 8
        # Autoencoder for single pose
        self.posenet_latent_dim = posenet_latent_dim
        self.posenet_dropout_p = posenet_dropout_p
        self.posenet_kld = posenet_kld
        # Autoencoder for a sequence of poses (motion)
        self.motionnet_latent_dim = motionnet_latent_dim
        self.motionnet_hidden_dim = motionnet_hidden_dim
        self.motionnet_dropout_p = motionnet_dropout_p
        self.motionnet_kld = motionnet_kld
        # Others
        super(GraphSpatioTemporalVAE, self).__init__()
        self.device = torch.device('cuda:0') if device is None else device

        # # Initializing network architecture
        self.flatten_lastdim_layer = FlattenLastDim(dim=3)
        self.unflattent_lastdim_layer = UnflattenLastDim(2)
        self.transpose_layer = Transpose()
        self.flatten_dim2_layer = Flatten(dim=2)
        self.flatten_dim3_layer = Flatten(dim=3)
        self.unflatten_dim2_layer = UnFlatten(self.seq_dim, dim=2)
        self.unflatten_dim3_layer = UnFlatten(self.seq_dim, dim=3)

        self.pose_vae = GraphPoseVAE(node_dim=self.fea_dim,
                                     node_fea_dim=2,
                                     latent_dim=self.posenet_latent_dim,
                                     kld=self.posenet_kld,
                                     dropout_p=self.posenet_kld,
                                     device=self.device)

        self.motion_vae = MotionVAE(fea_dim=self.posenet_latent_dim,
                                    seq_dim=self.seq_dim,
                                    hidden_dim=self.motionnet_hidden_dim,
                                    latent_dim=self.motionnet_latent_dim,
                                    kld=self.motionnet_kld,
                                    dropout_p=self.motionnet_dropout_p,
                                    device=self.device)

        self.class_net = ClassNet(input_dim=self.motionnet_latent_dim,
                                  n_classes=self.n_classes,
                                  device=self.device)

    def forward(self, x):
        out = self.flatten_dim3_layer(x)  # (m, seq, fea, 2) to (m * seq, fea, 2)
        pose_out = self.pose_encode(out)  # (m * seq, fea,  2) to (m * seq, pose_latent_dim (or *2 if kld=True) )

        pose_z, pose_mu, pose_logvar = self.pose_bottoleneck(pose_out)  # all outputs (m * seq, pose_latent_dim)
        pose_z_seq = self.unflatten_dim2_layer(pose_z)  # (m * seq, pose_latent_dim) to (m, seq, pose_latent_dim)
        pose_z_seq = self.transpose_layer(pose_z_seq)  # (m, seq, pose_latent_dim) to (m, pose_latent_dim, seq)
        out = self.motion_encode(
            pose_z_seq)  # Convert (m, pose_latent_dim, seq) to (m, motion_latent_dim (or *2 if kld=True) )
        motion_z, motion_mu, motion_logvar = self.motion_bottoleneck(out)  # all outputs (m, motion_latent_dim)
        out = self.motion_decode(motion_z)  # (m, motion_latent_dim) to  (m, pose_latent_dim, seq)
        out = self.transpose_layer(out)  # (m, pose_latent_dim, seq) to (m, seq, pose_latent_dim)
        out = self.flatten_dim2_layer(out)  # (m, seq, pose_latent_dim) to (m * seq, pose_latent_dim)
        out = self.pose_decode(out)  # (m * seq, pose_latent_dim) to (m * seq, fea, 2)


        recon_motion = self.unflatten_dim3_layer(out)  # (m * seq, fea, 2) to (m, seq, fea, 2)
        pred_labels = self.class_net(motion_z)  # (m, motion_latent_dim) to (m, n_classes)
        return recon_motion, pred_labels, (pose_z_seq, pose_mu, pose_logvar), (motion_z, motion_mu, motion_logvar)

    def pose_encode(self, x):
        out = self.pose_vae.encode(x)
        return out

    def pose_bottoleneck(self, x):
        if self.posenet_kld:
            z, mu, logvar = self.pose_vae.pose_reparams(x)
        else:
            z = x
            mu = z.clone()  # dummy assignment
            logvar = mu  # dummy assignment

        return z, mu, logvar

    def pose_decode(self, x):
        out = self.pose_vae.decode(x)
        return out

    def motion_encode(self, x):
        out = self.motion_vae.encode(x)
        return out

    def motion_bottoleneck(self, x):
        if self.motionnet_kld:
            z, mu, logvar = self.motion_vae.motion_reparams(x)
        else:
            z = x.clone()
            mu, logvar = z, z  # dummy assignment
        return z, mu, logvar

    def motion_decode(self, x):
        out = self.motion_vae.decode(x)
        return out


class GraphPoseVAE(nn.Module):

    def __init__(self, node_dim, node_fea_dim, latent_dim, kld, dropout_p, device=None):
        """
        encode() converts (m * seq, node_dim, node_fea_dim) to (m * seq, latent_dim)
        decode() converts (m * seq, latent_dim) to (m * seq, node_dim, node_fea_dim)
        """

        # Model setting
        super(GraphPoseVAE, self).__init__()
        self.kld = kld
        self.node_dim, self.node_fea_dim, self.latent_dim = node_dim, node_fea_dim, latent_dim
        self.connecting_dim = self.latent_dim * 2 if self.kld else self.latent_dim
        self.encode_units = [4, 4, 4, 4]
        self.decode_units = [4, 4, 4, 4]
        self.device = torch.device('cuda:0') if device is None else device
        self.edge_index = construct_edge_indices()

        # Encoder

        self.first_layer = GCNConv(self.node_fea_dim, self.encode_units[0])

        self.en_graph1 = GCNConv(self.encode_units[0], self.encode_units[1])
        self.en_blk1 = nn.Sequential(*activation_block(output_channels=self.encode_units[1],
                                                       dropout_p=dropout_p))

        self.en_graph2 = GCNConv(self.encode_units[1], self.encode_units[2])
        self.en_blk2 = nn.Sequential(*activation_block(output_channels=self.encode_units[2],
                                                       dropout_p=dropout_p))

        self.en_graph3 = GCNConv(self.encode_units[2], self.encode_units[3])
        self.en_blk3 = nn.Sequential(*activation_block(output_channels=self.encode_units[3],
                                                       dropout_p=dropout_p))
        self.unflatten_firstdim_layer = UnFlatten(b=self.node_dim, dim=2)
        self.flatten_lastdim_later = FlattenLastDim(dim=2)

        self.en2latents = nn.Linear(self.node_dim * self.encode_units[3], self.connecting_dim)

        self.pose_reparams = Reparameterize(self.device, self.latent_dim)

        # Decode
        self.latents2de = nn.Linear(self.latent_dim, self.node_dim * self.decode_units[0])

        self.unflatten_lastdim_layer = UnflattenLastDim(self.decode_units[0])

        self.de_graph1 = GCNConv(self.decode_units[0], self.decode_units[1])
        self.de_blk1 = nn.Sequential(*activation_block(output_channels=self.decode_units[1],
                                                       dropout_p=dropout_p))

        self.de_graph2 = GCNConv(self.decode_units[1], self.decode_units[2])
        self.de_blk2 = nn.Sequential(*activation_block(output_channels=self.decode_units[2],
                                                       dropout_p=dropout_p))

        self.de_graph3 = GCNConv(self.decode_units[2], self.decode_units[3])
        self.de_blk3 = nn.Sequential(*activation_block(output_channels=self.decode_units[3],
                                                       dropout_p=dropout_p))

        self.final_layer = GCNConv(self.decode_units[3], self.node_fea_dim)

    def forward(self, x, edge_index):
        """

        Parameters
        ----------
        x : torch.tensor
        edge_index : torch.tensor
            with shape (2, num_edges)

        Returns
        -------
        out : torch.tensor
        mu : torch.tensor
        logvar : torch.tensor
        z : torch.tensor
        """
        # Encoder
        out = self.encode(x, edge_index)

        if self.kld:
            # Sampling from latents and concatenate with labels
            z, mu, logvar = self.pose_reparams(out)
        else:
            z = out
            mu = z.clone()  # dummy assignment
            logvar = z  # dummy assignment
        out = self.decode(z, edge_index)
        return out, mu, logvar, z

    def encode(self, x):
        """
        encode() converts (m * seq, node_dim, node_fea_dim) to (m * seq, latent_dim)
        """
        # x ~ (m * seq, node_dim, node_fea_dim)
        out, batch_edge_index = construct_batch_edge_index(x, self.edge_index)  # (m * seq * node_dim, node_fea_dim)

        out = self.first_layer(out, batch_edge_index)

        out = self.en_graph1(out, batch_edge_index)
        out = self.en_blk1(out)

        out = self.en_graph2(out, batch_edge_index)
        out = self.en_blk2(out)

        out = self.en_graph3(out, batch_edge_index)
        out = self.en_blk3(out)  # (m * seq * node_dim, last_output_chs)

        out = self.unflatten_firstdim_layer(
            out)  # (m * seq * node_dim, last_output_chs) to (m * seq, node_dim, last_output_chs)
        out = self.flatten_lastdim_later(
            out)  # (m * seq, node_dim, last_output_chs) to (m * seq, node_dim * last_output_chs)

        out = self.en2latents(out)  # (m * seq, node_dim * last_output_chs) to (m * seq, latent_dim)
        return out

    def decode(self, z):
        """
        decode() converts (m * seq, latent_dim) to (m * seq, node_dim, node_fea_dim)
        """
        out = self.latents2de(z)  # (m * seq, latent_dim) to (m * seq, node_dim * first_decode_chs)

        out = self.unflatten_lastdim_layer(
            out)  # (m * seq, node_dim * first_decode_chs) to (m * seq, node_dim, first_decode_chs)

        out, batch_edge_index = construct_batch_edge_index(out,
                                                           self.edge_index)  # (m * seq, node_dim, first_decode_chs) to (m * seq * node_dim, first_decode_chs)

        out = self.de_graph1(out, batch_edge_index)
        out = self.de_blk1(out)

        out = self.de_graph2(out, batch_edge_index)
        out = self.de_blk2(out)

        out = self.de_graph3(out, batch_edge_index)
        out = self.de_blk3(out)

        out = self.final_layer(out, batch_edge_index)
        out = self.unflatten_firstdim_layer(out)  # (m * seq * node_dim, node_fea_dim) to (m * seq, node_dim, node_fea_dim)
        return out
