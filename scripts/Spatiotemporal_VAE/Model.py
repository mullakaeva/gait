import torch
import torch.nn as nn
import logging


def pose_block(input_channels,
               output_channels,
               dropout_p=0.25):
    LN_layer = nn.Linear(input_channels, output_channels)
    bn_layer = nn.BatchNorm1d(output_channels)
    relu_layer = nn.ReLU()
    droput_layer = nn.Dropout(dropout_p)
    block_list = [
        LN_layer,
        bn_layer,
        relu_layer,
        droput_layer
    ]
    return block_list


def motion_encoding_block(input_channels,
                          output_channels,
                          kernel_size,
                          stride,
                          dropout_p=0.25):
    conv_layer = nn.Conv1d(input_channels, output_channels, kernel_size=kernel_size, stride=stride)
    bn_1_layer = nn.BatchNorm1d(output_channels)
    selu_1_layer = nn.SELU()
    droput_1_layer = nn.Dropout(dropout_p)
    fc_layer = nn.Conv1d(output_channels, output_channels, kernel_size=1)
    bn_2_layer = nn.BatchNorm1d(output_channels)
    selu_2_layer = nn.SELU()
    droput_2_layer = nn.Dropout(dropout_p)
    block_list = [
        conv_layer,
        bn_1_layer,
        selu_1_layer,
        droput_1_layer,
        fc_layer,
        bn_2_layer,
        selu_2_layer,
        droput_2_layer
    ]
    return block_list


def motion_decoding_block(input_channels, output_channels, kernel_size, stride):
    conv_layer = nn.ConvTranspose1d(input_channels, output_channels,
                                    kernel_size=kernel_size,
                                    stride=stride)
    bn_1_layer = nn.BatchNorm1d(output_channels)
    selu_1_layer = nn.SELU()
    fc_layer = nn.ConvTranspose1d(output_channels, output_channels, kernel_size=1)
    bn_2_layer = nn.BatchNorm1d(output_channels)
    selu_2_layer = nn.SELU()
    block_list = [
        conv_layer,
        bn_1_layer,
        selu_1_layer,
        fc_layer,
        bn_2_layer,
        selu_2_layer
    ]
    return block_list


class Reparameterize(nn.Module):
    def __init__(self, device, latent_dim):
        super(Reparameterize, self).__init__()
        self.device = device
        self.latent_dim = latent_dim

    def forward(self, h):
        # Reparameterize
        mu, logvar = h[:, 0:self.latent_dims], h[:, self.latent_dims:]
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z, mu, logvar



class Flatten(nn.Module):
    """
    Convert tensor from shape (a, b, c) to (a * b, c)
    """
    def forward(self, x):
        return x.view(x.size(0) * x.size(1), x.size(2))


class UnFlatten(nn.Module):
    """
    Convert tensor from shape (a * b, c) to (a, b, c)
    """
    def __init__(self, b):
        super(UnFlatten, self).__init__()
        self.b = b

    def forward(self, x):
        return x.view(x.size(0)/self.b, self.b, x.size(1))


class Transpose(nn.Module):
    """
    Convert tensor from shape (a, b, c) to (a, c, b)
    """
    def forward(self, x):
        return x.permute(0, 2, 1)



class SpatioTemporalVAE(nn.Module):
    def __init__(self,
                 network_params,
                 fea_dim=50,
                 seq_dim=128):
        """

        Parameters
        ----------
        network_params : dict


        """
        # # Loading parameters
        # Data dimension
        self.fea_dim = fea_dim
        self.seq_dim = seq_dim
        # Autoencoder for single pose
        self.posenet_latent_dim = network_params["posenet"]["latent_dim"]
        self.posenet_dropout_p = network_params["posenet"]["dropout_p"]
        self.posenet_kld = network_params["posenet"]["kld"]  # bool
        # Autoencoder for a sequence of poses (motion)
        self.motionnet_latent_dim = network_params["motionnet"]["latent_dim"]
        self.motionnet_hidden_dim = network_params["motionnet"]["hidden_dim"]
        self.motionnet_dropout_p = network_params["motionnet"]["dropout_p"]
        self.motionnet_kld = network_params["motionnet"]["kld"]  # bool
        # Others
        super(SpatioTemporalVAE, self).__init__()
        self.device = torch.device('cuda:0')

        # # Initializing network architecture
        self.transpose_layer = Transpose()
        self.flatten_layer = Flatten()
        self.unflatten_layer = UnFlatten(self.seq_dim)
        self.pose_reparams = Reparameterize(device=self.device, latent_dim=self.posenet_latent_dim)
        self.motion_reparams = Reparameterize(device=self.device, latent_dim=self.motionnet_latent_dim)
        self.pose_vae = PoseVAE(input_dims=self.fea_dim,
                                latent_dims=self.posenet_latent_dim,
                                kld=self.posenet_kld,
                                dropout_p=self.posenet_kld)


    def forward(self, x):
        out = self.transpose_flatten(x)  # Convert (m, fea, seq) to (m * seq, fea)
        pose_out = self.pose_encode(out)  # Convert (m * seq, fea) to (m * seq, pose_latent_dim)
        pose_z, pose_mu, pose_logvar = self.pose_bottoleneck(pose_out)  # all are (m * seq, pose_latent_dim)

        out = self.motion_encode(out)
        out = self.motion_decode(out)
        out = self.pose_decode(out)
        return out

    def pose_encode(self, x):
        out = self.pose_vae.encode(x)
        return out

    def pose_bottoleneck(self, x):
        if self.posenet_kld:
            z, mu, logvar = self.pose_reparams(x)
        else:
            z = x
            mu = z.clone()  # dummy assignment
            logvar = mu # dummy assignment

        return z, mu, logvar

    def pose_decode(self):
        return None

    def motion_encode(self):
        return None

    def motion_bottoleneck(self, x):
        if self.motionnet_kld:
            z, mu, logvar = self.motion_reparams(x)
        else:
            z = x.clone()
            mu, logvar = z, z # dummy assignment
        return z, mu, logvar

    def motion_decode(self):
        return None

    def transpose_flatten(self, x):
        out = self.transpose_layer(x)  # Convert (m, fea, seq) to (m, seq, fea)
        out = self.flatten_layer(out)  # Convert (m, seq, fea) to (m * seq, fea)
        return out

    def unflatten_transpose(self, x):
        out = self.unflatten_layer(x)  # Convert (m * seq, fea) to (m, seq, fea)
        out = self.transpose_layer(out)  # Convert (m, seq, fea) to (m, fea, seq)
        return out



class PoseVAE(nn.Module):

    def __init__(self, input_dims, latent_dims, kld, dropout_p):
        """
        PoseVAE takes in data with shape (m, input_dims), and reconstructs it with bottleneck layer (m, latent_dims),
        where m is number of samples.

        Parameters
        ----------
        input_dims : int
        latent_dims : int
            A value smaller than the dimension of the last layer (Default=32) before bottleneck layer.
        kld : bool
            Stochastic sampling if True, deterministic if False
        dropout_p : int
        """

        # Model setting
        super(PoseVAE, self)
        self.kld = kld
        self.input_dims, self.latent_dims = input_dims, latent_dims
        self.encode_units = [512, 128, 64, 32]
        self.decode_units = [32, 64, 128, 512]

        # Encoder
        self.first_layer = nn.Linear(self.input_dims, self.encode_units[0])

        self.en_blk1 = nn.Sequential(*pose_block(input_channels=self.encode_units[0],
                                                 output_channels=self.encode_units[1],
                                                 dropout_p=dropout_p))
        self.en_blk2 = nn.Sequential(*pose_block(input_channels=self.encode_units[1],
                                                 output_channels=self.encode_units[2],
                                                 dropout_p=dropout_p))
        self.en_blk3 = nn.Sequential(*pose_block(input_channels=self.encode_units[2],
                                                 output_channels=self.encode_units[3],
                                                 dropout_p=dropout_p))
        if self.kld:
            self.en2latents = nn.Linear(self.encode_units[3], self.latent_dims)
        else:
            self.en2latents = nn.Linear(self.encode_units[3], self.latent_dims * 2)
        self.pose_reparams = Reparameterize(self.device, self.latent_dims)
        # Decode
        self.latents2de = nn.Linear(self.latent_dims, self.decode_units[0])

        self.de_blk1 = nn.Sequential(*pose_block(input_channels=self.decode_units[0],
                                                 output_channels=self.decode_units[1],
                                                 dropout_p=dropout_p))
        self.de_blk2 = nn.Sequential(*pose_block(input_channels=self.decode_units[1],
                                                 output_channels=self.decode_units[2],
                                                 dropout_p=dropout_p))
        self.de_blk3 = nn.Sequential(*pose_block(input_channels=self.decode_units[2],
                                                 output_channels=self.decode_units[3],
                                                 dropout_p=dropout_p))
        self.final_layer = nn.Linear(self.decode_units[3], self.input_dims)

    def forward(self, x):
        """

        Parameters
        ----------
        x : pytorch.tensor

        Returns
        -------
        out : pytorch.tensor
        mu : pytorch.tensor
        logvar : pytorch.tensor
        z : pytorch.tensor
        """
        # Encoder
        out = self.encode(x)

        if self.kld:
            # Sampling from latents and concatenate with labels
            z, mu, logvar = self.pose_reparams(out)
            self.logging.debug("z's Shape: %s" % (str(z.shape)))
            # Decoder
            out = self.decode(z)
        else:
            z = out.clone()
            mu, logvar = z, z  # dummy assignment
            out = self.decode(out)
        return out, mu, logvar, z

    def encode(self, x):
        logging.debug("Input's Shape: %s"%(str(x.shape)))

        out = self.first_layer(x)
        logging.debug("Encode's Shape: %s" % (str(out.shape)))

        out = self.en_blk1(out) + out[:, 0:int(self.encode_units[1])]
        logging.debug("Encode's Shape: %s" % (str(out.shape)))

        out = self.en_blk2(out) + out[:, 0:int(self.encode_units[2])]
        logging.debug("Encode's Shape: %s" % (str(out.shape)))

        out = self.en_blk3(out) + out[:, 0:int(self.encode_units[3])]
        logging.debug("Encode's Shape: %s" % (str(out.shape)))

        out = self.en2latents(out)
        logging.debug("Encode's Shape: %s" % (str(out.shape)))
        return out

    def decode(self, z):
        logging.debug("z's Shape: %s" % (str(z.shape)))

        out = self.latents2de(z)
        logging.debug("Decode's Shape: %s" % (str(out.shape)))

        out = self.de_blk1(out)
        logging.debug("Decode's Shape: %s" % (str(out.shape)))

        out = self.de_blk2(out)
        logging.debug("Decode's Shape: %s" % (str(out.shape)))

        out = self.de_blk3(out)
        logging.debug("Decode's Shape: %s" % (str(out.shape)))

        out = self.final_layer(out)
        logging.debug("Decode's Shape: %s" % (str(out.shape)))
        return out
