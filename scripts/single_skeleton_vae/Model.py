import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging

class LshapeCounter:
    def __init__(self, L_in):
        self.L = L_in

    def updateL(self, kernel_size, dilation, padding=0, stride=1, decode=False):
        if decode == False:
            self.L = self.calc_conv1d_Lout(self.L, kernel_size, dilation, padding, stride)
        else:
            self.L = self.calc_deconv1d_Lout(self.L, kernel_size, dilation, padding, stride)
        return self.L
    @staticmethod
    def calc_conv1d_Lout(Lin, kernel_size, dilation, padding=0, stride=1):
        Lout = ((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1
        return Lout
    @staticmethod
    def calc_deconv1d_Lout(Lin, kernel_size, dilation, padding=0, stride=1, output_padding=0):
        Lout = (Lin-1) * stride - 2 * padding + dilation * (kernel_size-1) + output_padding + 1
        return Lout


def encoding_block(input_channels, output_channels, kernel_size, dilation):
    conv_layer = nn.Conv1d(input_channels, output_channels, kernel_size=kernel_size, dilation=dilation)
    bn_1_layer = nn.BatchNorm1d(output_channels)
    selu_1_layer = nn.SELU()
    fc_layer = nn.Conv1d(output_channels, output_channels, kernel_size=1, dilation=1)
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


def decoding_block(input_channels, output_channels, kernel_size, dilation):
    conv_layer = nn.ConvTranspose1d(input_channels, output_channels, kernel_size=kernel_size, dilation=dilation)
    bn_1_layer = nn.BatchNorm1d(output_channels)
    selu_1_layer = nn.SELU()
    fc_layer = nn.ConvTranspose1d(output_channels, output_channels, kernel_size=1, dilation=1)
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


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, channel_dims):
        super(UnFlatten, self).__init__()
        self.channel_dims = channel_dims

    def forward(self, x):
        return x.view(x.size(0), self.channel_dims, 1)


class VAE(nn.Module):
    def __init__(self, n_channels=1, L=50, hidden_channels=128, latent_dims=2, label_dims=0):
        """
        Variational Autoencoder (cVAE)
        In Single Skeleton Autoencoding. we want a VAE function f(x) that follows the shapes:

            x = (N, C, L), where N = batch_size, C = 1 (dummy channel), L = num_features, specifically,
            L = 25*2 = 50 (x, y openpose keypoints flattened)

            f(x) = (N, C, L), same shape as input x

        Parameters
        ----------
        n_channels : int
            Number of channels (C). For single skeleton auto-encoding, n_channels = 1
        L : int
            The length of input sequence. For single skeleton auto-encoding, L = 25 * 2 = 50
        label_dims: int
            If 0, it is a normal VAE. If a number, the model becomes conditional VAE, you will need to enter labels
            as argument of model.forward()
        """
        super(VAE, self).__init__()
        self.device = torch.device('cuda:0')
        self.logging = logging
        self.logging.basicConfig(level=50)
        self.n_channels, self.L, self.latent_dims, self.label_dims = n_channels, L, latent_dims, label_dims
        self.L_encode_counter = LshapeCounter(L)
        self.encoding_kernels = [6, 10, 10, 14, 5]
        self.encoding_dilations = [1, 1, 1, 1, 1]
        self.decoding_kernels = [6, 10, 10, 14, 14]
        self.decoding_dilations = [1, 1, 1, 1, 1]
        self.Ls_encode = [self.L_encode_counter.updateL(x, y) for x, y in
                          zip(self.encoding_kernels, self.encoding_dilations)]

        # Encoder
        self.first_layer = nn.Conv1d(self.n_channels,
                                     hidden_channels,
                                     kernel_size=self.encoding_kernels[0],
                                     dilation=self.encoding_dilations[0])
        self.en_blk1 = nn.Sequential(*encoding_block(hidden_channels,
                                                     hidden_channels,
                                                     kernel_size=self.encoding_kernels[1],
                                                     dilation=self.encoding_dilations[1]))
        self.en_blk2 = nn.Sequential(*encoding_block(hidden_channels,
                                                     hidden_channels,
                                                     kernel_size=self.encoding_kernels[2],
                                                     dilation=self.encoding_dilations[2]))
        self.en_blk3 = nn.Sequential(*encoding_block(hidden_channels,
                                                     hidden_channels,
                                                     kernel_size=self.encoding_kernels[3],
                                                     dilation=self.encoding_dilations[3]))
        self.en_blk4 = nn.Sequential(*encoding_block(hidden_channels,
                                                     hidden_channels,
                                                     kernel_size=self.encoding_kernels[4],
                                                     dilation=self.encoding_dilations[4]))
        self.en2latents = nn.Sequential(
            Flatten(),
            nn.Linear(hidden_channels*int(self.Ls_encode[4]), 2*self.latent_dims)
        )
        self.latents2de = nn.Sequential(
            UnFlatten(self.latent_dims + self.label_dims),
            nn.ConvTranspose1d(self.latent_dims + self.label_dims,
                               hidden_channels,
                               kernel_size=self.decoding_kernels[0],
                               dilation=self.decoding_dilations[0],
                               output_padding=0)
        )

        self.de_blk1 = nn.Sequential(*decoding_block(hidden_channels,
                                                     hidden_channels,
                                                     kernel_size=self.decoding_kernels[1],
                                                     dilation=self.decoding_dilations[1]))
        self.de_blk2 = nn.Sequential(*decoding_block(hidden_channels,
                                                     hidden_channels,
                                                     kernel_size=self.decoding_kernels[2],
                                                     dilation=self.decoding_dilations[2]))
        self.de_blk3 = nn.Sequential(*decoding_block(hidden_channels,
                                                     hidden_channels,
                                                     kernel_size=self.decoding_kernels[3],
                                                     dilation=self.decoding_dilations[3]))
        self.de_blk4 = nn.Sequential(*decoding_block(hidden_channels,
                                                     hidden_channels,
                                                     kernel_size=self.decoding_kernels[4],
                                                     dilation=self.decoding_dilations[4]))
        self.final_layer = nn.Sequential(
            nn.Conv1d(hidden_channels, self.n_channels, kernel_size=1)
        )
    def forward(self, x, labels=None):
        ""
        # Encoder
        out = self.encode(x)

        # Sampling from latents and concatenate with labels
        z, mu, logvar  = self.bottleneck(out)
        self.logging.debug("z's Shape: %s" % (str(z.shape)))

        if labels is None:
            z_c = z
        else:
            assert labels.shape[1] == self.label_dims
            z_c = torch.cat((z, labels), dim=-1)
        # Decoder
        out = self.decode(z_c)

        return out, mu, logvar, z

    def encode(self, x):
        self.logging.debug("Input's Shape: %s"%(str(x.shape)))
        out = self.first_layer(x)
        self.logging.debug("Encode's Shape: %s" % (str(out.shape)))
        out = self.en_blk1(out) + out[:, :, 0:int(self.Ls_encode[1])]
        self.logging.debug("Encode's Shape: %s" % (str(out.shape)))
        out = self.en_blk2(out) + out[:, :, 0:int(self.Ls_encode[2])]
        self.logging.debug("Encode's Shape: %s" % (str(out.shape)))
        out = self.en_blk3(out) + out[:, :, 0:int(self.Ls_encode[3])]
        self.logging.debug("Encode's Shape: %s" % (str(out.shape)))
        out = self.en_blk4(out) + out[:, :, 0:int(self.Ls_encode[4])]
        self.logging.debug("Encode's Shape: %s" % (str(out.shape)))
        out = self.en2latents(out)
        self.logging.debug("Encode's Shape: %s" % (str(out.shape)))
        return out

    def decode(self, z_c):
        self.logging.debug("z_c's Shape: %s" % (str(z_c.shape)))
        out = self.latents2de(z_c)
        self.logging.debug("Decode's Shape: %s" % (str(out.shape)))
        out = self.de_blk1(out)
        self.logging.debug("Decode's Shape: %s" % (str(out.shape)))
        out = self.de_blk2(out)
        self.logging.debug("Decode's Shape: %s" % (str(out.shape)))
        out = self.de_blk3(out)
        self.logging.debug("Decode's Shape: %s" % (str(out.shape)))
        out = self.de_blk4(out)
        self.logging.debug("Decode's Shape: %s" % (str(out.shape)))
        out = self.final_layer(out)
        self.logging.debug("Decode's Shape: %s" % (str(out.shape)))
        return out

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = h[:, 0:self.latent_dims], h[:, self.latent_dims:]
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

def total_loss(x, pred, mu, logvar):
    recon_loss = 0.5 * torch.mean((x-pred)**2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + KLD
    return loss

if __name__ == "__main__":
    device = torch.device('cuda:0')
    x = torch.randn(512, 1, 50).to(device)
    y = torch.randn(512, 2).to(device)
    model = VAE().to(device)
    params = model.parameters()
    optimizer = optim.Adam(params, lr=0.001)
    for i in range(10):
        optimizer.zero_grad()
        out, mu, logvar, z = model.forward(x)
        loss = total_loss(x, out, mu, logvar)
        print(loss)
        loss.backward()
        optimizer.step()

