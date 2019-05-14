import torch
import torch.nn as nn
import logging

class LshapeCounter:
    def __init__(self, L_in):
        self.L = L_in

    def updateL(self, kernel_size, dilation=1, padding=0, stride=1, decode=False):
        if decode is False:
            self.L = self.calc_conv1d_Lout(self.L, kernel_size, dilation, padding, stride)
        else:
            self.L = self.calc_deconv1d_Lout(self.L, kernel_size, dilation, padding, stride)
        return self.L

    @staticmethod
    def calc_conv1d_Lout(Lin, kernel_size, dilation=1, padding=0, stride=1):
        Lout = ((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1
        return Lout

    @staticmethod
    def calc_deconv1d_Lout(Lin, kernel_size, dilation=1, padding=0, stride=1, output_padding=0):
        Lout = (Lin - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
        return Lout


def encoding_block(input_channels, output_channels, kernel_size, stride, dropout_p=0.25, dilation=1):
    conv_layer = nn.Conv1d(input_channels, output_channels, kernel_size=kernel_size, dilation=dilation, stride=stride)
    bn_1_layer = nn.BatchNorm1d(output_channels)
    selu_1_layer = nn.SELU()
    droput_1_layer = nn.Dropout(dropout_p)
    fc_layer = nn.Conv1d(output_channels, output_channels, kernel_size=1, dilation=1)
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


def decoding_block(input_channels, output_channels, kernel_size, stride, dilation=1):
    conv_layer = nn.ConvTranspose1d(input_channels, output_channels,
                                    kernel_size=kernel_size,
                                    dilation=dilation,
                                    stride=stride)
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


class TemporalVAE(nn.Module):
    def __init__(self, n_channels=50, L=128, hidden_channels=1024, latent_dims=8):
        """
        Temporal Variational Autoencoder (TemporalVAE)
        In Gait analysis. we want a VAE function f(x) that follows the shapes:

            x = (N, C, L), where N = batch_size, C = n_features + n_labels, L = sequence length (time), specifically,
            C = 25*2+8.

            f(x) = (N, C_2, L), where C_2 = 25*2, denoting the x, y-coordinates of the 25 keypoints.

        Parameters
        ----------
        n_channels : int
            Number of channels (C). In gait analysis, n_channels = 25 * 2 + 8, which is (n_features + n_labels).
        L : int
            The length of input sequence. In gait analysis, it is the sequence length.
        """
        super(TemporalVAE, self).__init__()
        self.device = torch.device('cuda:0')
        self.n_channels, self.L, self.latent_dims = n_channels, L, latent_dims
        self.L_encode_counter = LshapeCounter(L)
        self.encoding_kernels = [5, 5, 5, 5, 5]
        self.encoding_strides = [1, 2, 2, 2, 2]
        self.decoding_kernels = [5, 5, 5, 5, 8]
        self.decoding_strides = [1, 2, 2, 2, 2]
        self.Ls_encode = [self.L_encode_counter.updateL(kernel_size=x, stride=y) for x, y, in
                          zip(self.encoding_kernels, self.encoding_strides)]

        # Encoder
        self.first_layer = nn.Conv1d(self.n_channels,
                                     hidden_channels,
                                     kernel_size=self.encoding_kernels[0],
                                     stride=self.encoding_strides[0])
        self.en_blk1 = nn.Sequential(*encoding_block(hidden_channels,
                                                     hidden_channels,
                                                     kernel_size=self.encoding_kernels[1],
                                                     stride=self.encoding_strides[1]))
        self.en_blk2 = nn.Sequential(*encoding_block(hidden_channels,
                                                     hidden_channels,
                                                     kernel_size=self.encoding_kernels[2],
                                                     stride=self.encoding_strides[2]))
        self.en_blk3 = nn.Sequential(*encoding_block(hidden_channels,
                                                     hidden_channels,
                                                     kernel_size=self.encoding_kernels[3],
                                                     stride=self.encoding_strides[3]))
        self.en_blk4 = nn.Sequential(*encoding_block(hidden_channels,
                                                     hidden_channels,
                                                     kernel_size=self.encoding_kernels[4],
                                                     stride=self.encoding_strides[4]))
        self.en2latents = nn.Sequential(
            Flatten(),
            # nn.Linear(hidden_channels * int(self.Ls_encode[4]), 2 * self.latent_dims)
            nn.Linear(hidden_channels * int(self.Ls_encode[4]), self.latent_dims)
        )
        self.latents2de = nn.Sequential(
            UnFlatten(self.latent_dims),
            nn.ConvTranspose1d(self.latent_dims,
                               hidden_channels,
                               kernel_size=self.decoding_kernels[0],
                               stride=self.decoding_strides[0])
        )

        self.de_blk1 = nn.Sequential(*decoding_block(hidden_channels,
                                                     hidden_channels,
                                                     kernel_size=self.decoding_kernels[1],
                                                     stride=self.decoding_strides[1]))
        self.de_blk2 = nn.Sequential(*decoding_block(hidden_channels,
                                                     hidden_channels,
                                                     kernel_size=self.decoding_kernels[2],
                                                     stride=self.decoding_strides[2]))
        self.de_blk3 = nn.Sequential(*decoding_block(hidden_channels,
                                                     hidden_channels,
                                                     kernel_size=self.decoding_kernels[3],
                                                     stride=self.decoding_strides[3]))
        self.de_blk4 = nn.Sequential(*decoding_block(hidden_channels,
                                                     hidden_channels,
                                                     kernel_size=self.decoding_kernels[4],
                                                     stride=self.decoding_strides[4]))
        self.final_layer = nn.Sequential(
            nn.Conv1d(hidden_channels, self.n_channels, kernel_size=1)
        )

    def forward(self, x):
        # Encoder
        out = self.encode(x)
        z_latent = out.clone()
        # Sampling from latents and concatenate with labels
        z, mu, logvar = self.bottleneck(out)

        # Decoder
        # out = self.decode(z)
        out = self.decode(out)
        return out, mu, logvar, z_latent

    def encode(self, x):
        logging.debug("Ls_encode = %s" % self.Ls_encode)
        logging.debug("Input's Shape: %s"%(str(x.shape)))
        out = self.first_layer(x)
        logging.debug("Encode's Shape: %s" % (str(out.shape)))
        out = self.en_blk1(out) + out[:, :, 0:int(self.Ls_encode[1])]
        logging.debug("Encode's Shape: %s" % (str(out.shape)))
        out = self.en_blk2(out) + out[:, :, 0:int(self.Ls_encode[2])]
        logging.debug("Encode's Shape: %s" % (str(out.shape)))
        out = self.en_blk3(out) + out[:, :, 0:int(self.Ls_encode[3])]
        logging.debug("Encode's Shape: %s" % (str(out.shape)))
        out = self.en_blk4(out) + out[:, :, 0:int(self.Ls_encode[4])]
        logging.debug("Encode's Shape: %s" % (str(out.shape)))
        out = self.en2latents(out)
        logging.debug("Encode's Shape: %s" % (str(out.shape)))
        return out

    def decode(self, z_c):
        logging.debug("z_c's Shape: %s" % (str(z_c.shape)))
        out = self.latents2de(z_c)
        logging.debug("Decode's Shape: %s" % (str(out.shape)))
        out = self.de_blk1(out)
        logging.debug("Decode's Shape: %s" % (str(out.shape)))
        out = self.de_blk2(out)
        logging.debug("Decode's Shape: %s" % (str(out.shape)))
        out = self.de_blk3(out)
        logging.debug("Decode's Shape: %s" % (str(out.shape)))
        out = self.de_blk4(out)
        logging.debug("Decode's Shape: %s" % (str(out.shape)))
        out = self.final_layer(out)
        logging.debug("Decode's Shape: %s" % (str(out.shape)))
        return out

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = h, h
        # mu, logvar = h[:, 0:self.latent_dims], h[:, self.latent_dims:]
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


if __name__ == "__main__":
    # pass

    from torch import optim
    logging.basicConfig(level=logging.CRITICAL)
    device = torch.device('cuda:0')
    x = torch.randn(512, 50, 128).to(device)
    model = TemporalVAE(latent_dims=2).to(device)
    params = model.parameters()
    optimizer = optim.Adam(params, lr=0.001)
    for i in range(10):
        optimizer.zero_grad()
        out, mu, logvar, z = model(x)
        loss = torch.sum((x-out)**2)
        print(loss)
        loss.backward()
        optimizer.step()
