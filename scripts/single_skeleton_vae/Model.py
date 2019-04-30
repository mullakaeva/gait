import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging



def linear_block(input_channels, output_channels, dropout_p=0.25):

    LN_layer = nn.Linear(input_channels, output_channels)
    bn_layer = nn.BatchNorm1d(output_channels)
    relu_layer = nn.ReLU()

    block_list = [
        LN_layer,
        bn_layer,
        relu_layer
    ]

    return block_list


class VAE(nn.Module):
    def __init__(self, input_dims=50, latent_dims=2):

        super(VAE, self).__init__()
        self.device = torch.device('cuda:0')
        self.logging = logging
        self.logging.basicConfig(level=50)
        self.input_dims, self.latent_dims = input_dims, latent_dims

        self.encode_units = [512, 128, 32, 8]
        self.decode_units = [8, 32, 128, 512]


        # Encoder
        self.first_layer = nn.Linear(self.input_dims, self.encode_units[0])

        self.en_blk1 = nn.Sequential(*linear_block(input_channels=self.encode_units[0],
                                                   output_channels=self.encode_units[1]))
        self.en_blk2 = nn.Sequential(*linear_block(input_channels=self.encode_units[1],
                                                   output_channels=self.encode_units[2]))
        self.en_blk3 = nn.Sequential(*linear_block(input_channels=self.encode_units[2],
                                                   output_channels=self.encode_units[3]))
        self.en2latents = nn.Linear(self.encode_units[3], self.latent_dims*2)

        # Decode
        self.latents2de = nn.Linear(self.latent_dims, self.decode_units[0])

        self.de_blk1 = nn.Sequential(*linear_block(input_channels=self.decode_units[0],
                                                   output_channels=self.decode_units[1]))
        self.de_blk2 = nn.Sequential(*linear_block(input_channels=self.decode_units[1],
                                                   output_channels=self.decode_units[2]))
        self.de_blk3 = nn.Sequential(*linear_block(input_channels=self.decode_units[2],
                                                   output_channels=self.decode_units[3]))
        self.final_layer = nn.Linear(self.decode_units[3], self.input_dims)

    def forward(self, x, labels=None):

        # Encoder
        out = self.encode(x)

        # Sampling from latents and concatenate with labels
        z, mu, logvar = self.bottleneck(out)
        self.logging.debug("z's Shape: %s" % (str(z.shape)))

        # Decoder
        out = self.decode(z)
        return out, mu, logvar, z

    def encode(self, x):
        self.logging.debug("Input's Shape: %s"%(str(x.shape)))
        out = self.first_layer(x)
        self.logging.debug("Encode's Shape: %s" % (str(out.shape)))
        # out = self.en_blk1(out)
        out = self.en_blk1(out) + out[:, 0:int(self.encode_units[1])]
        self.logging.debug("Encode's Shape: %s" % (str(out.shape)))
        # out = self.en_blk2(out)
        out = self.en_blk2(out) + out[:, 0:int(self.encode_units[2])]
        self.logging.debug("Encode's Shape: %s" % (str(out.shape)))
        # out = self.en_blk3(out)
        out = self.en_blk3(out) + out[:, 0:int(self.encode_units[3])]
        self.logging.debug("Encode's Shape: %s" % (str(out.shape)))
        out = self.en2latents(out)
        self.logging.debug("Encode's Shape: %s" % (str(out.shape)))
        return out

    def decode(self, z):
        self.logging.debug("z's Shape: %s" % (str(z.shape)))
        out = self.latents2de(z)
        self.logging.debug("Decode's Shape: %s" % (str(out.shape)))
        out = self.de_blk1(out)
        self.logging.debug("Decode's Shape: %s" % (str(out.shape)))
        out = self.de_blk2(out)
        self.logging.debug("Decode's Shape: %s" % (str(out.shape)))
        out = self.de_blk3(out)
        self.logging.debug("Decode's Shape: %s" % (str(out.shape)))
        out = self.final_layer(out)
        self.logging.debug("Decode's Shape: %s" % (str(out.shape)))
        return out

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(self.device)
        # esp = torch.zeros(*mu.size()).to(self.device)
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
    x = torch.randn(512, 50).to(device)
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

