import os
import logging
import time
import numpy as np
import numpy.random as npr
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pdb



from .torchdiffeq import odeint_adjoint as odeint


class LatentODEfunc(nn.Module):

    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out


class RecognitionRNN(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=25, nbatch=1):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.elu = nn.ELU(inplace=True)
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2h = nn.Linear(nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        h = self.elu(self.i2h(combined))
        h = self.h2h(h)
        h = torch.tanh(h)
        out = self.h2o(h)
        return out, h

    def initHidden(self):
        return torch.zeros(self.nbatch, self.nhidden)


class Decoder(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def log_normal_pdf(x, mean, logvar):

    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl



class GaitLatentODEModel:
    def __init__(self, data_gen, latent_dim, n_hidden, rnn_n_hidden, obs_dim, device_num, lr,
                 save_chkpt_dir=None, noise_std=0.3):
        self.latent_dim, self.n_hidden, self.rnn_n_hidden, self.obs_dim = latent_dim, n_hidden, rnn_n_hidden, obs_dim
        self.noise_std, self.lr = noise_std, lr
        self.device = torch.device('cuda:' + str(device_num)
                          if torch.cuda.is_available() else 'cpu')

        self.data_gen, self.n_samples = data_gen, data_gen.m
        self.func, self.rec, self.dec, self.parmas, self.optimizer, self.loss_meter, self.cv_loss_meter = self._build_model()
        self.save_chkpt_dir = save_chkpt_dir

    def _build_model(self):
        func = LatentODEfunc(self.latent_dim, self.n_hidden).to(self.device)
        rec = RecognitionRNN(self.latent_dim, self.obs_dim, self.rnn_n_hidden, self.n_samples).to(self.device)
        dec = Decoder(self.latent_dim, self.obs_dim, self.n_hidden).to(self.device)
        params = (list(func.parameters()) + list(dec.parameters()) + list(rec.parameters()))
        optimizer = optim.Adam(params, lr=self.lr)
        loss_meter = RunningAverageMeter()
        cv_loss_meter = RunningAverageMeter()
        return func, rec, dec, params, optimizer, loss_meter, cv_loss_meter

    def train(self, n_epochs):
        try:
            for epoch in range(1, n_epochs + 1):
                for idx, (sampled_data, times) in enumerate(self.data_gen.iterator()):
                    print("\rEpoch %d/%d at iter %d/%d | Loss: %0.4f | CV Loss: %0.4f"%(epoch, n_epochs, idx,
                                                                       self.data_gen.num_rows/self.data_gen.m,
                                                                       -self.loss_meter.avg, -self.cv_loss_meter.avg), flush=True, end="")
                    batch_xt, batch_xt_test = sampled_data
                    batch_xt = torch.from_numpy(batch_xt).float().to(self.device)
                    batch_xt_test = torch.from_numpy(batch_xt_test).float().to(self.device)
                    times = torch.from_numpy(times).float().to(self.device)
                    self.optimizer.zero_grad()
                    _, loss = self._predict_with_grad(batch_xt, times) # Forward pass
                    _, cv_loss = self.predict(batch_xt_test, times)
                    loss.backward()
                    self.optimizer.step()
                    self.loss_meter.update(loss.item())
                    self.cv_loss_meter.update(cv_loss.item())

                print()

        except KeyboardInterrupt:
            self._save_progress()
        print('Training complete after {} epochs.'.format(epoch))
        self._save_progress()

    def predict(self, batch_xt, times):
        with torch.no_grad():
            pred_z, loss = self._predict_with_grad(batch_xt, times)
        return pred_z, loss

    def sample_from_latent(self, z0_np, time_steps_np):
        """

        Parameters
        ----------
        z0_np : Numpy array with shape (num_samples, num_latent_dim)
        time_steps_np : Numpy array with shape (num_time_steps, )

        Returns
        -------
        xs : Numpy array with shape (num_samples, num_time_steps, num_obs_dim)
        """
        z0_tensor = torch.from_numpy(z0_np).float().to(self.device)
        time_steps_tensor = torch.from_numpy(time_steps_np).float().to(self.device)
        xs_tensor = self._solve_from_latent(z0_tensor, time_steps_tensor)
        return xs_tensor.cpu().numpy()

    def _predict_with_grad(self, batch_xt, times):
        """

        Parameters
        ----------
        batch_xt : tensor
            Data that you want to model. Shape = (num_samples, times, dim)
        times : tensor
            Time points of the data. Shape = (times, )

        Returns
        -------
        loss : tensor
            Loss (Reconstruction loss + ELBO)
        """
        h = self.rec.initHidden().to(self.device)
        for t in reversed(range(times.shape[0])):
            obs = batch_xt[:, t, :]
            out, h = self.rec.forward(obs, h)
        qz0_mean, qz0_logvar = out[:, :self.latent_dim], out[:, self.latent_dim:]
        epsilon = torch.randn(qz0_mean.size()).to(self.device)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

        # forward in time and solve ode for reconstructions
        pred_z = odeint(self.func, z0, times).permute(1, 0, 2)
        pred_x = self.dec(pred_z)

        # compute loss
        noise_std_ = torch.zeros(pred_x.size()).to(self.device) + self.noise_std
        noise_logvar = 2. * torch.log(noise_std_).to(self.device)
        logpx = log_normal_pdf(
            batch_xt, pred_x, noise_logvar).sum(-1).sum(-1)
        pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(self.device)
        analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                                pz0_mean, pz0_logvar).sum(-1)
        loss = torch.mean(-logpx + analytic_kl, dim=0)
        return pred_x, loss

    def _solve_from_latent(self, z0, time_steps):
        with torch.no_grad():
            zs = odeint(self.func, z0, time_steps).permute(1, 0, 2)
            xs = self.dec(zs)
        return xs

    def load_saved_progress(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        self.func.load_state_dict(checkpoint['func_state_dict'])
        self.rec.load_state_dict(checkpoint['rec_state_dict'])
        self.dec.load_state_dict(checkpoint['dec_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss_meter = checkpoint['loss_meter']
        self.cv_loss_meter = checkpoint['cv_loss_meter']
        print('Loaded ckpt from {}'.format(ckpt_path))
        return None

    def _save_progress(self):
        if self.save_chkpt_dir is not None:
            if os.path.isdir(self.save_chkpt_dir) is not True:
                os.makedirs(self.save_chkpt_dir, exist_ok=True)
            ckpt_path = os.path.join(self.save_chkpt_dir, 'ckpt.pth')
            torch.save({
                'func_state_dict': self.func.state_dict(),
                'rec_state_dict': self.rec.state_dict(),
                'dec_state_dict': self.dec.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss_meter': self.loss_meter,
                'cv_loss_meter': self.cv_loss_meter
            }, ckpt_path)
            print('Stored ckpt at {}'.format(ckpt_path))


if __name__ == '__main__':
    pass
