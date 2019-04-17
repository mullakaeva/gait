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



from torchdiffeq import odeint_adjoint as odeint


def generate_spiral2d(nspiral=1000,
                      ntotal=500,
                      nsample=100,
                      start=0.,
                      stop=1,  # approximately equal to 6pi
                      noise_std=.1,
                      a=0.,
                      b=1.,
                      savefig=True):
    """Parametric formula for 2d spiral is `r = a + b * theta`.

    Args:
      nspiral: number of spirals, i.e. batch dimension
      ntotal: total number of datapoints per spiral
      nsample: number of sampled datapoints for model fitting per spiral
      start: spiral starting theta value
      stop: spiral ending theta value
      noise_std: observation noise standard deviation
      a, b: parameters of the Archimedean spiral
      savefig: plot the ground truth for sanity check

    Returns: 
      Tuple where first element is true trajectory of size (nspiral, ntotal, 2),
      second element is noisy observations of size (nspiral, nsample, 2),
      third element is timestamps of size (ntotal,),
      and fourth element is timestamps of size (nsample,)
    """

    # add 1 all timestamps to avoid division by 0
    orig_ts = np.linspace(start, stop, num=ntotal)
    samp_ts = orig_ts[:nsample]

    # generate clock-wise and counter clock-wise spirals in observation space
    # with two sets of time-invariant latent dynamics
    zs_cw = stop + 1. - orig_ts
    rs_cw = a + b * 50. / zs_cw
    xs, ys = rs_cw * np.cos(zs_cw) - 5., rs_cw * np.sin(zs_cw)
    cw_labels = np.zeros(xs.shape)
    orig_traj_cw = np.stack((xs, ys, cw_labels), axis=1)

    zs_cc = orig_ts
    rw_cc = a + b * zs_cc
    xs, ys = rw_cc * np.cos(zs_cc) + 5., rw_cc * np.sin(zs_cc)
    cc_labels = np.ones(xs.shape)
    orig_traj_cc = np.stack((xs, ys, cc_labels), axis=1)

    if savefig:
        plt.figure()
        plt.plot(orig_traj_cw[:, 0], orig_traj_cw[:, 1], label='clock')
        plt.plot(orig_traj_cc[:, 0], orig_traj_cc[:, 1], label='counter clock')
        plt.legend()
        plt.savefig('./ground_truth.png', dpi=500)
        print('Saved ground truth spiral at {}'.format('./ground_truth.png'))

    # sample starting timestamps
    orig_trajs = []
    samp_trajs = []
    for _ in range(nspiral):
        # don't sample t0 very near the start or the end
        t0_idx = npr.multinomial(
            1, [1. / (ntotal - 2. * nsample)] * (ntotal - int(2 * nsample)))
        t0_idx = np.argmax(t0_idx) + nsample

        cc = bool(npr.rand() > .5)  # uniformly select rotation
        orig_traj = orig_traj_cc if cc else orig_traj_cw
        orig_trajs.append(orig_traj)

        samp_traj = orig_traj[t0_idx:t0_idx + nsample, :].copy()
        samp_traj += npr.randn(*samp_traj.shape) * noise_std
        samp_trajs.append(samp_traj)

    # batching for sample trajectories is good for RNN; batching for original
    # trajectories only for ease of indexing
    orig_trajs = np.stack(orig_trajs, axis=0)
    samp_trajs = np.stack(samp_trajs, axis=0)

    return orig_trajs, samp_trajs, orig_ts, samp_ts


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
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        h = torch.tanh(self.i2h(combined))
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

def ExampleSpirals():
    nspiral = 1000
    ntotal = 500
    nsample = 100
    start = 0.
    stop = 6 * np.pi
    noise_std = .3
    a = 0.
    b = .3
    save_fig = True
    orig_trajs, samp_trajs, orig_ts, samp_ts = generate_spiral2d(
        nspiral=nspiral,
        ntotal=ntotal,
        nsample=nsample,
        start=start,
        stop=stop,
        noise_std=noise_std,
        a=a,
        b=b,
        savefig=save_fig
    )
    return orig_trajs, samp_trajs, orig_ts, samp_ts


class LatentODEModel:
    def __init__(self, xt, t, latent_dim, n_hidden, rnn_n_hidden, obs_dim, device_num, lr,
                 save_chkpt_dir=None, noise_std=0.3):
        self.latent_dim, self.n_hidden, self.rnn_n_hidden, self.obs_dim = latent_dim, n_hidden, rnn_n_hidden, obs_dim
        self.noise_std, self.lr = noise_std, lr
        self.device = torch.device('cuda:' + str(device_num)
                          if torch.cuda.is_available() else 'cpu')

        self.xt, self.t, self.n_samples = self._load_data(xt, t)
        self.func, self.rec, self.dec, self.parmas, self.optimizer, self.loss_meter = self._build_model()
        self.save_chkpt_dir = save_chkpt_dir

    def _load_data(self, xt, t):
        xt = torch.from_numpy(xt).float().to(self.device)
        t = torch.from_numpy(t).float().to(self.device)
        n_samples = xt.shape[0]
        return xt, t, n_samples

    def _build_model(self):
        func = LatentODEfunc(self.latent_dim, self.n_hidden).to(self.device)
        rec = RecognitionRNN(self.latent_dim, self.obs_dim, self.rnn_n_hidden, self.n_samples).to(self.device)
        dec = Decoder(self.latent_dim, self.obs_dim, self.n_hidden).to(self.device)
        params = (list(func.parameters()) + list(dec.parameters()) + list(rec.parameters()))
        optimizer = optim.Adam(params, lr=self.lr)
        loss_meter = RunningAverageMeter()
        return func, rec, dec, params, optimizer, loss_meter

    def train(self, n_iters):
        try:
            for itr in range(1, n_iters + 1):
                self.optimizer.zero_grad()
                # backward in time to infer q(z_0)
                h = self.rec.initHidden().to(self.device)
                import pdb
                pdb.set_trace()
                for t in reversed(range(self.xt.size(1))):
                    obs = self.xt[:, t, :]
                    out, h = self.rec.forward(obs, h)
                qz0_mean, qz0_logvar = out[:, :self.latent_dim], out[:, self.latent_dim:]
                epsilon = torch.randn(qz0_mean.size()).to(self.device)
                z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

                # forward in time and solve ode for reconstructions
                pred_z = odeint(self.func, z0, self.t).permute(1, 0, 2)
                pred_x = self.dec(pred_z)

                # compute loss
                noise_std_ = torch.zeros(pred_x.size()).to(self.device) + self.noise_std
                noise_logvar = 2. * torch.log(noise_std_).to(self.device)
                logpx = log_normal_pdf(
                    self.xt, pred_x, noise_logvar).sum(-1).sum(-1)
                pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(self.device)
                analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                                        pz0_mean, pz0_logvar).sum(-1)
                loss = torch.mean(-logpx + analytic_kl, dim=0)
                loss.backward()
                self.optimizer.step()
                self.loss_meter.update(loss.item())

                print('Iter: {}, running avg elbo: {:.4f}'.format(itr, -self.loss_meter.avg))

        except KeyboardInterrupt:
            self._save_progress()
        print('Training complete after {} iters.'.format(itr))
        self._save_progress()

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

    def predict(self):
        pass

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
        self.xt = checkpoint['samp_trajs']
        self.t = checkpoint['samp_ts']
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
                'samp_trajs': self.xt,
                'samp_ts': self.t,
            }, ckpt_path)
            print('Stored ckpt at {}'.format(ckpt_path))


if __name__ == '__main__':
    pass
