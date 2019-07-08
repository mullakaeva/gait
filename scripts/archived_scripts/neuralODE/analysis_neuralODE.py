import numpy as np
import matplotlib.pyplot as plt
import skvideo.io as skv
import os
from .gait_neuralODE.gait_latent_ode import GaitLatentODEModel


def gen_latent(dim, max_dim, start=0, end=2, num_samples=10):
    latent_vecs = np.zeros((num_samples, max_dim))
    latent_vecs[:, dim] = np.linspace(start, end, num_samples)
    return latent_vecs


def gait_neural_ode_init(data_gen, load_path=None):
    model = GaitLatentODEModel(data_gen=data_gen,
                               latent_dim=8,
                               n_hidden=150,
                               rnn_n_hidden=150,
                               obs_dim=75,
                               device_num=0,
                               lr=0.001,
                               save_chkpt_dir="neuralODE/gait_ODE_chkpt")
    if load_path is not None:
        model.load_saved_progress(load_path)
    return model


def gait_neural_ode_train(data_gen, n_epochs=50, load_path=None):
    model = gait_neural_ode_init(data_gen, load_path=load_path)
    model.train(n_epochs)


def gait_neural_ode_vis(load_path, vis_dir, data_gen=None):
    model = gait_neural_ode_init(data_gen=data_gen,
                                 load_path=load_path)
    max_dim = 8
    times = np.arange(128) / 25
    # Per dimension
    for dim in range(max_dim):
        latents_per_dim = gen_latent(dim=dim,
                                     max_dim=max_dim,
                                     start=-2,
                                     end=2,
                                     num_samples=4)

        pred_x = model.sample_from_latent(z0_np=latents_per_dim,
                                          time_steps_np=times)
        pred_x = pred_x.reshape(pred_x.shape[0], pred_x.shape[1], 25, 3)
        for sample_idx in range(pred_x.shape[0]):
            latents_val = latents_per_dim[sample_idx, dim]
            save_vid_path = os.path.join(vis_dir, "vid_dim-%d_latent-%0.2f.mp4" % (dim, latents_val))
            vwriter = skv.FFmpegWriter(save_vid_path)

            for t in range(times.shape[0]):
                print("\rNow writing dim-%d | latent-%0.4f | time-%0.4f/128"% (dim, latents_val, t), flush=True, end="")
                fig, ax = plt.subplots()
                ax.scatter(pred_x[sample_idx, t, :, 0], pred_x[sample_idx, t, :, 1])

                fig.suptitle("Class = %0.4f | Time = %0.4fs" % (np.mean(pred_x[sample_idx, t, :, 2])*8, t))
                ax.set_xlim(-0.5, 1)
                ax.set_ylim(1, -0.5)
                fig.tight_layout()
                fig.canvas.draw()
                # Now we can save it to a numpy array.
                data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                vwriter.writeFrame(data)
                plt.close()
            print()
            vwriter.close()

