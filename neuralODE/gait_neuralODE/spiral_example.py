from latent_ode import generate_spiral2d, LatentODEModel, ExampleSpirals
import numpy as np
import matplotlib.pyplot as plt


def model_data_initialisation(latent_dim=4,
                              nhidden=20,
                              rnn_nhidden=25,
                              obs_dim=2,
                              lr=0.01,
                              noise_std=0.3,
                              gpu_num=0):
    orig_trajs, samp_trajs, orig_ts, samp_ts = ExampleSpirals()
    model = LatentODEModel(xt=samp_trajs,
                           t=samp_ts,
                           latent_dim=latent_dim,
                           n_hidden=nhidden,
                           rnn_n_hidden=rnn_nhidden,
                           obs_dim=obs_dim,
                           device_num=gpu_num,
                           lr=lr,
                           save_chkpt_dir="model_chkpt",
                           noise_std=noise_std)
    return model, (orig_trajs, samp_trajs, orig_ts, samp_ts)

def gen_latent(dim, start=0, end=2, num_samples=10):
    latent_vecs = np.zeros((num_samples, 4))
    latent_vecs[:, dim] = np.linspace(start, end, num_samples)
    return latent_vecs

def gen_latent_random(start=0, end=2, num_samples=10):
    latent_vecs = np.random.uniform(start, end, (num_samples, 4))
    return latent_vecs

def training():
    num_iters = 1000
    model, data_info = model_data_initialisation()
    model.load_saved_progress("model_chkpt/ckpt.pth")
    model.train(num_iters)

def visualise_latents():
    # Define color maps
    cmap_dict = {
        0: plt.get_cmap("spring"),
        1: plt.get_cmap("summer"),
        2: plt.get_cmap("autumn"),
        3: plt.get_cmap("winter")
    }
    latents_num_per_dim = 5

    # Generate original and sampled data
    model, data_info = model_data_initialisation()
    model.load_saved_progress("model_chkpt/ckpt.pth")
    orig_trajs, samp_trajs, orig_ts, samp_ts = data_info

    # Plot original and sampled data
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.plot(orig_trajs[0, :, 0], orig_trajs[0, :, 1], c="r", label="GT")
    ax.plot(orig_trajs[1, :, 0], orig_trajs[1, :, 1], c="r", label="GT")
    for i in range(10):
        ax.plot(samp_trajs[i, :, 0], samp_trajs[i, :, 1], label="Samp_{}".format(i), linewidth=0.5, alpha=0.5)
        ax.scatter(samp_trajs[i, 0, 0], samp_trajs[i, 0, 1], marker="x")
    fig.savefig("Orig_Samps_trags.png")

    # Sample outputs from latent vectors
    # num_latents = 10
    # latent_vecs = gen_latent_random(-2, 2, 10)
    # for i in range(num_latents):
    #     fig_out, ax_out = plt.subplots(figsize=(12, 12))
    #     times = np.linspace(0, 6 * np.pi, 100)
    #
    #     xt = model.sample_from_latent(latent_vecs[[i], :], times)
    #
    #     ax_out.plot(xt[0, :, 0], xt[0, :, 1])
    #     ax_out.scatter(xt[:, 0, 0], xt[:, 0, 1],
    #                    marker="x",
    #                    )
    #     fig_out.savefig("latents_visualisation/random/Decoded_latents{}_.png".format(i))


    for dim_each in range(4):
        fig_out, ax_out = plt.subplots(figsize=(12, 12))
        # Create latents with non-zero entries in only one dimension
        latent_vecs = gen_latent(dim_each, start=-2, end=2, num_samples=latents_num_per_dim)
        val_range = np.max(latent_vecs) - np.min(latent_vecs)

        # Create times and infer
        times = np.linspace(0, 6*np.pi, 100)
        xt = model.sample_from_latent(latent_vecs, times)

        # Plot
        for i in range(latent_vecs.shape[0]):
            latent_val = latent_vecs[i, dim_each]
            xt_each = xt[i, :, :]
            ax_out.plot(xt_each[:, 0], xt_each[:, 1],
                        c=cmap_dict[dim_each](latent_val/val_range),
                        label="dim_%d | n_%0.3f" % (dim_each, latent_val))
            ax_out.scatter(xt_each[0, 0], xt_each[0, 1],
                           marker="x",
                           c=cmap_dict[dim_each](latent_val / val_range)
                           )
        ax_out.legend()
        fig_out.savefig("latents_visualisation/Decoded_latents{}_.png".format(dim_each))



if __name__ == "__main__":
    # training()
    visualise_latents()
