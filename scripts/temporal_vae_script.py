# Environment $ nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=0 -v /data/hoi/gait_analysis:/mnt yyhhoi/neuro:1 bash

def run_train_and_vis_on_tvae():
    from TemporalVAE.TemporalVAE_run import GaitTVAEmodel, GaitCVAEvisualiser
    from common.generator import GaitGeneratorFromDFforTemporalVAE
    import os

    df_path = "/mnt/data/raw_features_zmatrix_row_labels.pickle"
    kld_list = ([100, 150, 0.0001],)
    # latent_dims_list = (20,)
    latent_dims_list = (100,)
    hidden_units = 512
    dropout_p = 0
    times = 128
    init_lr = 0.001
    lr_milestones = [15]
    lr_decay_gamma = 0.1

    u_neighbors = [15,]
    min_dists = [0.1,]
    metrics = ["euclidean"]
    pcas = [False]

    for kld in kld_list:
        for latent_dims in latent_dims_list:
            print("Drop = {} | KLD = {} | Latent_dims = {} | hidden = {}".format(dropout_p, kld, latent_dims, hidden_units))

            model_identifier = "Drop-{}_KLD-{}_l-{}_h-{}".format(dropout_p, kld, latent_dims, hidden_units)

            # Train
            # data_gen = GaitGeneratorFromDFforTemporalVAE(df_path, m=512, n=times)
            # save_model_path = "TemporalVAE/model_chkpt/ckpt_%s.pth" % (model_identifier)
            # tvae = GaitTVAEmodel(data_gen,
            #                      hidden_channels=hidden_units,
            #                      latent_dims=latent_dims,
            #                      kld=kld,
            #                      dropout_p=dropout_p,
            #                      init_lr=init_lr,
            #                      lr_milestones=lr_milestones,
            #                      lr_decay_gamma=lr_decay_gamma,
            #                      save_chkpt_path=save_model_path)
            # if os.path.isfile(save_model_path):
            #     tvae.load_model(save_model_path)
            # tvae.train(250)

            # Visualize
            data_gen = GaitGeneratorFromDFforTemporalVAE(df_path, m=4000, n=times, seed=60)
            load_model_path = "TemporalVAE/model_chkpt/ckpt_%s.pth" % (model_identifier)
            save_vid_dir = "TemporalVAE/vis/"

            viser = GaitCVAEvisualiser(data_gen, load_model_path, save_vid_dir,
                                       hidden_channels=hidden_units,
                                       latent_dims=latent_dims,
                                       model_identifier=model_identifier,
                                       init_lr=init_lr,
                                       lr_milestones=lr_milestones,
                                       lr_decay_gamma=lr_decay_gamma
                                       )
            # viser.visualise_random_reconstruction_label_clusters(5)

            viser.visualize_umap_embedding(
                n_neighs=u_neighbors,
                min_dists=min_dists,
                metrics=metrics,
                pca_enableds=pcas,
            )