# Environment $ nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=0 -v /data/hoi/gait_analysis:/mnt yyhhoi/neuro:1 bash


def run_train_and_vis_on_ssvae():

    from single_skeleton_vae.VAE_run import GaitVAEmodel
    from common.generator import GaitGeneratorFromDFforTemporalVAE, GaitGeneratorFromDFforSingleSkeletonVAE
    from single_skeleton_vae.VAE_run import GaitSingleSkeletonVAEvisualiser, GaitSingleSkeletonVAEvisualiserCollapsed
    import os

    df_path = "/mnt/data/raw_features_zmatrix_row_labels.pickle"
    save_vid_dir = "single_skeleton_vae/vis/"

    kld_list = (0.001, 0.0001)
    latent_dims_list = (2, )
    drop_p = 0
    space_samples = 6400

    init_lr = 0.001
    lr_milestones = [15, 50]
    lr_decay_gamma = 0.1

    for kld in kld_list:
        for latent_dims in latent_dims_list:

            # Define condition-specific paths/identifiers
            model_identifier = "Drop-{}_KLD-{}_latent-{}".format(drop_p, kld, latent_dims)
            save_model_path = "single_skeleton_vae/model_chkpt/ckpt_{}.pth".format(model_identifier)
            load_model_path = "single_skeleton_vae/model_chkpt/ckpt_{}.pth".format(model_identifier)
            print(model_identifier)

            # Training
            # data_gen = GaitGeneratorFromDFforSingleSkeletonVAE(df_path, m=space_samples, train_portion=0.999)
            # vae = GaitVAEmodel(data_gen=data_gen, input_dims=50, latent_dims=latent_dims, kld=kld, dropout_p=drop_p,
            #                    init_lr=init_lr, lr_milestones=lr_milestones, lr_decay_gamma=lr_decay_gamma,
            #                    save_chkpt_path=save_model_path, data_gen_type="single")
            #
            # if os.path.isfile(load_model_path):
            #     vae.load_model(load_model_path)
            # vae.train(100)
            #
            # # Visualize low-dimensional space
            # data_gen = GaitGeneratorFromDFforSingleSkeletonVAE(df_path, m=space_samples, train_portion=0.999)
            # viser = GaitSingleSkeletonVAEvisualiser(data_gen=data_gen, load_model_path=load_model_path,
            #                                         save_vid_dir=save_vid_dir, latent_dims=latent_dims,
            #                                         kld=kld, dropout_p=drop_p, model_identifier=model_identifier,
            #                                         data_gen_type="single")
            # viser.visualise_latent_space()

            # Visualize action sequence
            data_gen = GaitGeneratorFromDFforTemporalVAE(df_path, m=50, seed=60)
            viser = GaitSingleSkeletonVAEvisualiser(data_gen=data_gen, load_model_path=load_model_path,
                                                    save_vid_dir=save_vid_dir, latent_dims=latent_dims,
                                                    kld=kld, dropout_p=drop_p, model_identifier=model_identifier,
                                                    data_gen_type="temporal")
            viser.visualise_vid()