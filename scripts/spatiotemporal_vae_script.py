# Environment $ nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=0 -v /data/hoi/gait_analysis:/mnt yyhhoi/neuro:1 bash

def run_train_and_vis_on_stvae():
    from Spatiotemporal_VAE.STVAE_run import STVAEmodel
    from common.generator import GaitGeneratorFromDFforTemporalVAE
    import os

    df_path = "/mnt/data/raw_features_zmatrix_row_labels.pickle"
    model_chkpt_found = False
    seq_dim = 128
    model_type = "graph"
    # model_type = "normal"
    posenet_latent_dim = 16
    posenet_dropout_p = 0
    posenet_kld = None
    pose_latent_gradient = 0.0001
    motionnet_latent_dim = 128
    motionnet_dropout_p = 0
    motionnet_kld = [200, 250, 0.0001]
    recon_gradient = 0.0001
    class_weight = 0.001
    rmse_weighting_startepoch = 150
    init_lr = 0.001
    lr_milestones = [75, 150]
    lr_decay_gamma = 0.1

    # Train
    data_gen = GaitGeneratorFromDFforTemporalVAE(df_path, m=256, n=seq_dim)
    model_identifier = "{}_pose_l-{}_d-{}_kld-{}_grad-{}_motion_l-{}_d-{}_kld-{}_recongrad-{}_class-{}_rmseW-{}".format(
        model_type,
        posenet_latent_dim,
        posenet_dropout_p,
        posenet_kld,
        pose_latent_gradient,
        motionnet_latent_dim,
        motionnet_dropout_p,
        motionnet_kld,
        recon_gradient,
        class_weight,
        rmse_weighting_startepoch
    )
    save_model_path = "Spatiotemporal_VAE/model_chkpt/ckpt_%s.pth" % model_identifier
    save_vid_dir = "Spatiotemporal_VAE/vis/{}".format(model_identifier)
    if os.path.isdir(save_vid_dir) is not True:
        os.makedirs(save_vid_dir)

    if os.path.isfile(save_model_path):
        print("Model checkpoint identified.")
        model_chkpt_found = True
        load_model_path = save_model_path
    else:
        load_model_path = None

    model_container = STVAEmodel(data_gen=data_gen, fea_dim=50, seq_dim=seq_dim, model_type=model_type,
                                 posenet_latent_dim=posenet_latent_dim,
                                 posenet_dropout_p=posenet_dropout_p, posenet_kld=posenet_kld,
                                 motionnet_latent_dim=motionnet_latent_dim, motionnet_hidden_dim=512,
                                 motionnet_dropout_p=motionnet_dropout_p, motionnet_kld=motionnet_kld,
                                 pose_latent_gradient=pose_latent_gradient, recon_gradient=recon_gradient,
                                 classification_weight=class_weight,
                                 rmse_weighting_startepoch=rmse_weighting_startepoch,
                                 init_lr=init_lr, lr_milestones=lr_milestones, lr_decay_gamma=lr_decay_gamma,
                                 save_chkpt_path=save_model_path, load_chkpt_path=load_model_path)
    # model_container.train(50)

    # Visualization
    if model_chkpt_found:
        data_gen2 = GaitGeneratorFromDFforTemporalVAE(df_path, m=512, n=seq_dim, seed=60)
        model_container.vis_reconstruction(data_gen2, 10, save_vid_dir, model_identifier)
        model_container.save_model_losses_data(save_vid_dir, model_identifier)
    else:
        print("Required chkpt file cannot be found")
