# Environment $ nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=0 -v /data/hoi/gait_analysis:/mnt yyhhoi/neuro:1 bash
from Spatiotemporal_VAE.STVAE_run import STVAEmodel
from common.generator import GaitGeneratorFromDFforTemporalVAE
from common.utils import dict2json, json2dict
import os
import pprint

def print_model_info(model_identifier, hyper_params):
    print("%s's hyper-paramters:" % model_identifier)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(hyper_params)


def run_train_and_vis_on_stvae():
    # Hard-coded stuffs
    seq_dim = 128
    init_lr = 0.001
    lr_milestones = [75]
    lr_decay_gamma = 0.1

    # Naming of models: N=Normal
    model_identifier = "NB-K(0.00001)-C-G-S2-New"

    # Hyper-parameters
    hyper_params = {
        "model_name": model_identifier,
        "model_type": "normal",
        "recon_weight": 1,
        "posenet_latent_dim": 16,
        "posenet_dropout_p": 0,
        "posenet_kld": None,
        "pose_latent_gradient": 0.0001,  # 0.0001
        "motionnet_latent_dim": 128,
        "motionnet_dropout_p": 0,
        "motionnet_kld": [200, 250, 0.00001],  # [200, 250, 0.0001],
        "recon_gradient": 0.0001,  # 0.0001
        "class_weight": 0.001,  # 0.001
        "rmse_weighting_startepoch": None,
        "latent_recon_loss": 1,
        "recon_loss_power": 2
    }

    # Define paths
    # df_path = "/mnt/data/raw_features_zmatrix_row_labels_withNanMasks.pickle"

    df_path = "/mnt/data/feas_incompleteLabels_nanMasks.pickle"
    save_model_path = "Spatiotemporal_VAE/model_chkpt/ckpt_%s.pth" % model_identifier
    project_dir = "Spatiotemporal_VAE"
    save_hyper_params_path = "Spatiotemporal_VAE/model_chkpt/hyperparms_%s.json" % model_identifier

    if os.path.isfile(save_model_path):
        print("Model checkpoint identified.")
        load_model_path = save_model_path
    else:
        load_model_path = None

    if os.path.isfile(save_hyper_params_path):
        print("Existing hyper-params file found")
        hyper_params = json2dict(save_hyper_params_path)
    else:
        print("Hyper-params file saved")
        dict2json(save_hyper_params_path, hyper_params)

    print_model_info(model_identifier, hyper_params)

    # Train
    data_gen = GaitGeneratorFromDFforTemporalVAE(df_path, m=512, n=seq_dim)

    model_container = STVAEmodel(data_gen=data_gen, fea_dim=50, seq_dim=seq_dim, model_type=hyper_params["model_type"],
                                 posenet_latent_dim=hyper_params["posenet_latent_dim"],
                                 posenet_dropout_p=hyper_params["posenet_dropout_p"],
                                 posenet_kld=hyper_params["posenet_kld"],
                                 motionnet_latent_dim=hyper_params["motionnet_latent_dim"], motionnet_hidden_dim=512,
                                 motionnet_dropout_p=hyper_params["motionnet_dropout_p"],
                                 motionnet_kld=hyper_params["motionnet_kld"],
                                 pose_latent_gradient=hyper_params["pose_latent_gradient"],
                                 recon_gradient=hyper_params["recon_gradient"],
                                 classification_weight=hyper_params["class_weight"],
                                 rmse_weighting_startepoch=hyper_params["rmse_weighting_startepoch"],
                                 latent_recon_loss=hyper_params["latent_recon_loss"],
                                 init_lr=init_lr, lr_milestones=lr_milestones, lr_decay_gamma=lr_decay_gamma,
                                 save_chkpt_path=save_model_path, load_chkpt_path=load_model_path)
    # model_container._save_model()

    model_container.train(200)

    # # Visualization
    # if os.path.isfile(save_model_path):
    #     data_gen2 = GaitGeneratorFromDFforTemporalVAE(df_path, m=4096, n=seq_dim, seed=60)
    #     model_container.vis_reconstruction(data_gen2, 10, project_dir, model_identifier)
    #     # model_container.save_model_losses_data(project_dir, model_identifier)
    #     # model_container.evaluate_all_models(data_gen2, project_dir, None, draw_vid=True)
    # else:
    #     print("Chkpt cannot be found")
