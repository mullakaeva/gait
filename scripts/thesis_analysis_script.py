# Environment $ nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=0 -v /data/hoi/gait_analysis:/mnt yyhhoi/neuro:1 bash

from Spatiotemporal_VAE.Containers import BaseContainer, ConditionalContainer, PhenoCondContainer
from common.generator import GaitGeneratorFromDFforTemporalVAE
from common.utils import dict2json, json2dict
from Spatiotemporal_VAE.analysis_scripts.latent_space_visualization import LatentSpaceSaver_CondDirect, \
    LatentSpaceSaver_CondDirectTask, LatentSpaceSaver_CondDirectLeg, LatentSpaceSaver_CondDirectIdentity
from Spatiotemporal_VAE.analysis_scripts.fingerprint import GaitprintSaver
import os
import pprint


def print_model_info(model_identifier, hyper_params):
    print("%s's hyper-paramters:" % model_identifier)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(hyper_params)


def load_model_container(model_class, model_identifier, df_path, datagen_batch_size=512, gaitprint_completion=False,
                         train_portion=0.99, seed=0):
    # Hard-coded stuffs
    seq_dim = 128
    init_lr = 0.001
    lr_milestones = [75]
    lr_decay_gamma = 0.1

    # Hyper-parameters
    hyper_params = {
        "model_name": model_identifier,
        "conditional_label_dim": 3,
        "recon_weight": 1,
        "posenet_latent_dim": 16,
        "posenet_dropout_p": 0,
        "posenet_kld": None,
        "pose_latent_gradient": 0.0001,  # 0.0001
        "motionnet_latent_dim": 128,
        "motionnet_dropout_p": 0,
        "motionnet_kld": [0, 250, 0.0001],  # [200, 250, 0.0001],
        "recon_gradient": 0.0001,  # 0.0001
        "class_weight": 0.001,  # 0.001
        "latent_recon_loss": 1,
    }
    save_model_path = "Spatiotemporal_VAE/model_chkpt/ckpt_%s.pth" % model_identifier
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

    data_gen = GaitGeneratorFromDFforTemporalVAE(df_path, m=datagen_batch_size, n=seq_dim, train_portion=train_portion,
                                                 gait_print=gaitprint_completion, seed=seed)

    model_container = model_class(data_gen=data_gen,
                                  fea_dim=50,
                                  seq_dim=seq_dim,
                                  conditional_label_dim=hyper_params["conditional_label_dim"],
                                  posenet_latent_dim=hyper_params["posenet_latent_dim"],
                                  posenet_dropout_p=hyper_params["posenet_dropout_p"],
                                  posenet_kld=hyper_params["posenet_kld"],
                                  motionnet_latent_dim=hyper_params["motionnet_latent_dim"],
                                  motionnet_hidden_dim=512,
                                  motionnet_dropout_p=hyper_params["motionnet_dropout_p"],
                                  motionnet_kld=hyper_params["motionnet_kld"],
                                  pose_latent_gradient=hyper_params["pose_latent_gradient"],
                                  recon_gradient=hyper_params["recon_gradient"],
                                  classification_weight=hyper_params["class_weight"],
                                  latent_recon_loss=hyper_params["latent_recon_loss"],
                                  init_lr=init_lr,
                                  lr_milestones=lr_milestones,
                                  lr_decay_gamma=lr_decay_gamma,
                                  save_chkpt_path=save_model_path,
                                  load_chkpt_path=load_model_path)
    return model_container, save_model_path


def run_train_and_vis_on_stvae():
    df_path = "/mnt/data/fea_thesis_analysis_37836_allLabelled_withoutLeg.pickle"
    # model_identifier = "Thesis_B"
    # model_identifier = "Thesis_B+T"
    # model_identifier = "Thesis_B+T+C"
    model_identifier = "Thesis_B+T+C+P"
    gaitprint_completion = True  # True for B+T+C+P, False for others
    batch_size = 64  # 64 for B+T+C+P, 512 for others
    model_container, save_model_path = load_model_container(model_class=PhenoCondContainer,
                                                            model_identifier=model_identifier,
                                                            df_path=df_path,
                                                            datagen_batch_size=batch_size,
                                                            gaitprint_completion=gaitprint_completion,
                                                            train_portion=0.80,
                                                            seed=0)
    model_container.train(300)

    # # Visualization
    # if os.path.isfile(save_model_path):
    #     data_gen2 = GaitGeneratorFromDFforTemporalVAE(df_path,
    #                                                   m=model_container.data_gen.num_rows-1,
    #                                                   n=model_container.seq_dim,
    #                                                   train_portion=0.99,
    #                                                   seed=60)
    #     viser = LatentSpaceSaver_CondDirect(
    #         model_container=model_container,
    #         data_gen=data_gen2,
    #         fit_samples_num=4096,
    #         save_data_dir="/mnt/JupyterNotebook/interactive_latent_exploration/data",
    #         df_save_fn="LatentSpace_Cond-Direct-alldata.pickle",
    #         vid_dirname="Cond-Direct-alldata",
    #         model_identifier=model_identifier,
    #         draw=True
    #     )
    #     viser.process()

    # else:
    #     print("Chkpt cannot be found")
