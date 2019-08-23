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
    if df_path:
        data_gen = GaitGeneratorFromDFforTemporalVAE(df_path, m=datagen_batch_size, n=seq_dim,
                                                     train_portion=train_portion,
                                                     gait_print=gaitprint_completion, seed=seed)
    else:
        data_gen = None

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
    df_path = "/mnt/data/full_feas_tasks_phenos_nanMasks_idpatient_leg.pickle"
    # model_identifier = "Thesis_B"
    # model_identifier = "Thesis_B+C"
    model_identifier = "Thesis_B+C+T"
    # model_identifier = "Thesis_B+C+T+P"

    gaitprint_completion = False  # True for B+T+C+P, False for others
    batch_size = 512  # 64 for Thesis_B+C+T+P, 512 for others
    model_container, save_model_path = load_model_container(model_class=ConditionalContainer,
                                                            model_identifier=model_identifier,
                                                            df_path=df_path,
                                                            datagen_batch_size=batch_size,
                                                            gaitprint_completion=gaitprint_completion,
                                                            train_portion=0.80,
                                                            seed=0)
    model_container.train(992)


def run_save_model_outputs():
    from Spatiotemporal_VAE.analysis_scripts.thesis_save_model_outputs import OutputSavers
    df_path = "/mnt/data/full_feas_tasks_phenos_nanMasks_idpatient_leg.pickle"
    df_save_path = "/mnt/thesis_results/data/model_outputs_full.pickle"
    df_pheno_save_path = "/mnt/thesis_results/data/model_phenos_outputs_full.pickle"

    identifier_set = ["Thesis_B", "Thesis_B+C", "Thesis_B+C+T", "Thesis_B+C+T+P"]
    model_classess = [BaseContainer, ConditionalContainer, ConditionalContainer, PhenoCondContainer]
    model_container_set = []

    data_gen = GaitGeneratorFromDFforTemporalVAE(df_path, m=512, n=128,
                                                 train_portion=0.8,
                                                 gait_print=False, seed=0)



    for model_identifier, model_class in zip(identifier_set, model_classess):
        model_container, _ = load_model_container(model_class=model_class,
                                                  model_identifier=model_identifier,
                                                  df_path=df_path,
                                                  datagen_batch_size=512,
                                                  gaitprint_completion=False,
                                                  train_portion=0.80,
                                                  seed=0)
        model_container_set.append(model_container)

    saver = OutputSavers(data_gen=data_gen,
                         model_container_set=model_container_set,
                         identifier_set=identifier_set,
                         save_df_path=df_save_path,
                         save_pheno_df_path=df_pheno_save_path)
    saver.forward_batch()