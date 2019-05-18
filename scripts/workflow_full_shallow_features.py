# from common.preprocess import openpose_preprocess_wrapper
# from common.feature_extraction import FeatureExtractorForODE
# from common.generator import GaitGeneratorFromDF
# from neuralODE.analysis_neuralODE import gait_neural_ode_train, gait_neural_ode_vis

# from TemporalVAE.cVAE_run import GaitCVAEmodel, GaitCVAEvisualiser


# %%  ======================= Step 1: OpenPose inference ============================
# This section find all videos from Mustafa's gait data, select those that has labels and infer them
# 1. $ nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=0 -v /:/mnt yyhhoi/openpose:u16cuda9dnn7-2 bash
# 2. # cp -r /mnt/data/hoi/gait_analysis/scripts/openpose_shellscripts/generate_openpose_shellscript_for_FSF.py /mnt/data/hoi/gait_analysis/scripts/common ./
# 3. # python generate_openpose_shellscript_for_FSF.py
# 4. # sh openpose_inference_script.sh
# Configuration of behaviours is stored in the generate_openpose_shellscript_for_FSF.py above


# %% ======================== Step 2: Keypoints Pre-processing =======================
# Environment $ nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=0 -v /:/mnt yyhhoi/neuro:1 bash
# src_vid_dir = "/mnt/data/gait/data/videos_mp4/"
# input_data_main_dir = "/mnt/data/hoi/gait_analysis/data/openpose_keypoints"
# output_vid_dir = "/mnt/data/hoi/gait_analysis/data/preprocessed_visualisation"
# output_data_dir = "/mnt/data/hoi/gait_analysis/data/preprocessed_keypoints"
# error_log_path = "/mnt/data/hoi/gait_analysis/logs/preprocess_error_log.txt"
# openpose_preprocess_wrapper(src_vid_dir, input_data_main_dir, output_vid_dir, output_data_dir, error_log_path,
#                             plot_keypoints=True)

# %% ======================== Step 3: Extracting feature for ODE =======================
# Environment $ nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=0 -v /data/hoi/gait_analysis:/mnt yyhhoi/neuro:1 bash
# scr_keyps_dir = "/mnt/data/preprocessed_keypoints"
# labels_path = "/mnt/data/labels/z_matrix/df_gait_vid_linked_190718.pkl"
# df_save_path = "/mnt/data/raw_features_zmatrix_row_labels.pickle"
# minimum_sequence_window = 128
# extractor = FeatureExtractorForODE(scr_keyps_dir=scr_keyps_dir,
#                                    labels_path=labels_path,
#                                    df_save_path=df_save_path)
# extractor.extract(minimum_sequence_window)

# %% ======================== Step A.A.4: Train on temporal_VAE =======================
# # Environment $ nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=0 -v /data/hoi/gait_analysis:/mnt yyhhoi/neuro:1 bash
#
from TemporalVAE.TemporalVAE_run import GaitTVAEmodel, GaitCVAEvisualiser
from common.generator import GaitGeneratorFromDFforTemporalVAE
import os

# df_path = "/mnt/data/raw_features_zmatrix_row_labels.pickle"
# kld_list = (None,)
# latent_dims_list = (20,)
# # latent_dims_list = (1600,)
# hidden_units = 512
# dropout_p = 0
# times = 128
# init_lr = 0.001
# lr_milestones = [15]
# lr_decay_gamma = 0.1
#
# u_neighbors = [15,]
# min_dists = [0.1,]
# metrics = ["euclidean"]
# pcas = [False]
#
# for kld in kld_list:
#     for latent_dims in latent_dims_list:
#         print("Drop = {} | KLD = {} | Latent_dims = {} | hidden = {}".format(dropout_p, kld, latent_dims, hidden_units))
#
#         model_identifier = "Drop-{}_KLD-{}_l-{}_h-{}".format(dropout_p, kld, latent_dims, hidden_units)
#
#         # Train
#         data_gen = GaitGeneratorFromDFforTemporalVAE(df_path, m=512, n=times)
#         save_model_path = "TemporalVAE/model_chkpt/ckpt_%s.pth" % (model_identifier)
#         tvae = GaitTVAEmodel(data_gen,
#                              hidden_channels=hidden_units,
#                              latent_dims=latent_dims,
#                              kld=kld,
#                              dropout_p=dropout_p,
#                              init_lr=init_lr,
#                              lr_milestones=lr_milestones,
#                              lr_decay_gamma=lr_decay_gamma,
#                              save_chkpt_path=save_model_path)
#         if os.path.isfile(save_model_path):
#             tvae.load_model(save_model_path)
#         tvae.train(250)
#
#         # Visualize
#         data_gen = GaitGeneratorFromDFforTemporalVAE(df_path, m=4000, n=times, seed=60)
#         load_model_path = "TemporalVAE/model_chkpt/ckpt_%s.pth" % (model_identifier)
#         save_vid_dir = "TemporalVAE/vis/"
#
#         viser = GaitCVAEvisualiser(data_gen, load_model_path, save_vid_dir,
#                                    hidden_channels=hidden_units,
#                                    latent_dims=latent_dims,
#                                    model_identifier=model_identifier,
#                                    init_lr=init_lr,
#                                    lr_milestones=lr_milestones,
#                                    lr_decay_gamma=lr_decay_gamma
#                                    )
#         viser.visualise_random_reconstruction_label_clusters(5)
#
#         viser.visualize_umap_embedding(
#             n_neighs=u_neighbors,
#             min_dists=min_dists,
#             metrics=metrics,
#             pca_enableds=pcas,
#         )

# %% ======================== Step A.B.4: Train and visualize on single_skeleton_VAE =======================
# Environment $ nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=0 -v /data/hoi/gait_analysis:/mnt yyhhoi/neuro:1 bash

from single_skeleton_vae.VAE_run import GaitVAEmodel
from common.generator import GaitGeneratorFromDFforTemporalVAE, GaitGeneratorFromDFforSingleSkeletonVAE
from single_skeleton_vae.VAE_run import GaitSingleSkeletonVAEvisualiser, GaitSingleSkeletonVAEvisualiserCollapsed
import os

df_path = "/mnt/data/raw_features_zmatrix_row_labels.pickle"
save_vid_dir = "single_skeleton_vae/vis/"

kld_list = (None,)
latent_dims_list = (3, )
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
        data_gen = GaitGeneratorFromDFforSingleSkeletonVAE(df_path, m=space_samples, train_portion=0.999)
        vae = GaitVAEmodel(data_gen=data_gen, input_dims=50, latent_dims=latent_dims, kld=kld, dropout_p=drop_p,
                           init_lr=init_lr, lr_milestones=lr_milestones, lr_decay_gamma=lr_decay_gamma,
                           save_chkpt_path=save_model_path, data_gen_type="single")

        if os.path.isfile(load_model_path):
            vae.load_model(load_model_path)
        vae.train(150)

        # Visualize low-dimensional space
        # data_gen = GaitGeneratorFromDFforSingleSkeletonVAE(df_path, m=space_samples, train_portion=0.999)
        # viser = GaitSingleSkeletonVAEvisualiser(data_gen=data_gen, load_model_path=load_model_path,
        #                                         save_vid_dir=save_vid_dir, latent_dims=latent_dims,
        #                                         kld=kld, drop_p=drop_p, model_identifier=model_identifier,
        #                                         data_gen_type="single")
        # viser.visualise_latent_space()

        # Visualize action sequence
        data_gen = GaitGeneratorFromDFforTemporalVAE(df_path, m=50, seed=60)
        viser = GaitSingleSkeletonVAEvisualiser(data_gen=data_gen, load_model_path=load_model_path,
                                                save_vid_dir=save_vid_dir, latent_dims=latent_dims,
                                                kld=kld, dropout_p=drop_p, model_identifier=model_identifier,
                                                data_gen_type="temporal")
        viser.visualise_vid()
#

# %% ======================== (Defunkt) Step A.C.4: Train on neural ODE =======================
# Environment $ nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=0 -v /data/hoi/gait_analysis:/mnt yyhhoi/neuro:1 bash
# model_path = "neuralODE/gait_ODE_chkpt/ckpt.pth"
# data_gen = GaitGeneratorFromDF("/mnt/data/raw_features_zmatrix_row_labels.pickle",
#                                m=512)
# gait_neural_ode_train(data_gen)

# %% ======================== (Defunkt) Step A.C.4: Visualise =======================
# data_gen = GaitGeneratorFromDF("/mnt/data/raw_features_zmatrix_row_labels.pickle",
#                                m=512)
# model_path = "neuralODE/gait_ODE_chkpt/ckpt.pth"
# gait_neural_ode_vis(model_path, "neuralODE/gait_vis_results", data_gen)
