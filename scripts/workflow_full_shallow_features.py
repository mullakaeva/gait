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
# Environment $ nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=0 -v /data/hoi/gait_analysis:/mnt yyhhoi/neuro:1 bash

from TemporalVAE.TemporalVAE_run import GaitTVAEmodel, GaitCVAEvisualiser
from common.generator import GaitGeneratorFromDFforTemporalVAE
import numpy as np
import os
kld_consts = (1e-10, )
velo_consts = (1e-10, )
for kld_const in kld_consts:
    for velo_const in velo_consts:
        print("KLD = %f | velo = %f" % (kld_const, velo_const))
        kld_identifier = -np.log10(kld_const)
        velo_identifier = -np.log10(velo_const)

        model_identifier = "NoVar"
        # model_identifier = "KLD-%f_velo-%f" % (kld_identifier, velo_identifier)

        # Train
        data_gen = GaitGeneratorFromDFforTemporalVAE("/mnt/data/raw_features_zmatrix_row_labels.pickle",
                                                     m=512)
        save_model_path = "TemporalVAE/model_chkpt/ckpt_%s.pth" % (model_identifier)
        tvae = GaitTVAEmodel(data_gen,
                             KLD_const=kld_const,
                             velo_const=velo_const,
                             save_chkpt_path=save_model_path)
        if os.path.isfile(save_model_path):
            tvae.load_model(save_model_path)
        tvae.train(20)
        # Visualize
        data_gen = GaitGeneratorFromDFforTemporalVAE("/mnt/data/raw_features_zmatrix_row_labels.pickle",
                                                     m=4000)
        load_model_path = "TemporalVAE/model_chkpt/ckpt_%s.pth" % (model_identifier)
        save_vid_dir = "TemporalVAE/vis/"

        viser = GaitCVAEvisualiser(data_gen, load_model_path, save_vid_dir, model_identifier=model_identifier)
        viser.visualise_random_reconstruction_label_clusters(5)

# %% ======================== Step A.B.4: Train and visualize on single_skeleton_VAE =======================
# Environment $ nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=0 -v /data/hoi/gait_analysis:/mnt yyhhoi/neuro:1 bash

# from single_skeleton_vae.VAE_run import GaitVAEmodel
# from common.generator import GaitGeneratorFromDFforTemporalVAE, GaitGeneratorFromDFforSingleSkeletonVAE
# from single_skeleton_vae.VAE_run import GaitSingleSkeletonVAEvisualiser, GaitSingleSkeletonVAEvisualiserCollapsed
#
#
# save_model_path = "single_skeleton_vae/model_chkpt/ckpt.pth"
# load_model_path = "single_skeleton_vae/model_chkpt/ckpt.pth"
#
# KLD_consts = (0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001)
# # KLD_consts = (0.01,)
# space_samples = 6400
# for KLD_const in KLD_consts:
#     save_model_path = "single_skeleton_vae/model_chkpt/ckpt_KLD-%f.pth" % KLD_const
#     load_model_path = "single_skeleton_vae/model_chkpt/ckpt_KLD-%f.pth" % KLD_const
#     data_gen = GaitGeneratorFromDFforSingleSkeletonVAE("/mnt/data/raw_features_zmatrix_row_labels.pickle",
#                                                        m=space_samples, train_portion=0.999)
#     vae = GaitVAEmodel(data_gen, latent_dims=2, step_lr_decay=0.8, KLD_const=KLD_const,
#                        save_chkpt_path=save_model_path)
#     # vae.load_model(save_model_path)
#     # vae.train(5)
#
#     save_vid_dir = "single_skeleton_vae/vis/"
#     data_gen = GaitGeneratorFromDFforSingleSkeletonVAE("/mnt/data/raw_features_zmatrix_row_labels.pickle",
#                                                        m=space_samples, train_portion=0.999)
#     # data_gen = GaitGeneratorFromDFforTemporalVAE("/mnt/data/raw_features_zmatrix_row_labels.pickle", m=50)
#     viser = GaitSingleSkeletonVAEvisualiser(data_gen, load_model_path, save_vid_dir, latent_dims=2)
#     viser.visualise_latent_space()
#     # viser.visualise_vid()

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
