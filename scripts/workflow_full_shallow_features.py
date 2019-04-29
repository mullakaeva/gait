# from common.preprocess import openpose_preprocess_wrapper
# from common.feature_extraction import FeatureExtractorForODE
# from common.generator import GaitGeneratorFromDF
# from neuralODE.analysis_neuralODE import gait_neural_ode_train, gait_neural_ode_vis

# from cVAE.cVAE_run import GaitCVAEmodel, GaitCVAEvisualiser


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

# %% ======================== Step A.A.4: Train on cVAE =======================
# Environment $ nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=0 -v /data/hoi/gait_analysis:/mnt yyhhoi/neuro:1 bash
# data_gen = GaitGeneratorFromDFforCVAE("/mnt/data/raw_features_zmatrix_row_labels.pickle",
#                                       m=512)
# save_model_path = "cVAE/model_chkpt/ckpt.pth"
# cvae = GaitCVAEmodel(data_gen, save_chkpt_path=save_model_path)
# # cvae.load_model(save_model_path)
# cvae.train(10)

# %% ======================== Step A.A.5: Visualise on cVAE =======================
# Environment $ nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=0 -v /data/hoi/gait_analysis:/mnt yyhhoi/neuro:1 bash
# data_gen = GaitGeneratorFromDFforCVAE("/mnt/data/raw_features_zmatrix_row_labels.pickle",
#                                       m=512)
# load_model_path = "cVAE/model_chkpt/ckpt.pth"
# save_vid_dir = "cVAE/vis/"
# viser = GaitCVAEvisualiser(data_gen, load_model_path, save_vid_dir)
# viser.visualise_vid()

# %% ======================== Step A.B.4: Train on single_skeleton_VAE =======================
# from common.generator import GaitGeneratorFromDFforSingleSkeletonVAE
# from single_skeleton_vae.VAE_run import GaitVAEmodel
#
# data_gen = GaitGeneratorFromDFforSingleSkeletonVAE("/mnt/data/raw_features_zmatrix_row_labels.pickle",
#                                                    m=8192, train_portion=0.999)
# save_model_path = "single_skeleton_vae/model_chkpt/ckpt.pth"
# vae = GaitVAEmodel(data_gen, latent_dims=2, save_chkpt_path=save_model_path)
# # vae.load_model(save_model_path)
# vae.train(10)

# %% ======================== Step A.B.5: Visualise on single_skeleton_VAE =======================
# Environment $ nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=0 -v /data/hoi/gait_analysis:/mnt yyhhoi/neuro:1 bash
from common.generator import GaitGeneratorFromDFforCVAE
from single_skeleton_vae.VAE_run import GaitSingleSkeletonVAEvisualiser
data_gen = GaitGeneratorFromDFforCVAE("/mnt/data/raw_features_zmatrix_row_labels.pickle",
                                      m=5)
load_model_path = "single_skeleton_vae/model_chkpt/ckpt.pth"
save_vid_dir = "single_skeleton_vae/vis/"
viser = GaitSingleSkeletonVAEvisualiser(data_gen, load_model_path, save_vid_dir)
viser.visualise_vid()
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
