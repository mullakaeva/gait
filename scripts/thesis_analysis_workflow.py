# -*- coding: utf-8 -*-

# This file will guide you all the steps that I did for the analysis in the Master thesis --Hoi


# %%  ======================= Step 1: OpenPose inference ============================
# This section find all videos from Mustafa's gait data, select those with labels and infer them with openpose
# Configuration of OpenPose behaviours is stored in the "generate_openpose_shellscript_for_FSF.py" as shown below

# 1. Environment $ NV_GPU=0,1,2 nvidia-docker run --rm -it -v /:/mnt yyhhoi/openpose:u16cuda9dnn7-2 bash
# 2. # cp -r /mnt/data/hoi/gait_analysis/scripts/openpose_shellscripts/generate_openpose_shellscript_for_FSF.py /mnt/data/hoi/gait_analysis/scripts/common ./
# 3. # python generate_openpose_shellscript_for_FSF.py
# 4. # sh openpose_inference_script.sh | tee /mnt/data/hoi/gait_analysis/scripts/openpose_shellscripts/clt_output.txt


# %% ======================== Step 2: Keypoints Pre-processing Part 1 =======================
# It is the first part of the preprocessing in the thesis. It includes
# 1. Eliminate the flipping artefact by reversing the wrong coordinates
# 2. Translation of keypoints to the bounding box's cooridnate system
# 3. Crop videos to their bounding box of human subject and resize the bounding box to a fixed size
# 4. Torso length normalization
# 5. Extract only the video segment with whole skeleton visible

# Environment $ nvidia-docker run --rm -it -e NV_GPU=0 -v /:/mnt yyhhoi/neuro:3 bash
# from common.preprocess import openpose_preprocess_wrapper
# src_vid_dir = "/mnt/media/dsgz2tb_2/videos_converted"
# input_data_main_dir = "/mnt/data/hoi/gait_analysis/data/openpose_keypoints"
# output_vid_dir = "/mnt/data/hoi/gait_analysis/data/preprocessed_visualisation"
# output_data_dir = "/mnt/data/hoi/gait_analysis/data/preprocessed_keypoints"
# error_log_path = "/mnt/data/hoi/gait_analysis/logs/preprocess_error_log.txt"
# openpose_preprocess_wrapper(src_vid_dir, input_data_main_dir, output_vid_dir, output_data_dir,
#                             error_log_path=error_log_path,
#                             write_video=False,
#                             plot_keypoints=False)

# %% ======================== Step 3: Keypoints Pre-processing Part 2 =======================
# This section do the following
# 1. Further normalization, clipping, walking direction detection and so on.
# 2. Pack all information into a dataframe (see the class doctring in common.feature_extaction.FeatureExtractorForODE)
# Environment $ nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=0 -v /data/hoi/gait_analysis:/mnt yyhhoi/neuro:2 bash
# from common.feature_extraction import FeatureExtractorForODE
# scr_keyps_dir = "/mnt/data/preprocessed_keypoints"
# labels_path = "/mnt/data/labels/fn_tasks_phenos_validated_rename.pkl"
# df_save_path = "/mnt/data/full_feas_tasks_phenos_nanMasks_idpatient_leg.pickle"
# minimum_sequence_window = 128  # Predefined fixed video segment length
# extractor = FeatureExtractorForODE(scr_keyps_dir=scr_keyps_dir,
#                                    labels_path=labels_path,
#                                    df_save_path=df_save_path)
# extractor.extract(minimum_sequence_window)

# %% ======================== Step 4: Train and visualize on combined_VAE =======================
# Environment $ nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=0 -v /data/hoi/gait_analysis:/mnt yyhhoi/neuro:3 bash
# from spatiotemporal_vae_script import run_train_and_vis_on_stvae, dual_fingerprint_analysis, single_fingerprint_analysis
from thesis_analysis_script import run_train_and_vis_on_stvae, run_save_model_outputs
# run_train_and_vis_on_stvae()
run_save_model_outputs()
# dual_fingerprint_analysis()
# single_fingerprint_analysis()





