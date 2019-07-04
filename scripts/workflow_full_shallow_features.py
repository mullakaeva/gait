# from common.generator import GaitGeneratorFromDF
# from neuralODE.analysis_neuralODE import gait_neural_ode_train, gait_neural_ode_vis

# from TemporalVAE.cVAE_run import GaitCVAEmodel, GaitCVAEvisualiser


# %%  ======================= Step 1: OpenPose inference ============================
# This section find all videos from Mustafa's gait data, select those that has labels and infer them
# 1. $ NV_GPU=0,1,2 nvidia-docker run --rm -it -v /:/mnt yyhhoi/openpose:u16cuda9dnn7-2 bash
# 2. # cp -r /mnt/data/hoi/gait_analysis/scripts/openpose_shellscripts/generate_openpose_shellscript_for_FSF.py /mnt/data/hoi/gait_analysis/scripts/common ./
# 3. # python generate_openpose_shellscript_for_FSF.py
# 4. # sh openpose_inference_script.sh | tee /mnt/data/hoi/gait_analysis/scripts/openpose_shellscripts/clt_output.txt
# Configuration of behaviours is stored in the generate_openpose_shellscript_for_FSF.py above

# Environment $ nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=0 -v /:/mnt yyhhoi/neuro:1 bash
# $ NV_GPU=0,1,2 nvidia-docker run --rm -it -v /:/mnt yyhhoi/openpose:u16cuda9dnn7-2 bash
from openpose_shellscripts.concatenate_generate_split import concat_vids_from_dir
vids_dir = "/mnt/data/gait/data/videos_mp4/"
output_vid_path = "/mnt/data/hoi/gait_analysis/data/grand_video.mp4"
output_df_path = "/mnt/data/hoi/gait_analysis/data/grand_video_info.csv"
skip_vids_dir = "/mnt/data/hoi/gait_analysis/data/openpose_visualisation"
concat_vids_from_dir(vids_dir=vids_dir,
                     output_vid_path=output_vid_path,
                     output_df_path=output_df_path,
                     skip_vids_dir=skip_vids_dir,
                     num_to_concat=10000)



# %% ======================== Step 2: Keypoints Pre-processing =======================
# Environment $ nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=0 -v /:/mnt yyhhoi/neuro:1 bash
# from common.preprocess import openpose_preprocess_wrapper
# src_vid_dir = "/mnt/data/gait/data/videos_mp4/"
# input_data_main_dir = "/mnt/data/hoi/gait_analysis/data/openpose_keypoints"
# output_vid_dir = "/mnt/data/hoi/gait_analysis/data/preprocessed_visualisation"
# output_data_dir = "/mnt/data/hoi/gait_analysis/data/preprocessed_keypoints"
# error_log_path = "/mnt/data/hoi/gait_analysis/logs/preprocess_error_log.txt"
# openpose_preprocess_wrapper(src_vid_dir, input_data_main_dir, output_vid_dir, output_data_dir, error_log_path,
#                             write_video=False,
#                             plot_keypoints=False)

# %% ======================== Step 3: Extracting feature for ODE =======================
# Environment $ nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=0 -v /data/hoi/gait_analysis:/mnt yyhhoi/neuro:1 bash
# from common.feature_extraction import FeatureExtractorForODE
# scr_keyps_dir = "/mnt/data/preprocessed_keypoints"
# labels_path = "/mnt/data/labels/z_matrix/df_gait_vid_linked_190718.pkl"
# df_save_path = "/mnt/data/feas_incompleteLabels_nanMasks.pickle"
# minimum_sequence_window = 128
# extractor = FeatureExtractorForODE(scr_keyps_dir=scr_keyps_dir,
#                                    labels_path=labels_path,
#                                    df_save_path=df_save_path)
# extractor.extract(minimum_sequence_window)

# %% ======================== Step A.A.4: Train and visualize on temporal_VAE =======================
# # Environment $ nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=0 -v /data/hoi/gait_analysis:/mnt yyhhoi/neuro:1 bash
# from temporal_vae_script import run_train_and_vis_on_tvae
# run_train_and_vis_on_tvae()

# %% ======================== Step A.B.4: Train and visualize on single_skeleton_VAE =======================
# Environment $ nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=0 -v /data/hoi/gait_analysis:/mnt yyhhoi/neuro:1 bash
# from single_skeleton_vae_script import run_train_and_vis_on_ssvae
# run_train_and_vis_on_ssvae()

# %% ======================== Step A.C.4: Train and visualize on combined_VAE =======================
# Environment $ nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=0 -v /data/hoi/gait_analysis:/mnt yyhhoi/neuro:1 bash
# from spatiotemporal_vae_script import run_train_and_vis_on_stvae
# run_train_and_vis_on_stvae()

# %% ======================== (Defunkt) Step A.D.4: Train on neural ODE =======================
# Environment $ nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=0 -v /data/hoi/gait_analysis:/mnt yyhhoi/neuro:1 bash
# model_path = "neuralODE/gait_ODE_chkpt/ckpt.pth"
# data_gen = GaitGeneratorFromDF("/mnt/data/raw_features_zmatrix_row_labels.pickle",
#                                m=512)
# gait_neural_ode_train(data_gen)

# %% ======================== (Defunkt) Step A.D.4: Visualise =======================
# data_gen = GaitGeneratorFromDF("/mnt/data/raw_features_zmatrix_row_labels.pickle",
#                                m=512)
# model_path = "neuralODE/gait_ODE_chkpt/ckpt.pth"
# gait_neural_ode_vis(model_path, "neuralODE/gait_vis_results", data_gen)
