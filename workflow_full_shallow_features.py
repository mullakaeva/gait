from common.preprocess import openpose_preprocess_wrapper

# %%  ======================= Step 2: OpenPose inference ============================
# This section infers the 1000 videos for their (1) 2D pose and (2) visualisation with OpenPose
# 1. $ nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=0 -v /:/mnt yyhhoi/openpose:u16cuda9dnn7-2 bash
# 2. # cp -r /mnt/data/hoi/gait_analysis/scripts/openpose_shellscripts/generate_openpose_shellscript_for_FSF.py /mnt/data/hoi/gait_analysis/scripts/common ./
# 3. # python generate_openpose_shellscript_for_FSF.py
# 4. # sh openpose_inference_script.sh
# Configuration of behaviours is stored in the generate_openpose_shellscript_for_FSF.py above


# %% ======================== Step 3: Keypoints PreProcessing =======================
# Environment $ nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=0 -v /:/mnt yyhhoi/neuro:1 bash
src_vid_dir = "/mnt/data/gait/data/videos_mp4/"
input_data_main_dir = "/mnt/data/hoi/gait_analysis/data/openpose_keypoints"
output_vid_dir = "/mnt/data/hoi/gait_analysis/data/preprocessed_visualisation"
output_data_dir = "/mnt/data/hoi/gait_analysis/data/preprocessed_keypoints"
error_log_path = "/mnt/data/hoi/gait_analysis/logs/preprocess_error_log.txt"
openpose_preprocess_wrapper(src_vid_dir, input_data_main_dir, output_vid_dir, output_data_dir, error_log_path, plot_keypoints=True)
