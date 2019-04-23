if __name__ == "__main__":
    pass
# Simple Shallow Features (SSF) analysis. 

#%%  ======================= Step 1: Sample 1000 videos ============================
    # Environment $ nvidia-docker run --rm -it -v /:/mnt yyhhoi/neuro:0 bash
    # from gait_common_code.utils import sample_and_copy_videos
    # number_videos_to_sample = 2000
    # data_dir = "/mnt/data/gait/data/videos_mp4/"
    # dest_dir = "/mnt/data/hoi/gait_analysis/simple_shallow_features_analysis/data/raw_videos/"
    # labels_path = "/mnt/data/hoi/gait_analysis/data/df_gait_vid_linked_190718.pkl"
    # sample_subset_of_videos(data_dir, dest_dir, number_videos_to_sample, labels_path=labels_path, with_labels=True)

#%%  ======================= Step 2: OpenPose inference ============================
    # This section infers the 1000 videos for their (1) 2D pose and (2) visualisation with OpenPose
    # 1. $ nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=0 -v /data/hoi/gait_analysis:/mnt yyhhoi/openpose:u16cuda9dnn7-1 bash
    # 2. # cp /mnt/simple_shallow_features_analysis/generate_openpose_shellscript_for_SSF.py ./
    # 3. # python generate_openpose_shellscript_for_SSF.py
    # 4. # sh openpose_inference_script.sh
    # Configuration of behaviours is stored in the generate_openpose_shellscript_for_SSF.py above
#%% ======================== Step 3: Keypoints PreProcessing =======================
    # Environment $ nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=0 -v /data/hoi/gait_analysis:/mnt yyhhoi/neuro:0 bash
    # src_vid_dir = "/mnt/simple_shallow_features_analysis/data/raw_videos"
    # input_data_main_dir = "/mnt/simple_shallow_features_analysis/data/openpose_keypoints"
    # output_vid_dir = "/mnt/simple_shallow_features_analysis/data/preprocessed_visualisation"
    # output_data_dir = "/mnt/simple_shallow_features_analysis/data/preprocessed_keypoints"
    # divison = [1600, 2000]
    
    # openpose_preprocess_wrapper(src_vid_dir, input_data_main_dir, output_vid_dir, output_data_dir,divison, plot_keypoints=True)

#%% ======================== Step 4: Feature Extraction ============================
    # Environment $ nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=0 -v /data/hoi/gait_analysis:/mnt yyhhoi/neuro:0 bash
    # from gait_common_code.feature_extraction import FeatureExtractor
    # data_dir = "/mnt/simple_shallow_features_analysis/data/preprocessed_keypoints"
    # save_dir = "/mnt/simple_shallow_features_analysis/data/extracted_features"
    # extractor = FeatureExtractor(data_dir, save_dir)
    # extractor.extract()

#%% =============================== Step 5: T-SNE  ===================================
    # $ nvidia-docker run --rm -it -v /data/hoi/gait_analysis:/mnt yyhhoi/neuro:0 bash
    
    # from simple_shallow_features_analysis.tsne_analysis import analyse_with_tsne

    # labels_path = "/mnt/data/df_gait_vid_linked_190718.pkl"
    # extracted_features_dir = "/mnt/simple_shallow_features_analysis/data/extracted_features"
    # vis_main_dir = "/mnt/simple_shallow_features_analysis/tsne_visualisation"
    # available_data_size = 456 
    # n_dims = 666
    # analyse_with_tsne(extracted_features_dir, labels_path, vis_main_dir, available_data_size, n_dims )

#%% =============================== Step 6: Random Forest  ===================================


#%% ======================== Count videos that have labels  ==========================
    # from glob import glob
    # from gait_common_code.utils import LabelsReader, fullfile
    # import numpy as np
    # import os
    # labels_path = "/mnt/data/df_gait_vid_linked_190718.pkl"
    # extracted_features_dir = "/mnt/simple_shallow_features_analysis/data/extracted_features"
    # lreader = LabelsReader(labels_path)
    # all_feature_paths = glob(os.path.join(extracted_features_dir, "*"))
    
    # print(lreader.get_all_filenames()[0:10])
    
    # existed_data = []
    # missing_data = []

    # for path in all_feature_paths:
    #     vid_name = fullfile(path)[1][1] + ".mp4"
    #     try:
    #         lreader.get_label(vid_name)
    #         existed_data.append(vid_name)
    #     except KeyError:
    #         missing_data.append(vid_name)

    # print("All:{}\nExisted:{}\nMissing:{}\n".format(len(all_feature_paths), len(existed_data), len(missing_data)))

        
        