from common.utils import LabelsReader, fullfile
from glob import glob
import pandas as pd
import os
import numpy as np
import pdb
import shutil

## conversion

labels_path = "/mnt/data/df_gait_vid_linked_190718.pkl"
extracted_features_dir = "/mnt/simple_shallow_features_analysis/data/extracted_features"

lreader = LabelsReader(labels_path)
df = lreader.loaded_df

all_vid_paths = glob(os.path.join(extracted_features_dir, "*"))
all_vid_base_names = []
for vid_path in all_vid_paths:
    vid_base_name = fullfile(vid_path)[0]
    all_vid_base_names.append(vid_base_name.replace(".npy", ".mp4"))

vid_name_exists = np.zeros(df.shape[0])
df["vid_name_exists"] = vid_name_exists

for i in range(df.shape[0]):
    vid_name_df = str(df.iloc[i,7]).replace("\n", "")
        
    if vid_name_df in all_vid_base_names:
        df.iloc[i,8] = 1
# pdb.set_trace()
df_filtered = df[df['vid_name_exists']== 1].reset_index(drop=True)
print(df_filtered.shape)

vec_list = []
for i in range(df_filtered.shape[0]):
    print(i)
    vid_name_df = str(df_filtered.iloc[i,7]).replace(".mp4\n", "")
    # pdb.set_trace()
    vid_name_np = vid_name_df + ".npy"
    feature_vec = np.load(os.path.join(extracted_features_dir, vid_name_np))
    vec_list.append(feature_vec)

df_filtered["feature_vector"] = vec_list

df_filtered = df_filtered.drop(columns=["vid_name_exists"])
print(df_filtered['feature_vector'])
print(df_filtered.shape)
print(df_filtered.columns)
save_path = "/mnt/simple_shallow_features_analysis/data/extracted_features_1993_with_meta_info.pickle"
df_filtered.to_pickle(save_path)
    
# lreader = LabelsReader("/mnt/simple_shallow_features_analysis/data/extracted_features_1993_with_meta_info.pickle")
# df = lreader.loaded_df
# print(df['feature_vector'])
# print(df.shape)
# print(df.columns)

# src_dir = "/mnt/data/hoi/MeetingMinutes/2019-03-27_Gait_1st_Attempt_on_Shallow_Features/compare_detectron_openpose"

# search_dir = "/mnt/data/hoi/gait_analysis/simple_shallow_features_analysis/data/preprocessed_visualisation"

# all_src_vid_paths = glob(os.path.join(src_dir, "*.mp4"))

# for vid_path in all_src_vid_paths:
#     vid_basename = fullfile(vid_path)[0]

#     to_copy_path = os.path.join(search_dir, vid_basename)
#     if os.path.isfile(to_copy_path):
#         shutil.copy2(vid_path, to_copy_path)
#     else:
#         print("{} does not exist".format(to_copy_path))


