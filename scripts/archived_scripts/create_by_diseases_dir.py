# Environment $ nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=0 -v /:/mnt yyhhoi/neuro:1 bash
import pickle
import os
import shutil

# %% Define paths
project_dir = "/mnt/data/hoi/gait_analysis"
raw_vid_dir = "/mnt/data/gait/data/videos_mp4"
by_dis_dir = os.path.join(project_dir, "data", "by_diseases")
labels_dir = os.path.join(project_dir, "data", "labels")

file_name_diag = os.path.join(labels_dir, "diagnosis", "dfAll_diag_2019.pkl")
file_name_vid = os.path.join(labels_dir, "z_matrix", "df_gait_vid_subject_linked_anon_2018-03.pkl")

# %% Load data
print("Load data")
with open(file_name_diag, "rb") as fh:
    df_diag = pickle.load(fh)
with open(file_name_vid, "rb") as fh:
    df_vid = pickle.load(fh)

# %% Get grt_Pt_id from df_diag
print("Get grt_Pt_id from df_diag")
label_grt = dict()
for diag_idx in range(df_diag.shape[0]):
    label = df_diag["label"][diag_idx]
    grt_Pt_id = int(df_diag["grt_Pt_id"][diag_idx])

    try:
        if grt_Pt_id not in label_grt[label]:
            label_grt[label].append(grt_Pt_id)
    except KeyError:
        label_grt[label] = list()

# %% get vid_filename from grt_Pt_id
print("get vid_filename from grt_Pt_id")
label_vidnames = dict()
for label in label_grt.keys():
    vid_names_each_label = []
    for grd_id in label_grt[label]:
        vid_names = list(df_vid[df_vid["grt_Pt_id"] == grd_id]["vid_filename"])
        vid_names_each_label += vid_names

    vid_names_each_label = list(set(vid_names_each_label))
    label_vidnames[label] = vid_names_each_label

# %% Create directories for each diseases and videos
print("Create directories and copy videos by diseases")
for label in label_vidnames.keys():
    print("Moving {}".format(label))
    disease_dir = os.path.join(by_dis_dir, label)
    os.makedirs(disease_dir, exist_ok=True)
    vid_count = 0
    for vid_name in label_vidnames[label]:
        if vid_count > 5:
            break
        vid_name = vid_name.rstrip()
        print("videos {}".format(vid_name))
        vid_subdir = os.path.join(disease_dir, vid_name)
        os.makedirs(vid_subdir, exist_ok=True)

        try:
            # Raw videos
            shutil.copy2(os.path.join(raw_vid_dir, vid_name), os.path.join(vid_subdir, "raw_{}".format(vid_name)))

            # Openpose
            shutil.copy2(os.path.join(project_dir, "data", "openpose_visualisation", vid_name),
                         os.path.join(vid_subdir, "openpose_{}".format(vid_name)))

            # Preprocessed
            shutil.copy2(os.path.join(project_dir, "data", "preprocessed_visualisation", vid_name),
                         os.path.join(vid_subdir, "preprocessed_{}".format(vid_name)))
            vid_count += 1
        except FileNotFoundError:
            print("vid_name {} not found, skipped".format(vid_name))