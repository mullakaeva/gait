# This script concatenate all videos into single "grand" video, and run openpose inference on it
# By doing so, the re-initilization of openpose problem can be avoided (no re-initialization for each video)
# In details, the video name and their time step will be recorded in a dict/dataframe, to facilitate splitting afterwards
# Overall: (1) Concatenate videos -> (2) Record corresponding time steps and video names -> (3) split the "grand" video and keypoints

import skvideo.io as skv
import os
import pandas as pd
import numpy as np
from glob import glob


def concat_vids_from_dir(vids_dir, output_vid_path, output_df_path, skip_vids_dir=None, intended_shape=(464, 640),
                         num_to_concat=None):
    """
    This function concatenates multiple videos from a directory, to a "grand" video for later processing

    Parameters
    ----------
    vids_dir : str
        Directory where the videos to be concatenate are stored. In default, only .mp4 files are recognised.
    output_vid_path : str
        The output path for the "grand" video
    output_df_path : str
        The output path for the .csv record, containing the start/end index for each video name
    skip_vids_dir : str
        Directory where the previously inferred openpose videos were located. Same video names will be skipped.
    intended_shape : tuple
        (width, height) of the video. If the video's shape does not match this, the video will not be concatenated

    Returns
    -------

    """
    all_vid_paths = sorted(glob(os.path.join(vids_dir, "*.mp4")))
    num_vids = len(all_vid_paths)
    vwriter = skv.FFmpegWriter(output_vid_path)
    vid_infos = []
    grand_vid_idx = 0
    stopping_idx = 0 if num_to_concat is None else num_to_concat
    finished_idx = 0

    for vid_idx, vid_path in enumerate(all_vid_paths):

        # Terminate if the vid_idx reaches the stopping number
        if finished_idx >= stopping_idx:
            print("Stop concatenation")
            break

        # Skip the video if the stored keypoints have been produced
        vid_name_root = os.path.splitext(os.path.basename(vid_path))[0]
        if skip_vids_dir is not None:
            skip_vid_path = os.path.join(skip_vids_dir, vid_name_root + ".mp4")
            if os.path.exists(skip_vid_path):
                print("{} skipped".format(vid_idx))
                continue

        # Concatenate and record for each video
        try:
            vreader_each = skv.FFmpegReader(vid_path)
        except ValueError as e:
            print("{} skipped".format(vid_idx))
            print(e)
            continue

        # Get vid's array infos. Skip if shape is not matched
        num_frames, w, h, ch = vreader_each.getShape()
        if (w, h) != intended_shape:
            print("Shape ({}, {}) does not match the intended {}".format(w, h, intended_shape))
            vreader_each.close()
            continue

        # Loop frames in each video and write it to grand video
        vid_start_idx = grand_vid_idx
        for frame_idx, frame in enumerate(vreader_each.nextFrame()):
            print("\r{}/{} vid: {}/{} frames".format(vid_idx, num_vids, frame_idx, num_frames), end="", flush=True)
            grand_vid_idx += 1
            vwriter.writeFrame(frame)
        print()
        vid_end_idx = grand_vid_idx

        # Record information
        vid_infos.append([vid_name_root, vid_start_idx, vid_end_idx])
        vreader_each.close()
        finished_idx += 1

    # Build dataframes
    vid_infos_np = np.array(vid_infos)
    vid_infos_np[:, 1] = vid_infos_np[:, 1].astype(np.int)
    vid_infos_np[:, 2] = vid_infos_np[:, 2].astype(np.int)
    infos_df = pd.DataFrame(vid_infos_np, columns=["vid_name_root", "vid_start_idx", "vid_end_idx"])
    infos_df.to_csv(output_df_path)
    vwriter.close()


def spliting(openpose_grand_vid_path, grand_info_df_path, grand_keypoints_dir, split_openpose_vid_dir,
             split_keypts_dir):
    vreader_openpose = skv.FFmpegReader(openpose_grand_vid_path)
    all_grand_keypts_paths = sorted(glob(os.path.join(grand_keypoints_dir, "*.json")))

    # Ensure the num_frames are the same
    num_frames_op, num_keypts = vreader_openpose.getShape()[0], len(all_grand_keypts_paths)
    if num_frames_op == num_keypts:
        print("Frame number is different across videos and keypoint number")
        raise

    # Load grand dataframe
    info_df = pd.read_csv(grand_info_df_path, index_col=True)

    # Loop the grand video's frames
    vid_idx = 0

    vid_name_root, start_idx, end_idx = info_df.iloc[vid_idx]
    vwriter = skv.FFmpegWriter(os.path.join(split_openpose_vid_dir, vid_name_root + ".mp4"))

    for idx, frame in enumerate(vreader_openpose.nextFrame()):

        if idx < end_idx:
            print("\rWriting current vid ({}) : {} | {} ~ ({}, {})".format(vid_idx, idx, vid_name_root, start_idx,
                                                                           end_idx),
                  end="", flush=True)
            vwriter.writeFrame(frame)
        elif vid_idx != info_df.shape[0]:

            print()

            # End the previous vid
            vwriter.close()
            vid_idx += 1

            # Open the next vid
            vid_name_root, start_idx, end_idx = info_df.iloc[vid_idx]
            vwriter = skv.FFmpegWriter(os.path.join(split_openpose_vid_dir, vid_name_root + ".mp4"))
            print("Open new vid ({}) at {} | {} ~ ({}, {})".format(vid_idx, idx, vid_name_root, start_idx, end_idx))
        else:
            print()
            # End the previous vid
            vwriter.close()
            print("Finished last vid")
