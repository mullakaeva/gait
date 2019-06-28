import numpy as np
import matplotlib.pyplot as plt
import skvideo.io as skv
import os
from glob import glob
from common.visualisation import plot2arr_skeleton, build_frame_2by2
from common.keypoints_format import openpose_L_indexes, openpose_R_indexes
from common.preprocess import openpose_preprocess_wrapper, read_openpose_keypoints

# Environment $ nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=0 -v /data/hoi/gait_analysis:/mnt yyhhoi/neuro:1 bash

def concat_vids(with_re, without_re, vid_out):
    vreader_with = skv.FFmpegReader(with_re)
    vreader_without = skv.FFmpegReader(without_re)
    vwriter = skv.FFmpegWriter(vid_out)
    for frame_with, frame_without in zip(vreader_with.nextFrame(), vreader_without.nextFrame()):
        output_frame = np.concatenate([frame_with, frame_without], axis=1)
        vwriter.writeFrame(output_frame)
    vreader_with.close()
    vreader_without.close()
    vwriter.close()




if __name__ == "__main__":
    basedir = "flipping_solution/preprocessed_vids/"
    with1 = os.path.join(basedir, "with1.mp4")
    without1 = os.path.join(basedir, "without1.mp4")
    compare1 = os.path.join(basedir, "compare1.mp4")
    with2 = os.path.join(basedir, "with2.mp4")
    without2 = os.path.join(basedir, "without2.mp4")
    compare2 = os.path.join(basedir, "compare2.mp4")
    concat_vids(with1, without1, compare1)
    concat_vids(with2, without2, compare2)
    # raw_dir = "flipping_solution/raw_vids/"
    # keypoints_dir = "flipping_solution/openpose_keypts"
    # output_vid_dir = "flipping_solution/preprocessed_vids"
    # output_keypts_dir = "flipping_solution/preprocessed_keypts"
    #
    # openpose_preprocess_wrapper(src_vid_dir=raw_dir,
    #                             input_data_main_dir=keypoints_dir,
    #                             output_vid_dir=output_vid_dir,
    #                             output_data_dir=output_keypts_dir,
    #                             error_log_path="flipping_solution/log.txt",
    #                             plot_keypoints=True,
    #                             write_video=True
    #                             )

