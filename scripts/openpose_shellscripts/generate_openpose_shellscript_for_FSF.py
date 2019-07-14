import os
import sys

sys.path.append("../")

from common.utils import sample_subset_of_videos


def gen_template(all_videos_list, output_videos_dir, output_data_dir):
    """
    Produce bash shell commands for inferring multiple videos with openpose

    Args:
        src_videos_dir: Directory of the videos that you infer
        output_videos_dir: Directory for storing the visualisation (keypoints overlaid on original video)
        output_data_dir: Directory for storing the keypoints data (.json)
        division: List of indexes that divide your inference list to multiple subsets. It is intended for parallel inference.

    """

    all_videos = sorted(all_videos_list)
    move_directory = "cd /opt/openpose/\n"
    command_template = "./build/examples/openpose/openpose.bin --video {} --write_video {} --write_json {} --display 0 --logging_level 4\n"
    whole_command = move_directory
    skipped_num = 0
    for idx, input_vid_path in enumerate(all_videos):

        vid_name_root = os.path.splitext(os.path.split(input_vid_path)[1])[0]
        output_vid_path = os.path.join(output_videos_dir, vid_name_root + ".mp4")

        # If output exists, skipp
        if os.path.isfile(output_vid_path):
            print("{} exists. Skipped".format(output_vid_path))
            skipped_num += 1
            continue

        # Make folder for storing keypoints 
        output_data_path = os.path.join(output_data_dir, vid_name_root)  # output_data_path should point to a directory!
        if os.path.isdir(output_data_path) is not True:
            os.mkdir(output_data_path)

        # Append to shell script
        whole_command += command_template.format(input_vid_path, output_vid_path, output_data_path)
        whole_command += "NUM=$(expr $NUM + 1)\n"
        whole_command += 'echo "$NUM videos"\n'

    with open("openpose_inference_script.sh", "w") as fh:
        fh.write("NUM=0\n")
        fh.write(whole_command)

    print("Progress: {}/{}".format(skipped_num, len(all_videos_list)))


if __name__ == "__main__":
    # src_videos_dir = "/mnt/data/gait/data/videos_mp4/"
    src_videos_dir = "/mnt/media/dsgz2tb_2/videos_converted/"
    # labels_path = "/mnt/data/hoi/gait_analysis/data/labels/z_matrix/df_gait_vid_linked_190718.pkl"
    all_videos_list = sample_subset_of_videos(src_videos_dir, sample_num=0, labels_path="", seed=50,
                                              with_labels=False)

    output_videos_dir = "/mnt/data/hoi/gait_analysis/data/openpose_visualisation"
    output_data_dir = "/mnt/data/hoi/gait_analysis/data/openpose_keypoints"
    gen_template(all_videos_list, output_videos_dir, output_data_dir)
    print("Number of video = {}".format(all_videos_list.shape[0]))
