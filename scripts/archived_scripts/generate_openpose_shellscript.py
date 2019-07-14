import os
from glob import glob


def gen_template(src_videos_dir, output_videos_dir, output_data_dir):
    move_directory = "cd /opt/openpose/\n"
    command_template = "./build/examples/openpose/openpose.bin --video {} --write_video {} --write_json {} --num_gpu 2 --num_gpu_start 0 --display 0\n"
    # command_template = "./build/examples/openpose/openpose.bin --video {} --write_json {} --display 0 --render_pose 0\n"
    whole_command = move_directory

    all_videos = glob(os.path.join(src_videos_dir, "*"))
    for input_vid_path in all_videos:
        vid_name_root = os.path.splitext(os.path.split(input_vid_path)[1])[0]
        output_vid_path = os.path.join(output_videos_dir, vid_name_root + ".mp4")
        output_data_path = os.path.join(output_data_dir, vid_name_root )

        output_folder_path = os.path.join(output_videos_dir, vid_name_root)
        if os.path.isdir(output_data_path) is not True:
            os.mkdir(output_data_path)
        whole_command += command_template.format(input_vid_path, output_vid_path, output_data_path)
        # whole_command += command_template.format(input_vid_path, output_data_path)
    with open("openpose_inference_script.sh", "w") as fh:
        fh.write(whole_command)

if __name__ == "__main__":
    src_videos_dir = "/mnt/pose_inference_pilot/videos_data"
    output_videos_dir = "/mnt/pose_inference_pilot/2D_visualisation/openpose"
    output_data_dir = "/mnt/pose_inference_pilot/2D_keypoints/openpose"

    gen_template(src_videos_dir, output_videos_dir, output_data_dir)







