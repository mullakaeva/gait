def gen_shellscript():
    raw_grand_vid_path = "/mnt/data/hoi/gait_analysis/data/grand_video.mp4"
    op_grand_vid_path = "/mnt/data/hoi/gait_analysis/data/op_grand_video.mp4"
    op_grand_keypts_dir = "/mnt/data/hoi/gait_analysis/data/op_grand_keypts"
    move_directory = "cd /opt/openpose/\n"
    command_template = "./build/examples/openpose/openpose.bin --video {} --write_video {} --write_json {} --display 0\n"
    command_template.format(raw_grand_vid_path, op_grand_vid_path, op_grand_keypts_dir)

    whole_command = move_directory
    whole_command += command_template.format(raw_grand_vid_path, op_grand_vid_path, op_grand_keypts_dir)
    with open("openpose_inference_script.sh", "w") as fh:

        fh.write(whole_command)

if __name__ == "__main__":
    gen_shellscript()