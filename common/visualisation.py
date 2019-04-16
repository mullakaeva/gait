from .utils import read_preprocessed_keypoints, fullfile
from glob import glob
import numpy as np
import skvideo.io as skv
import skimage.io as ski
from skimage.color import rgba2rgb
import matplotlib.pyplot as plt
import os


# OP = OpenPose, DE = Detectron
class Visualiser2D_OP_DE():
    def __init__(self, preprocessed_2D_OP, preprocessed_2D_DE, input_video_path):
        self.op_keyps = read_preprocessed_keypoints(preprocessed_2D_OP) # (m, 17, 3), m = num of frames
        self.de_keyps = read_preprocessed_keypoints(preprocessed_2D_DE) # (m, 17, 3)
        self.vreader = skv.FFmpegReader(input_video_path)
        self.m, self.h, self.w, _ = self.vreader.getShape()
        try:
            assert self.de_keyps.shape[0] == self.op_keyps.shape[0]
            assert self.m == self.op_keyps.shape[0]
        except:
            import pdb
            pdb.set_trace()
    def draw(self, output_video_path):
        self.vwriter = skv.FFmpegWriter(output_video_path)

        for idx, vid_frame in enumerate(self.vreader.nextFrame()):
            fig, ax = plt.subplots()
            ax.imshow(vid_frame)
            ax.scatter(self.op_keyps[idx, :, 0], self.op_keyps[idx, :, 1], marker = "x", c="r", alpha=0.7, label = "OpenPose")
            ax.scatter(self.de_keyps[idx, :, 0], self.de_keyps[idx, :, 1], marker = "x", c="g", alpha=0.7, label = "Detectron")
            ax.set_xlim(0, self.w)
            ax.set_ylim(self.h, 0)
            ax.legend()
            fig.tight_layout()
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            self.vwriter.writeFrame(data)
            plt.close()
    def __del__(self):
        self.vreader.close()
        if self.vwriter is not None:
            self.vwriter.close()

def Visualiser2D_OP_DE_wrapper(preprocessed_2D_OP_dir, preprocessed_2D_DE_dir, processed_video_dir, output_video_dir):
    all_vids_paths = glob(os.path.join(processed_video_dir, "*"))
    num_vids = len(all_vids_paths)
    for idx, vid_path in enumerate(all_vids_paths):
        print("{}/{}: {}".format(idx, num_vids, vid_path))
        vid_name_root = fullfile(vid_path)[1][1]
        if vid_name_root == 'demo_video':
            continue
        # if vid_name_root != 'vid0189_4898_20171130':
        #     continue
        preprocessed_2D_OP_path = os.path.join(preprocessed_2D_OP_dir, vid_name_root + ".npz")
        preprocessed_2D_DE_path = os.path.join(preprocessed_2D_DE_dir, vid_name_root + ".npz")
        output_video_path = os.path.join(output_video_dir, vid_name_root + ".mp4")

        preprocessor = Visualiser2D_OP_DE(preprocessed_2D_OP_path, preprocessed_2D_DE_path, vid_path)
        preprocessor.draw(output_video_path)


        
class RawVisualiser2D_OP_DE():
    """
    Visualise 4 videos:
        1. Top-left:    Original gait video
        2. Top-right:   OpenPose's skeleton
        3. Bottom-right:Preprocessed zoomed in video with markers from both Openpose and Detectron
        4. Bottom-left: Detectron's skeleton

    """
    def __init__(self, ori_vid_path, de_2D_images_dir, op_2D_vid_path, preprocessed_vid_path, preprocessed_data, output_vid_path):
        self.ori_vreader = skv.FFmpegReader(ori_vid_path)
        self.op_vreader = skv.FFmpegReader(op_2D_vid_path)
        self.pre_vreader = skv.FFmpegReader(preprocessed_vid_path)
        self.start_idx, self.end_idx = np.load(preprocessed_data)['vid_info'][()]['cut_duration']
        self.vid_name_root = fullfile(ori_vid_path)[1][1]
        self.num_frames = self.ori_vreader.getShape()[0]
        
        _, self.ori_h, self.ori_w, _ = self.ori_vreader.getShape()
        _, self.op_h, self.op_w, _ = self.op_vreader.getShape()
        _, self.pre_h, self.pre_w, _ = self.pre_vreader.getShape()

        self.vwriter = skv.FFmpegWriter(output_vid_path)

        # Detectron stores output as a folder of .png
        self.de_2D_images_dir = de_2D_images_dir

        print("Ori:{}\nOP:{}\nPre:{}\n".format(self.ori_vreader.getShape(),self.op_vreader.getShape(),self.pre_vreader.getShape()))
        
    
    def draw(self):
        # output dimension = (self.num_frames, 464+480, 640*2, 3)
        frame_idx = 0
        
        for ori_frame, op_frame in zip(self.ori_vreader.nextFrame(), self.op_vreader.nextFrame()):
            print("\r{}/{} : {}".format(frame_idx, self.num_frames, self.vid_name_root), flush=True, end="")
            
            # expected_height = max(self.ori_h, self.op_h, self.pre_h) + 480
            # expected_width = max(self.ori_w, self.op_w, self.pre_w) + 640
            # output_frame = np.zeros((expected_height, expected_width, 3))
            output_frame = np.zeros((480+464, 640*2, 3))
            # # Top left
            # # output_frame[0:self.ori_h, 0:self.ori_w,:] = ori_frame
            output_frame[0:464, 0:640,:] = ori_frame
            # # Top right
            output_frame[0:464:, 640:(640+640), : ] = op_frame

            # # Bottom-left
            de_img_path = os.path.join(self.de_2D_images_dir, "%s_%012d.png"%(self.vid_name_root, frame_idx))
            if os.path.isfile(de_img_path):
                de_img = ski.imread(de_img_path)
                de_img_rgb = rgba2rgb(de_img)
                de_img_rgb = np.around(de_img_rgb*255).astype(int)
                de_h, de_w, _ = de_img_rgb.shape
                output_frame[464:464+463, 0:640,:] = de_img_rgb
            # # else:
            # #     de_h, de_w = 463, 640 

            # # Bottom-right
            if (frame_idx >= self.start_idx) and (frame_idx <= self.end_idx):
                for pre_frame in self.pre_vreader.nextFrame():
                    # output_frame[self.op_h:(self.op_h + self.pre_h), de_w:(self.pre_w + de_w),:] = pre_frame
                    output_frame[464:464+480, 640: 640+640,:] = pre_frame
                    break
            
            self.vwriter.writeFrame(output_frame)
            frame_idx += 1

        
        

    def __del__(self):
        self.ori_vreader.close()
        self.op_vreader.close()
        self.pre_vreader.close()
        if self.vwriter is not None:
            self.vwriter.close()

def RawVisualiser2D_OP_DE_wrapper(ori_dir, de_2D_images_main_dir, op_2D_vid_dir,
                                 preprocessed_vid_dir, preprocessed_data_dir, output_vid_dir):

    all_vids_paths = glob(os.path.join(ori_dir, "*"))
    num_vids = len(all_vids_paths)
    for idx, vid_path in enumerate(all_vids_paths):
        print("{}/{}: {}".format(idx, num_vids, vid_path))
        vid_name_root = fullfile(vid_path)[1][1]
        exclusions = ["demo_video", "vid1119_9523_20110207", "vid1258_1935_20110411", "vid0894_6501_20101117"]
        if vid_name_root in exclusions :
            continue
        de_2D_images_subfolder = os.path.join(de_2D_images_main_dir, vid_name_root )
        op_2D_vid_path = os.path.join(op_2D_vid_dir, vid_name_root + ".mp4")
        preprocessed_vid_path = os.path.join(preprocessed_vid_dir, vid_name_root + ".mp4")
        preprocessed_data_path = os.path.join(preprocessed_data_dir, vid_name_root + ".npz")
        output_vid_path = os.path.join(output_vid_dir, vid_name_root + ".mp4")
        if os.path.isfile(output_vid_path):
            print("skipped")
            continue
        viser = RawVisualiser2D_OP_DE(vid_path, de_2D_images_subfolder, op_2D_vid_path, 
                                      preprocessed_vid_path, preprocessed_data_path, output_vid_path)
        viser.draw()
