import numpy as np
import skvideo.io as skv
import os
import matplotlib.pyplot as plt
import pdb
import re
from .utils import fullfile, read_openpose_keypoints, read_and_select_openpose_keypoints, moving_average, \
    OnlineFilter_scalar, OnlineFilter_np, extract_contagious
from glob import glob
from skimage.transform import resize
from .keypoints_format import openpose2detectron_indexes, openpose_L_indexes, openpose_R_indexes


def reverse_flips(keyps):
    """

    Parameters
    ----------
    keyps : numpy.darray
        (num_frames, num_keypoitns=25, num_fea=2)

    Returns
    -------

    """
    com_l, com_r = get_com(keyps)  # "com" stands for Centre of Mass

    com_l_sign = np.sign(np.mean(np.sign(com_l[:, 0])))
    if com_l_sign == -1:
        com_l_flipped = (com_l > 0)
        com_r_flipped = (com_r < 0)
    else:
        com_l_flipped = (com_l < 0)
        com_r_flipped = (com_r > 0)

    com_flipped = (com_l_flipped == True) & (com_r_flipped == True)


    com_flipped_filtered, _ = extract_contagious(com_flipped[:, 0], 10)
    com_flipped_filtered = com_flipped_filtered==1

    keypts_corrected = keyps.copy()
    keypts_flipped = keyps[com_flipped_filtered, :].copy()
    keypts_template = keyps[com_flipped_filtered, :].copy()
    keypts_flipped[:, openpose_R_indexes] = keypts_template[:, openpose_L_indexes]
    keypts_flipped[:, openpose_L_indexes] = keypts_template[:, openpose_R_indexes]
    keypts_corrected[com_flipped_filtered, :] = keypts_flipped
    return keypts_corrected, (com_l, com_r, com_l_flipped, com_r_flipped, com_flipped)


def get_com(keypts_all):
    """

    Parameters
    ----------
    keypts_all : numpy.darray
        (num_frames, num_keypoitns=25, num_fea=2)

    Returns
    -------
    com_l, com_r : numpy.darray
        (num_frames, num_fea=2), centre of mass of left/right shoulder
    """
    keypts_l = keypts_all[:, [5], :] - keypts_all[:, [8], :]
    keypts_r = keypts_all[:, [2], :] - keypts_all[:, [8], :]
    com_l = np.mean(keypts_l, axis=1)
    com_r = np.mean(keypts_r, axis=1)
    return com_l, com_r


def save_data_for_VideoPose3D(save_path, video_size, cut_duration, keypoints_list, records=None):
    """
    Args:
        video_size (tuple): Size of the video (width, height)

        save_path (str): Path for saving the the information

        cut_duration (tuple): (start_index, end_index) between start and end is the duration of the preprocessed video

        keypoints_list (list): [keypoints_1, keypoints_2, ..., keypoints_k] for video with k frames. 
                                keypoints_k ~ ndarray with shape (17, 3)

        records (list): Optional. For saving records of how to transform keypoints to the new coordinate system, after cropping the video.
                        List of dictionarys [record_1, record_2, ..., record_k] for video with k frames.
                        record_k = {'translation': X, # X = ndarray with shape (2, 1)
                                    'resizing': Y, # Y = float
                                    'bbox': Z } # Z = ndarray with shape (5, )                  
    Returns:
        None
    """

    keypoints_list_np = np.asarray(keypoints_list)
    keypoints_list_np = keypoints_list_np.reshape(1, keypoints_list_np.shape[0], keypoints_list_np.shape[1],
                                                  keypoints_list_np.shape[2])
    dictionarry_keypoints = {'S1': {'Directions 1': keypoints_list_np}}
    metadata = {'layout_name': 'h36m', 'num_joints': 17,
                'keypoints_symmetry': [[4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]]}
    vid_info = {'video_shape': video_size, 'cut_duration': cut_duration}
    np.savez(save_path, metadata=metadata, positions_2d=dictionarry_keypoints, vid_info=vid_info,
             preprocess_info=records)


def save_data_openpose(save_path, video_size, cut_duration, keypoints_list, records=None):
    """
    Args:
        video_size (tuple): Size of the video (width, height)

        save_path (str): Path for saving the the information

        cut_duration (tuple): (start_index, end_index) between start and end is the duration of the preprocessed video compared to original

        keypoints_list (list): [keypoints_1, keypoints_2, ..., keypoints_k] for video with k frames. 
                                keypoints_k ~ ndarray with shape (17, 3)

        records (list): Optional. For saving records of how to transform keypoints to the new coordinate system, after cropping the video.
                        List of dictionarys [record_1, record_2, ..., record_k] for video with k frames.
                        record_k = {'translation': X, # X = ndarray with shape (2, 1)
                                    'resizing': Y, # Y = float
                                    'bbox': Z } # Z = ndarray with shape (5, )                  
    Returns:
        None
    """

    keypoints_list_np = np.asarray(keypoints_list)  # ndarray with shape (num_frames, 25, 3)
    metadata = {'layout_name': 'body_25', 'num_joints': 25,
                'keypoints_symmetry': [[12, 13, 14, 5, 6, 7], [9, 10, 11, 2, 3, 4]]}
    vid_info = {'video_shape': video_size, 'cut_duration': cut_duration}
    np.savez(save_path, metadata=metadata, positions_2d=keypoints_list_np, vid_info=vid_info, preprocess_info=records)


# Currently not in use
def clip_to_video_size(keypoints_xy, cut_video_size):
    keypoints_x = keypoints_xy[0, :].copy()
    keypoints_x_mean = np.mean(keypoints_x[(keypoints_x > 0) & (keypoints_x < cut_video_size[1])])
    keypoints_x[(keypoints_x < 0)] = keypoints_x_mean
    keypoints_x[(keypoints_x > cut_video_size[1])] = keypoints_x_mean
    keypoints_y = keypoints_xy[1, :].copy()
    keypoints_y_mean = np.mean(keypoints_y[(keypoints_y > 0) & (keypoints_y < cut_video_size[0])])
    keypoints_y[(keypoints_y < 0)] = keypoints_y_mean
    keypoints_y[(keypoints_y > cut_video_size[0])] = keypoints_y_mean
    keypoints_xy[0, :] = keypoints_x
    keypoints_xy[1, :] = keypoints_y
    return keypoints_xy


def find_torso_length_from_keyps(keypoints_xy_input, padding):
    """
    Find the length of the bounding box which is longer than both the 2D dimension spanned by the keypoints
    In body_25 keypoint scheme, index 1, 8, 11, 14 represent neck, hip_centre, right and left ankle

    Parameters
    ==========
    keypoints_xy_input : numpy.darray
        (25, 3)
    padding : int
    """
    keypoints_xy = keypoints_xy_input.copy()
    keypoints_xy_masked = np.ma.array(keypoints_xy, mask=False)
    keypoints_xy_masked.mask[keypoints_xy[:, 2] == 0, :] = True

    ears_centre_xy = np.mean(keypoints_xy_masked[[17, 18], 0:2], axis=0)
    neck_xy = keypoints_xy[1, 0:2]
    hip_centre_xy = keypoints_xy[8, 0:2]

    ankle_centre_xy = np.mean(keypoints_xy[[11, 14], 0:2], axis=0)

    torso_length = np.linalg.norm(ears_centre_xy - neck_xy) + np.linalg.norm(neck_xy - hip_centre_xy) + np.linalg.norm(
        hip_centre_xy - ankle_centre_xy)

    box_half_length = (torso_length / 2) * 1.3 + padding

    return torso_length, box_half_length, [ears_centre_xy, neck_xy, hip_centre_xy, ankle_centre_xy]


class VideoManager():
    def __init__(self, input_video_path, output_video_path):
        self.vreader = skv.FFmpegReader(input_video_path)
        self.vwriter = skv.FFmpegWriter(output_video_path)
        self.vid_name = fullfile(input_video_path)[0]
        self.vid_name_root = fullfile(input_video_path)[1][1]
        self.num_frames, self.vid_h, self.vid_w, self.vid_channels = self.vreader.getShape()

    def __del__(self):
        self.vreader.close()
        self.vwriter.close()


class DetectronPreprocessor():
    """
    This class is used for running preprocessing on Detectron's output, including tasks:
        1. Select bounding box in the right of the frame
        2. Filter out frames with partial keypoints
        3. Clipping and resize the video to the selected bounding box, 
           s.t. reducing human changing size effect in 3D estimation
    
    """

    def __init__(self, input_vid, detectron_data):
        """
        Args:
            input_vid_path (str): Path to the video you want to process
            detectron_data (str): Path of the produced data from Detectron algorithm
        """
        all_data = np.load(detectron_data, encoding="latin1")[()]
        self.vid_name = os.path.basename(input_vid)
        self.vreader = skv.FFmpegReader(input_vid)
        self.boxes = all_data["boxes"]
        self.keyps = all_data["keyps"]
        self.num_frames, self.vid_h, self.vid_w, _ = self.vreader.getShape()
        self.cut_video_size = (250, 250)  # All frames will be resized to this size
        self.cut_padding = 10
        self.box_length_filter = OnlineFilter_scalar(kernel_size=15)
        self.keypoints_filter = OnlineFilter_np(input_size=(2, 17), kernel_size=3)
        # =============== Attributes initialized for later use ==================

        # 1. In self.initialize()
        self.boxes_selected, self.keyps_selected, self.keyps_confidence, self.bbox_areas = [], [], [], []
        self.boxes_centres_smooth = None
        # We filter frames with a threshold of the confidence of keyps. Frames with only partial keypoints
        # Assuming the confidence is a monotonously increase/decreaseing evolvement along time, we set a cut-off time point
        # where the frames before or after time point are discarded.
        self.keyps_confidence_threshold = None  # int
        self.start_idx, self.end_idx = 0, self.num_frames - 1

        # 2. In self.preprocess()
        self.vwriter = None
        # ======================================================================

    def initialize(self, keyps_threshold_quantile=0.4, bbox_areas_threshold_quantile=0.9):
        """
        This function should run before preprocessing.
        Internally, this function calculuates the following for the object:
            For each frame:
            1. Right bbox (choose from left and right)
            2. Right keypoints
            3. Confidence of keypoints
            4. Area of the bbox
            5. Index of the bbox (unused here)

        Internally, this function produces the criteria for preprocessing:
            1. the confidence threshold of key points (lower quantile of all confidences)

        """

        # Choose the right bbox and its keyps. Add the results to the list for each frame
        for i in range(self.num_frames):
            if self.boxes[i] is None or self.keyps[i] is None:

                boxes_each = self.boxes[i - 1][1]
                keyps_each = self.keyps[i - 1][1]
            else:
                boxes_each = self.boxes[i][1]
                keyps_each = self.keyps[i][1]

            box_selected, keyps_selected, keyps_confidence, _ = self._return_selected_bbox_keyps(boxes_each, keyps_each)
            self.keyps_selected.append(keyps_selected)
            self.boxes_selected.append(box_selected)
            self.keyps_confidence.append(keyps_confidence)

        # Determine the duration of the video to extract (cutting away frames with partial keypoints)
        # self.keyps_confidence_threshold = np.quantile(self.keyps_confidence, keyps_threshold_quantile)
        # self.keyps_confidence_threshold = np.max(self.keyps_confidence) - 2 * np.std(self.keyps_confidence)
        self.keyps_confidence_threshold = 0
        self._find_extracted_duration()

        # Low-pass filter out the centre vibration (frame-by-frame vibration noise)
        kernel_size = 30
        boxes_np = np.asarray(self.boxes_selected)
        boxes_centres_np = (boxes_np[:, [0, 1]] + boxes_np[:, [2, 3]]) / 2
        boxes_centres_x_smooth = moving_average(boxes_centres_np[:, 0], kernel_size)
        boxes_centres_y_smooth = moving_average(boxes_centres_np[:, 1], kernel_size)
        self.boxes_centres_smooth = np.array([boxes_centres_x_smooth, boxes_centres_y_smooth]).T

        return None

    def preprocess(self, output_vid, output_data_path, plot_mode=False):
        """
        Args:
            output_vid_path (str): Path of the preprocessed video
        """
        self.vwriter = skv.FFmpegWriter(output_vid)
        all_keypoints = []
        all_records = []
        for frame_idx, vid_frame in enumerate(self.vreader.nextFrame()):
            print("\r{}: {}/{}".format(self.vid_name, frame_idx, self.num_frames), flush=True, end="")

            # Skip frames with low keypoints confidence
            if frame_idx < self.start_idx or frame_idx > self.end_idx:
                continue
            # Dealing with frames that are confident
            else:
                output_frame, keypoints_xy_box_frame, records = self._find_video_clipping_area(vid_frame, frame_idx)
                # Temporal filtering to the x,y coordinates, but not the confidence
                only_xy = self.keypoints_filter.add(keypoints_xy_box_frame[[0, 1], :])
                keypoints_xy_box_frame[[0, 1], :] = only_xy

                all_keypoints.append(keypoints_xy_box_frame.T)  # from (3,17) transposed to (17,3)
                all_records.append(records)

                if plot_mode == False:
                    self.vwriter.writeFrame(output_frame)
                else:
                    # In plot mode, keypoints_xy_box_frame are plotted. 
                    fig, ax = plt.subplots()
                    ax.imshow(output_frame)
                    ax.scatter(keypoints_xy_box_frame[0, :], keypoints_xy_box_frame[1, :])
                    ax.set_xlim(0, output_frame.shape[1])
                    ax.set_ylim(output_frame.shape[0], 0)
                    fig.tight_layout()
                    fig.canvas.draw()
                    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    self.vwriter.writeFrame(data)
                    plt.close()
        cut_duration = (self.start_idx, self.end_idx)
        save_data_for_VideoPose3D(output_data_path, self.cut_video_size, cut_duration, all_keypoints, all_records)

    def __del__(self):
        self.vreader.close()
        if self.vwriter is not None:
            self.vwriter.close()

    def _find_extracted_duration(self):

        start_idx_found, end_idx_found = False, False
        self.keyps_confidence = moving_average(np.asarray(self.keyps_confidence), 5)
        for i in range(self.num_frames - 1):
            reverse_i = self.num_frames - 1 - i
            current_confidence_forward = self.keyps_confidence[i]
            current_confidence_backward = self.keyps_confidence[reverse_i]
            if (current_confidence_forward > self.keyps_confidence_threshold) and (start_idx_found == False):
                self.start_idx = i
                start_idx_found = True
            if (current_confidence_backward > self.keyps_confidence_threshold) and (end_idx_found == False):
                self.end_idx = reverse_i
                end_idx_found = True

        return None

    def _find_video_clipping_area(self, vid_frame, frame_idx):

        output_frame = vid_frame.copy()
        transformation_records = dict()
        # Find the square box length
        keypoints_xy = self.keyps_selected[frame_idx][[0, 1, 3], :]  # keypoints_xy has shape (3, 17) now
        max_xy, min_xy = np.max(keypoints_xy[[0, 1], :], axis=1), np.min(keypoints_xy[[0, 1], :], axis=1)
        w, h = np.abs(max_xy - min_xy)
        box_length_half = w / 2 + self.cut_padding
        if h > w:
            box_length_half = h / 2 + self.cut_padding

        box_length_half = self.box_length_filter.add(box_length_half)  # Temporal filtering

        # Retrieve smoothened bounding box centre and its boundary
        centre_x, centre_y = self.boxes_centres_smooth[frame_idx]
        bbox_boundary = np.around(np.array([
            centre_x - box_length_half, centre_y - box_length_half,
            centre_x + box_length_half, centre_y + box_length_half
        ])).astype(int)

        # Translation of keypoints
        keypoints_xy[[0, 1], :] = keypoints_xy[[0, 1], :] - bbox_boundary[0:2].reshape(2, 1)
        transformation_records["translation"] = -bbox_boundary[0:2].reshape(2, 1)

        # Bbox cannot cross the frame boundary
        if bbox_boundary[0] < 0:
            bbox_boundary[0] -= bbox_boundary[0]
            keypoints_xy[0] -= bbox_boundary[0]
            transformation_records["translation"][0] -= bbox_boundary[0]
        if bbox_boundary[1] < 0:
            bbox_boundary[1] -= bbox_boundary[1]
            keypoints_xy[1] -= bbox_boundary[1]
            transformation_records["translation"][1] -= bbox_boundary[1]
        if bbox_boundary[2] > self.vid_w:
            diff = bbox_boundary[2] - self.vid_w
            bbox_boundary[2] -= diff
            keypoints_xy[0] -= diff
            transformation_records["translation"][0] -= diff
        if bbox_boundary[3] > self.vid_h:
            diff = bbox_boundary[3] - self.vid_h
            bbox_boundary[3] -= diff
            keypoints_xy[1] -= diff
            transformation_records["translation"][1] -= diff

        # Resizing to a constant frame size
        old_width = bbox_boundary[2] - bbox_boundary[0]
        new_width = self.cut_video_size[0]
        resizing_factor = new_width / old_width
        output_frame = output_frame[bbox_boundary[1]:bbox_boundary[3], bbox_boundary[0]:bbox_boundary[2]]
        output_frame = resize(output_frame, self.cut_video_size, anti_aliasing=True)
        keypoints_xy[[0, 1], :] = keypoints_xy[[0, 1], :] * resizing_factor
        transformation_records["resizing"] = resizing_factor
        transformation_records["bbox"] = bbox_boundary

        # set out-of-boundary data to the mean
        # keypoints_xy = clip_to_video_size(keypoints_xy, self.cut_video_size)
        keypoints_xy = np.clip(keypoints_xy, 0, self.cut_video_size[0])

        output_frame = np.around(output_frame * 255).astype(int)
        return output_frame, keypoints_xy, transformation_records

    @staticmethod
    def _return_selected_bbox_keyps(boxes_each, keyps_each):

        idx_sorted = np.argsort(boxes_each[:, 4])[::-1]
        num_boxes = len(idx_sorted)

        def _return_info_by_index(idx):
            box_selected = boxes_each[idx, :]  # ndarray ~ [5,]
            keyps_selected = keyps_each[idx]
            keyps_all_confidence = keyps_selected[3, :]
            keyps_all_confidence[
                keyps_all_confidence < 0.01] = -10  # Penalize the confidence of frames with incomplete keypoints confidence
            keyps_confidence = np.mean(keyps_all_confidence)
            return box_selected, keyps_selected, keyps_confidence, idx

        # If there is only one box, then there is no ambiguity
        if num_boxes == 1:
            return _return_info_by_index(0)

        # If there is more than one box, search for the one with the highest x-coordinate
        elif num_boxes > 1:
            all_mean_x = []
            for j in range(num_boxes):
                each_box, each_keyps, each_keyps_confidence, _ = _return_info_by_index(j)
                each_mean_x, _ = np.mean(each_keyps[[0, 1], :], axis=1)
                # bbox confidence thresholding
                if each_box[4] < 0.2:
                    all_mean_x.append(-999)
                else:
                    all_mean_x.append(each_mean_x)
            max_idx = np.argmax(all_mean_x)
            return _return_info_by_index(max_idx)


class OpenposePreprocessor(VideoManager):
    def __init__(self, input_video_path, openpose_data_each_video_dir, output_video_path, output_data_path):
        self.op_data_each_video_dir = openpose_data_each_video_dir
        self.output_data_path = output_data_path
        self.all_keyps_dicts, self.all_num_people = [], []
        self.keyps_confidence_threshold = -0.2
        self.cut_padding = 0
        self.cut_video_size = (250, 250)
        self.box_length_filter = OnlineFilter_scalar(kernel_size=15)
        self.keypoints_filter = OnlineFilter_np(input_size=(25, 2), kernel_size=3)
        super(OpenposePreprocessor, self).__init__(input_video_path, output_video_path)

    def initialize(self):
        """
        This function does the necessary steps before processing the data frame by frame:
            1. Check if the number of frames in video match the number of .json files
            2. Find the duration where the whole skeleton is visible, by keypoints' confidence
        Keypoints' confidence is calculated by counting the number of keypoints with <0.01 confidence

        """

        self.all_kepys_data_paths = sorted(glob(os.path.join(self.op_data_each_video_dir, "*.json")))
        assert self.num_frames == len(self.all_kepys_data_paths)

        selected_keyps_list = []
        keyps_confidence_list = []
        selected_centres_list = []

        for keyps_path in self.all_kepys_data_paths:
            # Read data and store them into a list, which can be re-used in other sections
            keyps_dict, num_people, _ = read_openpose_keypoints(keyps_path)
            selected_centre, selected_keypoints, keyps_confidence = self._return_selected_keypoints(keyps_dict,
                                                                                                    num_people)
            selected_keyps_list.append(selected_keypoints)
            keyps_confidence_list.append(keyps_confidence)
            selected_centres_list.append(selected_centre)

        self.selected_keyps = np.stack(selected_keyps_list)  # (num_frames, 25, 3)
        self.selected_keyps[:, :, 0:2], _ = reverse_flips(self.selected_keyps[:, :, 0:2])
        self.keyps_confidences = np.stack(keyps_confidence_list)  # (num_frames,)
        self.selected_centres = np.stack(selected_centres_list)  # (num_frames, 3)

        # Low-pass filter out the centre vibration (frame-by-frame vibration noise)
        kernel_size = 30
        selected_centres_x_smooth = moving_average(self.selected_centres[:, 0], kernel_size)
        selected_centres_y_smooth = moving_average(self.selected_centres[:, 1], kernel_size)

        self.selected_centres_smooth = np.stack(
            [selected_centres_x_smooth, selected_centres_y_smooth]).T  # (num_frames, 2)

        self._find_extracted_duration()

    def preprocess(self, write_video=True, plot_keypoints=False):
        all_keypoints = []
        all_records = []
        for frame_idx, vid_frame in enumerate(self.vreader.nextFrame()):
            print("\r{}: {}/{}".format(self.vid_name, frame_idx, self.num_frames), flush=True, end="")

            # Skip frames with low keypoints confidence
            if frame_idx < self.start_idx or frame_idx > self.end_idx:
                continue
            # Dealing with frames that are confident
            else:
                output_frame, keypoints_xy_box_frame, records, box_half_length, central_points = self._find_video_clipping_area(
                    vid_frame, frame_idx)

                # Temporal filtering to the x,y coordinates, but not the confidence
                only_xy = self.keypoints_filter.add(keypoints_xy_box_frame[:, [0, 1]])
                keypoints_xy_box_frame[:, [0, 1]] = only_xy
                all_keypoints.append(keypoints_xy_box_frame)
                all_records.append(records)

                if plot_keypoints:
                    new_torso_length, _, _ = find_torso_length_from_keyps(keypoints_xy_box_frame, 0)
                    fig, ax = plt.subplots()
                    ax.imshow(output_frame)
                    ax.scatter(only_xy[openpose_L_indexes, 0], only_xy[openpose_L_indexes, 1], marker="x", c="r")
                    ax.scatter(only_xy[openpose_R_indexes, 0], only_xy[openpose_R_indexes, 1], marker="x", c="b")
                    # central_line = np.vstack(central_points)
                    # ax.plot(central_line[:, 0], central_line[:, 1])
                    fig.suptitle("Torso length:\n%0.2f" % (new_torso_length))
                    fig.canvas.draw()
                    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    self.vwriter.writeFrame(data)
                    plt.close()
                elif write_video:
                    self.vwriter.writeFrame(output_frame)

        cut_duration = (self.start_idx, self.end_idx)
        save_data_openpose(self.output_data_path, self.cut_video_size, cut_duration, all_keypoints, all_records)

    def _find_extracted_duration(self):
        start_idx_found, end_idx_found = False, False
        self.keyps_confidence_np_filtered = moving_average(self.keyps_confidences, 5)

        for i in range(self.num_frames - 1):
            reverse_i = self.num_frames - 1 - i
            current_confidence_forward = self.keyps_confidence_np_filtered[i]
            current_confidence_backward = self.keyps_confidence_np_filtered[reverse_i]
            if (current_confidence_forward > self.keyps_confidence_threshold) and (start_idx_found == False):
                self.start_idx = i
                start_idx_found = True
            if (current_confidence_backward > self.keyps_confidence_threshold) and (end_idx_found == False):
                self.end_idx = reverse_i
                end_idx_found = True
        return None

    def _return_selected_keypoints(self, keyps_dict, num_people):

        if num_people == 0:
            keyps_selected = np.zeros((25, 3))
            keyps_confidence = -25
            keyps_centre = np.array([0, 0, 0])
            return keyps_centre, keyps_selected, keyps_confidence

        elif num_people == 1:
            keyps_selected = np.asarray(keyps_dict['people'][0]['pose_keypoints_2d']).reshape(25, 3)
            keyps_centre = np.mean(keyps_selected[(keyps_selected[:, 2] > 0.05), :], axis=0)  # (3, )
            keyps_confidence = self.calc_confidence(keyps_selected)
        elif num_people > 1:
            max_x = 0
            max_index = 0
            # Find the person in the most right across all candidates
            for j in range(num_people):
                keyps_np = np.asarray(keyps_dict['people'][j]['pose_keypoints_2d']).reshape(25, 3)
                keyps_np_mask = np.ma.array(keyps_np, mask=False)
                filter_indexes = np.where(keyps_np[:, 2] < 0.005)  # We don't want to include low-confidence keypoints
                keyps_np_mask.mask[filter_indexes[0], :] = True
                mean_x = keyps_np_mask[:, 0].mean()
                if mean_x > max_x:
                    max_x = mean_x
                    max_index = j
            keyps_selected = np.asarray(keyps_dict['people'][max_index]['pose_keypoints_2d']).reshape(25, 3)
            keyps_centre = np.mean(keyps_selected[(keyps_selected[:, 2] > 0.05), :], axis=0)  # (3, )
            keyps_confidence = self.calc_confidence(keyps_selected)

        return keyps_centre, keyps_selected, keyps_confidence

    def _find_video_clipping_area(self, vid_frame, frame_idx):

        output_frame = vid_frame.copy()
        transformation_records = dict()

        # Find the square box length
        keypoints_xy = self.selected_keyps[frame_idx, :, :]  # keypoints_xy (25, 3)
        _, box_length_half, central_points = find_torso_length_from_keyps(keypoints_xy, self.cut_padding)
        if box_length_half == 0:
            box_length_half = self.box_length_filter.get_last()
        else:
            box_length_half = self.box_length_filter.add(box_length_half)  # Temporal filtering

        # Retrieve smoothened bounding box centre and its boundary
        centre_x, centre_y = self.selected_centres_smooth[frame_idx]
        bbox_boundary = np.around(np.array([
            centre_x - box_length_half, centre_y - box_length_half,
            centre_x + box_length_half, centre_y + box_length_half
        ])).astype(int)

        # Bbox cannot cross the frame boundary
        x_diff, y_diff = 0, 0
        if bbox_boundary[0] < 0:
            x_diff -= bbox_boundary[0]

        if bbox_boundary[1] < 0:
            y_diff -= bbox_boundary[1]

        if bbox_boundary[2] > self.vid_w:
            diff = bbox_boundary[2] - self.vid_w
            x_diff -= diff

        if bbox_boundary[3] > self.vid_h:
            diff = bbox_boundary[3] - self.vid_h
            y_diff -= diff

        # Translation of keypoints
        translation_vec = np.array([bbox_boundary[0], bbox_boundary[1]]).reshape(1, 2)
        keypoints_xy[:, [0, 1]] = keypoints_xy[:, [0, 1]] - translation_vec
        bbox_boundary[0] += x_diff
        bbox_boundary[1] += y_diff
        bbox_boundary[2] += x_diff
        bbox_boundary[3] += y_diff

        transformation_records["translation"] = translation_vec

        # Resizing to a constant frame size
        old_width = bbox_boundary[2] - bbox_boundary[0]
        new_width = self.cut_video_size[0]
        resizing_factor = new_width / old_width
        output_frame = output_frame[bbox_boundary[1]:bbox_boundary[3], bbox_boundary[0]:bbox_boundary[2]]
        output_frame = resize(output_frame, self.cut_video_size, anti_aliasing=True)
        keypoints_xy[:, [0, 1]] = keypoints_xy[:, [0, 1]] * resizing_factor
        transformation_records["resizing"] = resizing_factor
        transformation_records["bbox"] = bbox_boundary

        for i in range(4):
            central_points[i] = (central_points[i] - bbox_boundary[0:2]) * resizing_factor

        # set unconfident data to nan
        keypoints_xy[keypoints_xy[:, 2] < 0.1, :] = np.nan
        keypoints_xy[(keypoints_xy[:, 0] < 0) | (keypoints_xy[:, 0] > self.cut_video_size[0]),
        :] = np.nan
        keypoints_xy[(keypoints_xy[:, 1] < 0) | (keypoints_xy[:, 1] > self.cut_video_size[1]),
        :] = np.nan

        output_frame = np.around(output_frame * 255).astype(int)
        return output_frame, keypoints_xy, transformation_records, box_length_half, central_points

    @staticmethod
    def calc_confidence(keyps_selected):

        base_confidence = 0
        keyps_selected_for_confidence = keyps_selected.copy()
        # Nose and eyes are not included for confidence calculation. The maximum of left/right ears are selected to include.
        keyps_selected_for_confidence[[17, 18], 2] = np.max(keyps_selected[[17, 18], 2])
        # Indexes of nose, right and left_eye are 0,15,16 respectively
        keyps_confidence = base_confidence - np.sum(keyps_selected_for_confidence[1:15, 2] < 0.1) - np.sum(
            keyps_selected_for_confidence[17:, 2] < 0.1)

        return keyps_confidence

    def __del__(self):
        super(OpenposePreprocessor, self).__del__()


class OpenposePreprocesser_fromDetectronBox():
    def __init__(self, openpose_data_each_video_dir, detectron_preprocessed_keyps_path):
        self.op_data_each_video_dir = openpose_data_each_video_dir
        self.preprocessed_data = np.load(detectron_preprocessed_keyps_path + ".npz")
        self.de_info = self.preprocessed_data['preprocess_info'][()]
        self.cut_video_size = self.preprocessed_data['vid_info'][()]['video_shape']
        self.start_idx, self.end_idx = self.preprocessed_data['vid_info'][()]['cut_duration']
        self.keypoints_filter = OnlineFilter_np(input_size=(2, 17), kernel_size=3)

    def initialize(self):

        self.op_keyps_all_frame_paths = sorted(glob(os.path.join(self.op_data_each_video_dir, "*.json")))

    def preprocess(self, output_data_path):
        all_keypoints = []
        for op_keyps_each_frame_path in self.op_keyps_all_frame_paths:

            # Extract index from the file name. Skip the frames that were excluded
            base_name = fullfile(op_keyps_each_frame_path)[0]
            if "demo_video" in base_name:

                frame_idx = int(base_name.split("_")[2])
            else:
                frame_idx = int(base_name.split("_")[3])

            # print("current: {} | between {} - {} ".format(frame_idx, self.start_idx, self.end_idx))
            if frame_idx < self.start_idx or frame_idx > self.end_idx:
                continue

            # Transform openpose keypoints to the ordering of detectron's
            op_keyps = read_and_select_openpose_keypoints(op_keyps_each_frame_path)  # (3, 25)
            try:
                op_keyps_to_detectron = op_keyps[:, openpose2detectron_indexes]  # (3, 17), same as detectron's
            except IndexError:
                import pdb
                pdb.set_trace()

            # Get detectron's preprocessing records
            de_info_idx = self.de_info[frame_idx - self.start_idx]

            # Transform to cropped video's coordinate reference frame and append to list for saving
            op_keyps_to_detectron[0:2, :] += de_info_idx['translation']  # (3, 17)
            op_keyps_to_detectron[0:2, :] = op_keyps_to_detectron[0:2, :] * de_info_idx['resizing']  # (3, 17)

            # Clipping
            op_keyps_to_detectron = np.clip(op_keyps_to_detectron, 0, self.cut_video_size[0])
            # op_keyps_to_detectron = clip_to_video_size(op_keyps_to_detectron, self.cut_video_size)

            # Getting rid of the null detection (i.e. x,y = 0,0)
            mask = (op_keyps_to_detectron[0, :] == 0) & (op_keyps_to_detectron[1, :] == 0)
            mask_reshape = (mask.reshape(1, 17) * np.ones((3, 17)))
            op_keyps_to_detectron[mask_reshape.astype(np.bool)] = np.nan

            # Temporal filter on x, y coordinate but not confidence
            only_xy = self.keypoints_filter.add(op_keyps_to_detectron[[0, 1], :])
            op_keyps_to_detectron[[0, 1], :] = only_xy

            all_keypoints.append(op_keyps_to_detectron.T)  # transform to (17, 3) to fit the defined format

        save_data_for_VideoPose3D(output_data_path, self.cut_video_size, (self.start_idx, self.end_idx), all_keypoints)


def detectron_preprocess_wrapper(src_vid_dir, all_data_dir, output_vid_dir, output_keypoints_dir):
    all_vid_paths = glob(os.path.join(src_vid_dir, "*"))
    num_vids = len(all_vid_paths)

    for idx, vid_path in enumerate(all_vid_paths):
        print("\rPreprocessing {}/{}: from {}".format(idx, num_vids, vid_path))
        vid_name_root = os.path.splitext(os.path.split(vid_path)[1])[0]
        # if vid_name_root != 'vid0189_4898_20171130':
        #     continue
        all_data_path = os.path.join(all_data_dir, vid_name_root + ".npy")
        output_vid_path = os.path.join(output_vid_dir, vid_name_root + ".mp4")
        output_keypoints_path = os.path.join(output_keypoints_dir, vid_name_root + ".npz")

        preprocesser = DetectronPreprocessor(vid_path, all_data_path)
        preprocesser.initialize(keyps_threshold_quantile=0.25)
        preprocesser.preprocess(output_vid_path, output_keypoints_path, plot_mode=False)


def openpose_preprocess_fromDetectronBox_wrapper(openpose_main_dir,
                                                 detectron_preprocessed_2Dkeyps_dir,
                                                 output_preprocessed_2Dkeyps_dir):
    all_subfolders = glob(os.path.join(openpose_main_dir, "*"))
    num_subfolders = len(all_subfolders)
    for idx, subfolder in enumerate(all_subfolders):
        print("%d/%d: %s" % (idx, num_subfolders, subfolder))
        vid_name_root = fullfile(subfolder)[1][1]
        detectron_preprocessed_2Dkeyps_path = os.path.join(detectron_preprocessed_2Dkeyps_dir, vid_name_root)
        output_path = os.path.join(output_preprocessed_2Dkeyps_dir, vid_name_root)
        preprocesser = OpenposePreprocesser_fromDetectronBox(subfolder, detectron_preprocessed_2Dkeyps_path)
        preprocesser.initialize()
        preprocesser.preprocess(output_path)


def openpose_preprocess_wrapper(src_vid_dir, input_data_main_dir, output_vid_dir,
                                output_data_dir, error_log_path, plot_keypoints=False,
                                write_video=True):
    """
    This function preprocesses the raw videos and keypoints that were inferred by openpose, by cropping, centering and
    scaling the coordinates and bounding boxes.
    The original keypoints are stored in subfolders inside "input_data_main_dir". This fuunction searches for the raw
    videos in "src_vid_dir" with the same name as the subfolders' (with extension ".mp4")

    Parameters
    ----------
    src_vid_dir : str
        Directory that you store your raw videos.
    input_data_main_dir : str
        Directory that you store the subfolders of keypoints, inferred from openpose.
    output_vid_dir : str
        Directory that you store the output preprocessed visualisation videos.
    output_data_dir : str
        Directory that you store the output preprocessed keypoints.
    plot_keypoints : bool
        True if you want to plot the keypoints on the output visualisation videos. False if you don't.
    write_video : bool
        True if you want to store the output preprocessed visualisation videos.
    """

    subfolder_paths = sorted(glob(os.path.join(input_data_main_dir, "*")))
    num_vids = len(subfolder_paths)

    for idx, subfolder_path_each in enumerate(subfolder_paths):

        # Monitor progress
        print("\rPreprocessing {}/{}: from {}".format(idx, num_vids, subfolder_path_each))

        # Define paths
        vid_name_root = os.path.split(subfolder_path_each)[1]
        input_video_path = os.path.join(src_vid_dir, vid_name_root + ".mp4")
        output_vid_path = os.path.join(output_vid_dir, vid_name_root + ".mp4")
        output_keypoints_path = os.path.join(output_data_dir, vid_name_root + ".npz")

        # Skip if the outputp already exists
        if os.path.isfile(output_keypoints_path):
            print("Skipped: ", vid_name_root)
            continue

        # try:
        # Start preprocessing
        preprop = OpenposePreprocessor(input_video_path=input_video_path,
                                       openpose_data_each_video_dir=subfolder_path_each,
                                       output_video_path=output_vid_path,
                                       output_data_path=output_keypoints_path)
        preprop.initialize()
        preprop.preprocess(plot_keypoints=plot_keypoints, write_video=write_video)

        # except Exception as e:
        #
        #     print("Error encountered. Logged in {}".format(error_log_path))
        #     with open(error_log_path, "a") as fh:
        #         fh.write("\n{}\n".format(vid_name_root))
        #         fh.write(str(e))
        #     continue


if __name__ == "__main__":
    pass
