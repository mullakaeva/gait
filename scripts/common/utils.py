import os
import json
import numpy as np
import re
import pickle

class Detectron_data_loader():
    def __init__(self, data_path):
        self.boxes, self.keyps = self.read_detectron_output(data_path)

    def getbox(self, frame_idx):
        return self.boxes[frame_idx][1][0]
    def getkeyps(self,frame_idx):
        return self.keyps[frame_idx][1][0]
    def __len__(self):
        return len(self.keyps)

    @staticmethod
    def read_detectron_output(data_path):
        # output: bbox (dict), keypoints (dict), with key = frame index
        data = np.load(data_path, encoding = "latin1")
        return data[()]['boxes'], data[()]['keyps']

class RunningAverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

class MeterAssembly:

    def __init__(self, *args):
        self.meter_dicts = dict()
        self.recorder = dict()
        for arg in args:
            self.meter_dicts[arg] = RunningAverageMeter()
            self.recorder[arg] = []

    def update_meters(self, **kwargs):
        for key in kwargs:
            self.meter_dicts[key].update(kwargs[key])

    def get_meter_avg(self):
        output_dict = dict()
        for key in self.meter_dicts.keys():
            output_dict[key] = self.meter_dicts[key].avg
        return output_dict

    def update_recorders(self):
        for key in self.meter_dicts.keys():
            self.recorder[key].append(self.meter_dicts[key].avg)

    def get_recorders(self):
        return self.recorder


class OnlineFilter_scalar():
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
        self.input_records = np.zeros(kernel_size)
        self.input_times = 0
    def add(self, input_val):
        if self.input_times < self.kernel_size:
            self.input_records[self.input_times] = input_val
            self.input_times += 1
            return np.mean(self.input_records[0:self.input_times])
        else:
            arr_idx = self.input_times % self.kernel_size
            self.input_records[arr_idx] = input_val
            self.input_times += 1
            return np.mean(self.input_records)
    def get_last(self):
        if self.input_times > 0:
            return self.input_records[-1]

class OnlineFilter_np():
    def __init__(self, input_size, kernel_size):
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.input_records = self._create_records_arr()
        self.input_times = 0
    def add(self, input_val):
        assert input_val.shape == self.input_size
        if self.input_times < self.kernel_size:
            self.input_records[self.input_times, ] = input_val
            self.input_times += 1
            return np.nanmean(self.input_records[0:self.input_times, ], axis = 0)
        else:
            arr_idx = self.input_times % self.kernel_size
            self.input_records[arr_idx, ] = input_val
            self.input_times += 1
            return np.nanmean(self.input_records, axis = 0)
    def get_last(self):
        if self.input_times > 0:
            return self.input_records[-1, ]

    def _create_records_arr(self):
        """
        If input_size = (2, 17), and kernel_size = 15, this function will return zero's array of shape (15, 2, 17)
        Hence, input history will always be stored at axis=0
        """
        records_size = [self.kernel_size]
        for i in range(len(self.input_size)):
            records_size.append(self.input_size[i])
        input_records = np.zeros(records_size)
        return input_records

class LabelsReader():
    def __init__(self, labels_path):
        self.labels_path = labels_path
        self.loaded_df = self._read_data_meta_info()
        self.all_filenames = []
        self.vid2label, self.label2vid = self._construct_conversion_dict()
        
    def get_label(self, vid_name):
        return self.vid2label[vid_name]
    def get_vid(self, label):
        return self.label2vid[label]

    def get_vid2label(self):
        return self.vid2label
    def get_label2vid(self):
        return self.label2vid

    def get_all_filenames(self):
        return self.all_filenames

    def _construct_conversion_dict(self):
        # indexes: 6=zmatrix_rowidx, 7=vid_filename (with trailing '\n')
        vid2label = dict()
        label2vid = dict()
        for i in range(self.loaded_df.shape[0]):
            vid_name = self.loaded_df.iloc[i, 7].replace('\n', '')
            label = int(self.loaded_df.iloc[i, 6]) - 1
            vid2label[vid_name] = label
            label2vid[label] = vid_name
            self.all_filenames.append(vid_name)
        return vid2label, label2vid

    def _read_data_meta_info(self):
        with open(self.labels_path, "rb") as fh:
            try:
                loaded_df = pickle.load(fh, encoding='latin1')
            except TypeError:
                loaded_df = pickle.load(fh)
        return loaded_df


def convert_1d_to_onehot(arr):
    """
    Convert 1d numpy array (values parsed to integers) to 2d one-hot vector array.
    Parameters
    ----------
    arr : numpy.darray
        1d-array

    Returns
    -------
        2d-array with one hot vectors along axis 1
    """
    arr_int = arr.astype(np.int)
    label_types = [0, 1, 2, 3, 4, 5, 6, 7]
    output = np.zeros((arr_int.shape[0], len(label_types)))
    for i in range(arr_int.shape[0]):
        output[i, arr_int[i]] = 1
    return output


def load_df_pickle(df_path):
    with open(df_path, "rb") as fh:
        try:
            loaded_df = pickle.load(fh, encoding='latin1')
        except TypeError:
            loaded_df = pickle.load(fh)
    return loaded_df

def write_df_pickle(df, write_path):
    with open(write_path, "wb") as fh:
        pickle.dump(df, fh)



def moving_average(arr, kernel_size):
    """
    Unfortunately np.convolve() doesn't provide "boundary replication" padding option. 
    I need to pad the boundary by myself to avoid the boundary artefact of convolution with 0 padding.
    """
    arr_size = arr.shape[0]
    num_to_pad = arr_size - (arr_size - kernel_size + 1)
    if num_to_pad % 2 != 0:
        start_num = np.floor(num_to_pad/2)
        end_num = np.ceil(num_to_pad/2)
    else:
        start_num, end_num = num_to_pad/2, num_to_pad/2
    start_element, end_element = arr[0], arr[-1]
    front_arr = np.ones(int(start_num)) * start_element
    end_arr = np.ones(int(end_num)) * end_element
    padded_arr = np.concatenate((front_arr, arr, end_arr))
    moving_average = np.convolve(padded_arr, np.ones(kernel_size)/kernel_size, 'valid')
    assert moving_average.shape[0] == arr.shape[0]
    return moving_average


def fullfile(file_path):
    """
    /home/hoi/text.txt = ("text.txt", ("/home/hoi",
                                       "text",
                                       "txt"))
    """

    base_dir, file_name = os.path.split(file_path)
    file_root_name, extension = os.path.splitext(file_name)
    
    return (file_name, (base_dir, file_root_name, extension))


def json2dict(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data

def read_preprocessed_keypoints(data_path):
    return np.load(data_path)['positions_2d'][()]['S1']['Directions 1'].squeeze()


def read_openpose_keypoints(json_path):
    """
    Read openpose's 2D keypoints from .json file. Each json file represents the keypoints of each video frame.

    Args:
        json_path: (str) Path to the .json keypoints file
    Returns:
        keypoints_dict: (dict) Parsed dictionary of the keypoints. 
        num_people: (int) Number of people detected by openpose. It can be equal or greater than 0
        frame_idx: (int) Index of the corresponding video frame, as indicated in the file name of .json
    """
    # Find frame index
    file_base_name = fullfile(json_path)[0]
    frame_idx = int(re.findall('_(\\d{12})_', file_base_name)[0])

    # Find keypoints and number of people
    keypoints_dict = json2dict(json_path)
    num_people = len(keypoints_dict['people'])
    return keypoints_dict, num_people, frame_idx

def read_oenpose_preprocessed_keypoints(np_path):
    return np.load(np_path)['positions_2d']



def read_and_select_openpose_keypoints(json_path):
    """
    Extended from function read_openpose_keypoints(). Read and select the keypoints of the person in the rightest of the frame.

    Args:
        json_path: (str) Path to the .json keypoints file
    Returns:
        keypoints: (ndarray) Numpy array of keypoints with shape (3,25), representing x,y,confidence of the 25 keypoints
    """
    keypoints_dict, num_people, _ = read_openpose_keypoints(json_path)
    if num_people == 0:
        return np.zeros((3,25))
    elif num_people == 1:
        data_flat = np.asarray(keypoints_dict['people'][0]['pose_keypoints_2d'])
        data = data_flat.reshape(int(len(data_flat)/3), 3).T
        return data
    elif num_people > 1:
        all_people_keypoints = []
        for i in range(num_people):
            data_flat_each_person = np.asarray(keypoints_dict['people'][i]['pose_keypoints_2d'])
            data_each_person = data_flat_each_person.reshape(int(len(data_flat_each_person)/3), 3).T # (3, 25)
            all_people_keypoints.append(data_each_person)
        all_people_keypoints_np = np.asarray(all_people_keypoints)
        person_idx = np.argmax(np.mean(all_people_keypoints_np[:, [0], :], axis = 2), axis =0)
        return all_people_keypoints_np[int(person_idx), :, :]




def rename_files(folder, replace_arg):
    from glob import glob
    import os
    all_data = glob(os.path.join(folder, "*"))
    for data in all_data:
        data_root_name = os.path.splitext(os.path.split(data)[1])[0]
        new_name = data_root_name.replace(replace_arg[0], replace_arg[1])
        new_path = os.path.join(folder, new_name)
        os.rename(data, new_path)
        print(data)
        print("renamed to:\n{}\n".format(new_path))


def sample_subset_of_videos(src_dir, sample_num = 1000, labels_path = "", seed = 50, with_labels = True):
    """
    This function sample a number (sample_num, int) of .mp4 videos from source directory (src_dir, str) 
    with a state of randomness (seed, int) and copy them to destination directory (dest_dir, str), 
    and output a log file which records which videos have been copied if write_log == True.
    """
    from glob import glob

    # Grab all videos
    videos_list = glob(os.path.join(src_dir, "*.mp4"))
    filtered_videos_list = []

    # If labels are needed, filter out videos without labels
    if with_labels:
        print("Filtering in those videos with labels... ")
        lreader = LabelsReader(labels_path)
        filenames_with_labels = lreader.get_all_filenames()
        for vid_path_each in videos_list:
            vid_base_name = fullfile(vid_path_each)[0]
            if vid_base_name in filenames_with_labels:
                filtered_videos_list.append(vid_path_each)
    else:
        filtered_videos_list = videos_list

    print("Sample and copy videos")
    # Sample videos
    filtered_videos_list_np = np.array(filtered_videos_list)
    if sample_num > 0:
        np.random.seed(seed)
        selected_videos = np.random.choice(filtered_videos_list_np, sample_num, replace=False)
    else:
        selected_videos = filtered_videos_list_np

    return selected_videos

def sample_and_copy_videos(src_dir, dest_dir, sample_num = 1000, labels_path = "", seed = 50, write_log = False, with_labels = True):
    import shutil
    videos_list_np = sample_subset_of_videos(src_dir, sample_num, labels_path, seed, with_labels)
    num_videos = videos_list_np.shape[0]
    # Record the videos copied if needed
    i = 0
    for vid_path in videos_list_np:
        shutil.copy(vid_path, dest_dir)
        if write_log:
            with open("sampled_videos_subset.txt", "a") as fh:
                fh.write("%s"%vid_path)
        i += 1
    print("number of videos copied: %d/%d" % (i,num_videos))

def gaitclass(idx):
    return idx2class[idx]


idx2class = {
    0: 'Preferred speed',
    1: 'Slow speed',
    2: 'Max speed',
    3: 'Head extended gait',
    4: 'Dual (verbal)',
    5: 'Dual (subtraction)',
    6: 'Dual (tray)',
    7: 'Eye closed'
}