from glob import glob
from abc import ABC, abstractmethod
from .utils import LabelsReader, fullfile, load_df_pickle
import random
import os
import numpy as np


# Abbreviations:
# SSF = simple shallow features analysis

class DataGenerator(ABC):
    def __init__(self, data_dir, batch_size):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.all_data_paths = glob(os.path.join(data_dir, "*"))
        self.num_files = len(self.all_data_paths)

    def iterator(self):
        duration_indices = []
        start = 0
        for stop in range(0, self.num_files, self.batch_size):
            if stop - start > 0:
                duration_indices.append((start, stop))
                start = stop
        random.shuffle(self.all_data_paths)

        for start, stop in duration_indices:
            sampled_data = self._convert_paths_to_data(start, stop)
            yield sampled_data

    @abstractmethod
    def _convert_paths_to_data(self, start, stop):
        return None


class SingleNumpy_DataGenerator(DataGenerator):
    def __init__(self, data_dir, batch_size=1):
        super(SingleNumpy_DataGenerator, self).__init__(data_dir, batch_size)

    def _convert_paths_to_data(self, start, stop):
        batch_data_paths = self.all_data_paths[start:stop]
        for idx, batch_data_path in enumerate(batch_data_paths):
            data_each = np.load(batch_data_path)['positions_2d']

            return (data_each, batch_data_path)


class SSF_tSNE_DataGenerator(DataGenerator):
    def __init__(self, data_dir, labels_path, batch_size, n_dims):
        super(SSF_tSNE_DataGenerator, self).__init__(data_dir, batch_size)
        self.n_dims = n_dims
        self.data_shape = self._set_data_shape()
        self.label_reader = LabelsReader(labels_path)
        self._correct_to_only_labelled_videos()

    def _convert_paths_to_data(self, start, stop):
        batch_data_paths = self.all_data_paths[start:stop]

        batch_data = np.zeros(self.data_shape)
        batch_labels = np.zeros(self.batch_size)
        for idx, batch_data_path in enumerate(batch_data_paths):
            vid_base_name = fullfile(batch_data_path)[1][1] + ".mp4"  # vid1065_xxxx.mp4

            # Create batch data
            data_each = np.load(batch_data_path)
            batch_data[idx,] = data_each

            # Create batch labels
            label = self.label_reader.get_label(vid_base_name)
            batch_labels[idx] = label

        return batch_data, batch_labels, batch_data_paths

    def _set_data_shape(self):
        if isinstance(self.n_dims, int):
            data_shape = (self.batch_size, self.n_dims)
        elif isinstance(self.n_dims, (tuple, list, np.ndarray)):
            data_shape = [self.batch_size] + [x for x in self.n_dims]
        else:
            raise TypeError("n_dims has to be either int, typle, list or np.ndarray object.")
        return data_shape

    def _correct_to_only_labelled_videos(self):
        all_available_vid_base_names = self.label_reader.get_all_filenames()
        available_data_paths = []
        for path_each in self.all_data_paths:
            vid_base_name = fullfile(path_each)[1][1] + ".mp4"
            if vid_base_name in all_available_vid_base_names:
                available_data_paths.append(path_each)
        self.all_data_paths = available_data_paths
        self.num_files = len(self.all_data_paths)


class GaitGeneratorFromDF:

    def __init__(self, df_pickle_path, m=32, n=128, train_portion=0.95):
        self.df = load_df_pickle(df_pickle_path)
        self.total_num_rows = self.df.shape[0]
        split_index = int(self.total_num_rows * train_portion)
        self.df_train = self.df.iloc[0:split_index, :].reset_index(drop=True)
        self.df_test = self.df.iloc[split_index:, :].reset_index(drop=True)

        self.num_rows = self.df_train.shape[0]
        self.m, self.n = m, n

        self.label_range = np.max(self.df["labels"]) - np.min(self.df["labels"])

    def iterator(self):
        duration_indices = []
        start = 0
        for stop in range(0, self.num_rows, self.m):
            if stop - start > 0:
                duration_indices.append((start, stop))
                start = stop
        df_shuffled = self.df_train.iloc[np.random.permutation(self.num_rows), :]

        for start, stop in duration_indices:
            sampled_data, times = self._convert_df_to_data(df_shuffled, start, stop)
            yield sampled_data, times

    def _convert_df_to_data(self, df_shuffled, start, stop):

        selected_df = df_shuffled.iloc[start:stop, :].copy()
        output_arr, times = self._loop_for_array_construction(selected_df, self.m)
        output_arr_test, _ = self._loop_for_array_construction(self.df_test, self.m)
        output_arr = output_arr.reshape(self.m, self.n, 25*3)
        output_arr_test = output_arr_test.reshape(self.m, self.n, 25 * 3)
        return (output_arr, output_arr_test), times

    def _loop_for_array_construction(self, df, num_samples):
        output_arr = np.zeros((num_samples, self.n, 25, 3))

        for i in range(num_samples):

            # Get features and labels
            fea_vec = df["features"].iloc[i]  # numpy.darray (num_frames, 25, 2)
            label = df["labels"].iloc[i] / self.label_range  # numpy.int64

            # Slice to the receptive window
            slice_start = np.random.choice(fea_vec.shape[0] - self.n)
            fea_vec_sliced = fea_vec[slice_start:slice_start + self.n, :, :]

            # Expand label to match fea_vec_sliced
            label_np = np.ones((self.n, 25)) * label

            # Construct output
            output_arr[i, :, :, 0:2] = fea_vec_sliced
            output_arr[i, :, :, 2] = label_np

        # Construct times
        times = np.arange(self.n) / 25
        return output_arr, times


class GaitGeneratorFromDFforCVAE(GaitGeneratorFromDF):
    def __init__(self, df_pickle_path, m=32, n=128, label_dims=8, train_portion=0.95):
        # Hard-coded params
        self.keyps_x_dims, self.keyps_y_dims = 25, 25
        self.total_fea_dims = self.keyps_x_dims + self.keyps_y_dims
        # Define label dimension
        self.label_dims = label_dims
        super(GaitGeneratorFromDFforCVAE, self).__init__(df_pickle_path, m, n, train_portion)

    def _convert_df_to_data(self, df_shuffled, start, stop):
        selected_df = df_shuffled.iloc[start:stop, :].copy()
        output_arr, labels = self._loop_for_array_construction(selected_df, self.m)
        output_arr_test, labels_test = self._loop_for_array_construction(self.df_test, self.m)
        return (output_arr, labels), (output_arr_test, labels_test)

    def _loop_for_array_construction(self, df, num_samples):
        output_arr = np.zeros((num_samples, self.total_fea_dims+self.label_dims, self.n))
        label_arr = np.zeros((num_samples, self.label_dims))
        for i in range(num_samples):

            # Get features and labels
            fea_vec = df["features"].iloc[i]  # numpy.darray (num_frames, 25, 2)
            label = df["labels"].iloc[i]  # numpy.int64

            # Slice to the receptive window
            slice_start = np.random.choice(fea_vec.shape[0] - self.n)
            fea_vec_sliced = fea_vec[slice_start:slice_start + self.n, :, :]

            # Expand label to match fea_vec_sliced
            label_np = np.zeros((self.label_dims, self.n))
            label_np[label, :] = 1
            label_arr[i, label] = 1

            # Construct output
            output_arr[i, 0:self.keyps_x_dims, :] = fea_vec_sliced[:, :, 0].T
            output_arr[i, self.keyps_x_dims:self.total_fea_dims, :] = fea_vec_sliced[:, :, 1].T
            output_arr[i, self.total_fea_dims:self.total_fea_dims+self.label_dims, :] = label_np

        return output_arr, label_arr

class GaitGeneratorFromDFforSingleSkeletonVAE:
    def __init__(self, df_pickle_path, m=32, train_portion=0.95):
        # Load dataframe and collapse the num_samples and num_frames
        df = load_df_pickle(df_pickle_path)
        output_arr = self._flatten_feature_sequences(df)  # (num_frames * num_samples, 1, 50)
        self.m, self.total_num_rows = m, output_arr.shape[0]
        del df  # free memory to python process but not the system

        # Construct train and test set
        split_idx = int(self.total_num_rows*train_portion)
        self.data_train = output_arr[0:split_idx, ]
        self.data_test = output_arr[split_idx:, ]
        self.num_rows = self.data_train.shape[0]
        print("Shape of training set: %s\nShape of validating set: %s" % (self.data_train.shape, self.data_test.shape))

    def iterator(self):
        duration_indices = []
        start = 0
        for stop in range(0, self.num_rows, self.m):
            if stop - start > 0:
                duration_indices.append((start, stop))
                start = stop

        arr_train_shuffled = self.data_train[np.random.permutation(self.num_rows), ]
        for start, stop in duration_indices:
            sampled_data = self._convert_arr_to_data(arr_train_shuffled, start, stop)
            yield sampled_data, self.data_test.copy()

    def _convert_arr_to_data(self, arr_shuffled, start, stop):
        data_train_batch = arr_shuffled[start:stop, ].copy()
        return data_train_batch

    @staticmethod
    def _flatten_feature_sequences(df):
        print("Flattening sequences & Concatenating")
        vec_list = []
        for i in range(df.shape[0]):
            fea_vec = df["features"].iloc[i].copy()  # (num_frames, 25, 2)
            vec_list.append(fea_vec)
        output_arr = np.concatenate(vec_list, axis=0)  # (num_frames * num_samples, 25, 2)
        output_arr = output_arr.reshape(-1, 1, 50)  # (num_frames * num_samples, 1, 50)
        return output_arr

