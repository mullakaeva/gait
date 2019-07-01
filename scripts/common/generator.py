from glob import glob
from abc import ABC, abstractmethod
from .utils import LabelsReader, fullfile, load_df_pickle, convert_1d_to_onehot
from .keypoints_format import excluded_points_flatten
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

    def __init__(self, df_pickle_path, m=32, n=128, train_portion=0.95, seed=None):
        self.df = load_df_pickle(df_pickle_path)
        self.total_num_rows = self.df.shape[0]
        split_index = int(self.total_num_rows * train_portion)
        self.df_train = self.df.iloc[0:split_index, :].reset_index(drop=True)
        self.df_test = self.df.iloc[split_index:, :].reset_index(drop=True)

        self.seed = seed

        self.num_rows = self.df_train.shape[0]
        self.m, self.n = m, n

        self.label_range = np.max(self.df["labels"]) - np.min(self.df["labels"])

    def iterator(self):
        """
        Randomly sample the indexes from data frame, and yield the sampled batch with the same indexes

        Returns
        -------

        """
        duration_indices = []
        start = 0
        for stop in range(0, self.num_rows, self.m):
            if stop - start > 0:
                duration_indices.append((start, stop))
                start = stop

        if self.seed is not None:
            np.random.seed(self.seed)
        df_shuffled = self.df_train.iloc[np.random.permutation(self.num_rows), :]

        for start, stop in duration_indices:
            sampled_data, times = self._convert_df_to_data(df_shuffled, start, stop)
            yield sampled_data, times

    def _convert_df_to_data(self, df_shuffled, start, stop):

        selected_df = df_shuffled.iloc[start:stop, :].copy()
        output_arr, times = self._loop_for_array_construction(selected_df, self.m)
        output_arr_test, _ = self._loop_for_array_construction(self.df_test, self.m)
        output_arr = output_arr.reshape(self.m, self.n, 25 * 3)
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


class GaitGeneratorFromDFforTemporalVAE(GaitGeneratorFromDF):
    """
    Example of usage:

        data_gen = GaitGeneratorFromDFforTemporalVAE(df_pickle_path, m, n, 0.95)

        for (features_train, labels_train), (features_test, labels_test) in data_gen.iterator():
            ...

    where features_train/test has shape (m, num_features=50, n), and labels_train/test has shape (m, )

    """

    def __init__(self, df_pickle_path, m=32, n=128, train_portion=0.95, seed=None):
        # Hard-coded params
        self.keyps_x_dims, self.keyps_y_dims = 25, 25
        self.total_fea_dims = self.keyps_x_dims + self.keyps_y_dims

        # Call parent's init
        super(GaitGeneratorFromDFforTemporalVAE, self).__init__(df_pickle_path, m, n, train_portion, seed)
        self.batch_shape = (m, self.total_fea_dims, n)

    def _convert_df_to_data(self, df_shuffled, start, stop):
        selected_df = df_shuffled.iloc[start:stop, :].copy()
        x_train, x_mask_train, label_train, label_mask_train = self._loop_for_array_construction(selected_df, self.m)
        x_test, x_mask_test, label_test, label_mask_test = self._loop_for_array_construction(self.df_test, self.df_test.shape[0])
        return (x_train, x_mask_train, label_train, label_mask_train), (x_test, x_mask_test, label_test, label_mask_test)

    def _loop_for_array_construction(self, df, num_samples):
        """

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe with columns "features" (numpy.darray (num_frames, 25, 2)) and "labels" (numpy.int64)
        num_samples : int
            Size of the sampled data

        Returns
        -------
        input_features : numpy.darray
            It has shape (num_samples, num_features=50, num_time_window), as input array to the network
        labels : numpy.darray
            It has shape (num_samples, ) numpy.int64 [0, 7], as the labels for visualisation

        """
        features_arr = np.zeros((num_samples, self.total_fea_dims, self.n))
        nan_arr = np.zeros(features_arr.shape)
        label_arr = np.zeros(num_samples)
        label_mask_arr = np.zeros(num_samples)
        for i in range(num_samples):
            # Get features and labels
            fea_vec = df["features"].iloc[i]  # numpy.darray (num_frames, 25, 2)
            label = df["labels"].iloc[i]  # numpy.int64
            nan_mask = df["nan_masks"].iloc[i]  # numpy.bool (num_frames, 25, 2)
            label_mask = df["label_masks"].iloc[i]  # numpy.bool, True for non-nan, False for nan

            # Slice to the receptive window
            slice_start = np.random.choice(fea_vec.shape[0] - self.n)
            fea_vec_sliced = fea_vec[slice_start:slice_start + self.n, :, :]
            nan_arr_sliced = nan_mask[slice_start:slice_start + self.n, :, :]

            # Put integer label into numpy.darray
            label_arr[i] = label
            label_mask_arr[i] = label_mask

            # Construct output
            features_arr[i, 0:self.keyps_x_dims, :] = fea_vec_sliced[:, :, 0].T  # Store x-coordinates
            features_arr[i, self.keyps_x_dims:self.total_fea_dims, :] = fea_vec_sliced[:, :, 1].T  # Store y-coordinates
            nan_arr[i, 0:self.keyps_x_dims, :] = nan_arr_sliced[:, :, 0].T  # Store x-coordinates
            nan_arr[i, self.keyps_x_dims:self.total_fea_dims, :] = nan_arr_sliced[:, :, 1].T  # Store y-coordinates

        return features_arr, nan_arr, label_arr, label_mask_arr


class GaitGeneratorFromDFforSingleSkeletonVAE:
    def __init__(self, df_pickle_path, m=32, train_portion=0.95):
        # Hard-coded params
        self.keyps_x_dims, self.keyps_y_dims = 25, 25
        self.total_fea_dims = self.keyps_x_dims + self.keyps_y_dims
        self.excluded_keypoints = excluded_points_flatten

        # Load dataframe and collapse the num_samples and num_frames
        df = load_df_pickle(df_pickle_path)
        output_arr, labels = self._flatten_feature_sequences(df)  # (num_frames * num_samples, 50)
        self.m, self.total_num_rows = m, output_arr.shape[0]
        del df  # free memory to python process but not the system

        self.weighting_vec = np.ones(self.total_fea_dims)  # Construct weighting vector
        self.weighting_vec[excluded_points_flatten] = 0  # masked out nose, two eyes

        # Construct train and test set
        split_idx = int(self.total_num_rows * train_portion)
        self.data_train = output_arr[0:split_idx, ]
        self.labels_train = labels[0:split_idx, ]
        self.data_test = output_arr[split_idx:, ]
        self.labels_test = labels[split_idx:, ]

        self.num_rows = self.data_train.shape[0]
        print("Shape of training set: %s\nShape of validating set: %s" % (self.data_train.shape, self.data_test.shape))

    def iterator(self):
        duration_indices = []
        start = 0
        for stop in range(0, self.num_rows, self.m):
            if stop - start > 0:
                duration_indices.append((start, stop))
                start = stop
        ran_vec = np.random.permutation(self.num_rows)
        arr_train_shuffled = self.data_train[ran_vec,]
        labels_train_shuffled = self.labels_train[ran_vec,]

        for start, stop in duration_indices:
            sampled_data = self._convert_arr_to_data(arr_train_shuffled, start, stop)
            sampled_labels = self._convert_arr_to_data(labels_train_shuffled, start, stop)
            yield (sampled_data, sampled_labels), (self.data_test.copy(), self.labels_test.copy())

    def get_weighting_vec(self):
        return self.weighting_vec.copy()

    @staticmethod
    def _convert_arr_to_data(arr_shuffled, start, stop):
        data_train_batch = arr_shuffled[start:stop, ].copy()
        return data_train_batch

    @staticmethod
    def _flatten_feature_sequences(df):
        print("Flattening sequences & Concatenating")
        vec_list = []
        label_vec_list = []
        for i in range(df.shape[0]):
            # Fetch from dataframe
            fea_vec = df["features"].iloc[i].copy()  # (num_frames, 25, 2)
            label = df["labels"].iloc[i].copy()
            label_vec = np.ones(fea_vec.shape[0]) * label

            # Normalisation and flattening
            fea_vec = fea_vec - np.mean(fea_vec, axis=1, keepdims=True)
            fea_flatten = np.zeros((fea_vec.shape[0], 50))  # (num_frames, 50)
            fea_flatten[:, 0:25], fea_flatten[:, 25:50] = fea_vec[:, :, 0], fea_vec[:, :, 1]

            # Appending
            vec_list.append(fea_flatten)
            label_vec_list.append(label_vec)

        # Concatenation
        output_arr = np.concatenate(vec_list, axis=0)  # (num_frames * num_samples, 50)
        label_arr = np.concatenate(label_vec_list)  # (num_frames * num_samples, )

        return output_arr, label_arr
