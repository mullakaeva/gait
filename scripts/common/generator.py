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

        self.label_range = np.max(self.df["tasks"]) - np.min(self.df["tasks"])

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
            label = df["tasks"].iloc[i] / self.label_range  # numpy.int64

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
        (x_train, x_train_masks), (task_train, task_train_masks), (pheno_train, pheno_train_masks) = self._loop_for_array_construction(selected_df, self.m)

        (x_test, x_test_masks), (task_test, task_test_masks), (pheno_test, pheno_test_masks) = self._loop_for_array_construction(self.df_test, self.df_test.shape[0])

        train_info = (x_train, x_train_masks, task_train, task_train_masks, pheno_train, pheno_train_masks)
        test_info = (x_test, x_test_masks, task_test, task_test_masks, pheno_test, pheno_test_masks)
        return train_info, test_info

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
        fea_masks_arr = np.zeros(features_arr.shape)
        tasks_arr = np.zeros(num_samples)
        task_masks_arr = np.zeros(num_samples)
        phenos_arr = np.zeros(num_samples)
        pheno_masks_arr = np.zeros(num_samples)

        for i in range(num_samples):
            # Get features and labels
            fea_vec = df["features"].iloc[i]  # numpy.darray (num_frames, 25, 2)
            fea_mask_vec = df["feature_masks"].iloc[i]  # numpy.bool (num_frames, 25, 2)
            task = df["tasks"].iloc[i]  # numpy.int64
            task_mask = df["task_masks"].iloc[i]  # numpy.bool, True for non-nan, False for nan
            pheno = df["phenos"].iloc[i]  # numpy.int64
            pheno_mask = df["pheno_masks"].iloc[i]  # numpy.bool, True for non-nan, False for nan

            # Slice to the receptive window
            slice_start = np.random.choice(fea_vec.shape[0] - self.n)
            fea_vec_sliced = fea_vec[slice_start:slice_start + self.n, :, :]
            fea_mask_vec_sliced = fea_mask_vec[slice_start:slice_start + self.n, :, :]

            # Put integer label into numpy.darray
            tasks_arr[i] = task
            task_masks_arr[i] = task_mask
            phenos_arr[i] = pheno
            pheno_masks_arr[i] = pheno_mask

            # Construct output
            features_arr[i, 0:self.keyps_x_dims, :] = fea_vec_sliced[:, :, 0].T  # Store x-coordinates
            features_arr[i, self.keyps_x_dims:self.total_fea_dims, :] = fea_vec_sliced[:, :, 1].T  # Store y-coordinates
            fea_masks_arr[i, 0:self.keyps_x_dims, :] = fea_mask_vec_sliced[:, :, 0].T  # Store x-coordinates
            fea_masks_arr[i, self.keyps_x_dims:self.total_fea_dims, :] = fea_mask_vec_sliced[:, :,
                                                                         1].T  # Store y-coordinates

        return (features_arr, fea_masks_arr), (tasks_arr, task_masks_arr), (phenos_arr, pheno_masks_arr)

