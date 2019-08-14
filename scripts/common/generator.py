from glob import glob
from abc import ABC, abstractmethod
from .utils import LabelsReader, fullfile, load_df_pickle
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
        self.seed = seed
        self.df = self.df.sample(frac=1, random_state=self.seed)
        split_index = int(self.total_num_rows * train_portion)
        self.df_train = self.df.iloc[0:split_index, :].reset_index(drop=True)
        self.df_test = self.df.iloc[split_index:, :].reset_index(drop=True)


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
            info = self._convert_df_to_data(df_shuffled, start, stop)
            yield info

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

    def __init__(self, df_pickle_path, m=32, n=128, train_portion=0.95, seed=None, gait_print=False):
        """

        Parameters
        ----------
        df_pickle_path : str
        conditional_labels : int
            Number of labels. They are for conditional VAE - the discrete labels will be concatenated to the features.
        m : int
            Sample size for each batch
        n : int
            Sequence length
        train_portion : float
        seed : int

        """

        # Set number of features
        self.keyps_x_dims, self.keyps_y_dims = 25, 25
        self.total_fea_dims = self.keyps_x_dims + self.keyps_y_dims

        # Call parent's init
        super(GaitGeneratorFromDFforTemporalVAE, self).__init__(df_pickle_path, m, n, train_portion, seed)
        self.batch_shape = (m, self.total_fea_dims, n)
        self.gait_print = gait_print

        # Get number of unique patients
        self.num_uni_patients = self._get_num_uni_patients()

    def _convert_df_to_data(self, df_shuffled, start, stop):
        selected_df = df_shuffled.iloc[start:stop, :].copy()
        if self.gait_print:
            selected_df = self._complete_gaitprint(selected_df)

        # Retrieve train data
        x_train_info, task_train_info, pheno_train_info, towards_train, leg_train_info, idpatients = self._loop_for_array_construction(
            selected_df,
            self.m)
        x_train, x_train_masks = x_train_info
        task_train, task_train_masks = task_train_info
        pheno_train, pheno_train_masks = pheno_train_info
        leg_train, leg_train_masks = leg_train_info

        # Retrieve test data
        x_test_info, task_test_info, pheno_test_info, towards_test, leg_test_info, idpatients_test = self._loop_for_array_construction(
            self.df_test,
            self.df_test.shape[0])

        x_test, x_test_masks = x_test_info
        task_test, task_test_masks = task_test_info
        pheno_test, pheno_test_masks = pheno_test_info
        leg_test, leg_test_masks = leg_test_info

        # Combine as output
        train_info = (x_train, x_train_masks, task_train, task_train_masks, pheno_train, pheno_train_masks,
                      towards_train, leg_train, leg_train_masks, idpatients)
        test_info = (x_test, x_test_masks, task_test, task_test_masks, pheno_test, pheno_test_masks,
                     towards_test, leg_test, leg_test_masks, idpatients_test)

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
        # fea_vec/fea_mask_vec ~ (num_frames, 25, 2), task ~ int, task_mask ~ bool (True for non-nan, False for nan)
        # pheno ~ int, pheno_mask ~ bool (True for non-nan, False for nan), towards ~ int (0=unknown, 1=left, 2=right)
        # leg ~ float, leg_mask ~ bool (True for non-nan, False for nan)
        select_list = ["features", "feature_masks", "tasks", "task_masks", "phenos", "pheno_masks",
                       "towards_camera", "leg", "leg_masks", "idpatients"]

        df_np = np.asarray(df[select_list].iloc[0:num_samples])

        fea_vec, fea_mask_vec, task, task_mask, pheno, pheno_mask, towards, leg, leg_mask, idpatients = list(df_np.T)

        task, task_mask = task.astype(np.int), task_mask.astype(np.bool)
        pheno, pheno_mask = pheno.astype(np.int), pheno_mask.astype(np.bool)
        towards, leg, leg_mask = towards.astype(np.int), leg.astype(np.float), leg_mask.astype(np.bool)
        idpatients = idpatients.astype(np.float)

        features_arr = np.zeros((num_samples, self.total_fea_dims, self.n))
        fea_masks_arr = np.zeros(features_arr.shape)

        for i in range(num_samples):

            # Slice to the receptive window
            slice_start = np.random.choice(fea_vec[i, ].shape[0] - self.n)
            fea_vec_sliced = fea_vec[i][slice_start:slice_start + self.n, :, :]
            fea_mask_vec_sliced = fea_mask_vec[i][slice_start:slice_start + self.n, :, :]

            # Construct output
            x_end_idx, y_end_idx = self.keyps_x_dims, self.keyps_x_dims + self.keyps_y_dims
            features_arr[i, 0:x_end_idx, :] = fea_vec_sliced[:, :, 0].T  # Store x-coordinates
            features_arr[i, x_end_idx:y_end_idx, :] = fea_vec_sliced[:, :, 1].T  # Store y-coordinates
            fea_masks_arr[i, 0:x_end_idx, :] = fea_mask_vec_sliced[:, :, 0].T  # Store x-coordinates
            fea_masks_arr[i, x_end_idx:y_end_idx, :] = fea_mask_vec_sliced[:, :, 1].T  # Store y-coordinates
        return (features_arr, fea_masks_arr), (task, task_mask), (pheno, pheno_mask), towards, \
               (leg, leg_mask), idpatients

    def _get_num_uni_patients(self):
        idpatients = self.df["idpatients"]
        idpatients_nonan = idpatients[np.isnan(idpatients) == False]
        unique_ids = np.unique(idpatients_nonan)
        self._convert_idpatients_to_index(unique_ids)
        return unique_ids.shape[0]

    def _convert_idpatients_to_index(self, unique_ids):
        # Build conversion dict
        conversion_dict = dict()
        for idx, idpatient in enumerate(unique_ids):
            conversion_dict[idpatient] = idx
        conversion_dict[np.nan] = np.nan
        # Apply conversion to dataframe
        self.df_train["idpatients"] = self.df_train["idpatients"].apply(lambda x: conversion_dict.get(x, np.nan))
        self.df_test["idpatients"] = self.df_test["idpatients"].apply(lambda x: conversion_dict.get(x, np.nan))

    def _complete_gaitprint(self, df):
        import pdb
        pdb.set_trace()

        current_uni_ids = np.unique(df[df["idpatients"].isnull()==False]["idpatients"])

        pass
    def _construct_patient_tasks_dict(self):
        mask = (self.df_train["idpatients"].isnull()==False) & (self.df_train["task_masks"]==True)
        df_nonan = self.df_train[mask]
        uni_ids = np.unique(df_nonan)

        self.ids_tasks_map = dict()

        for id in uni_ids:
            self.ids_tasks_map[id] =



