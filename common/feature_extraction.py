from .utils import read_oenpose_preprocessed_keypoints, fullfile
from .generator import SingleNumpy_DataGenerator
from .keypoints_format import openpose_L_indexes, openpose_R_indexes, openpose_central_indexes

from sklearn.metrics.pairwise import pairwise_distances

import os
import numpy as np


class Imputator():
    def __init__(self, data):
        """
        Args:
            data: Numpy array with shape (num_frames, 25, 3)
        """
        self.data = data[:, :, 0:2]  # Exclude confidence terms

    def mean_imputation(self):
        """
        Returns:
            data: Numpy array of the same shape as input data with np.nan imputed as the mean
        """
        nan_mask = self._search_for_nan()
        mean_keyps = self._calc_means()
        mean_keyps_expanded = np.ones(self.data.shape) * mean_keyps

        self.data[nan_mask] = mean_keyps_expanded[nan_mask]

        return self.data

    def _search_for_nan(self):
        nan_mask = np.isnan(self.data) == True
        return nan_mask

    def _calc_means(self):
        return np.nanmean(self.data, axis=0, keepdims=True)


class CustomMeanImputator(Imputator):
    def __init__(self, data, mean):
        super(CustomMeanImputator, self).__init__(data)

        self.mean = mean[:, 0:2].reshape(1, self.data.shape[1], self.data.shape[2])

    def _calc_means(self):
        return self.mean


class FeatureExtractor():
    def __init__(self, data_dir, save_dir):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.keyps_shape = np.array([25, 3])  # (num_keypoints, keypoint_x_y_confidence)

    def extract(self):
        data_grand_mean = self._incremental_mean_estimation()  # Shape = (25, 3)

        data_gen = SingleNumpy_DataGenerator(self.data_dir, batch_size=1)
        for idx, data_info in enumerate(data_gen.iterator()):
            # Retrive data and print progress
            data, data_path = data_info

            print("Feature Extraction: {}/{} from {}".format(idx, data_gen.num_files, data_path))

            # Imputation
            data_imputed = self._mean_single_imputation(data, data_grand_mean)

            # Clipping (0, 250) and Rescaling (data/255)
            data_imputed = self._clipping_rescaling(data_imputed)

            # Feature extraction work
            extracted_features = self._iterative_workflow(data_imputed)  # Shape = (666,)

            # Saving the data
            assert extracted_features.shape[0] == 666
            save_path = os.path.join(self.save_dir, fullfile(data_path)[1][1])
            np.save(save_path, extracted_features)

    def _incremental_mean_estimation(self):

        data_gen = SingleNumpy_DataGenerator(self.data_dir, batch_size=1)
        num = data_gen.num_files
        data_accumulator = np.zeros([num] + [x for x in self.keyps_shape])

        for idx, data_info in enumerate(data_gen.iterator()):
            data, _ = data_info
            data_mean = np.nanmean(data, axis=0)
            data_accumulator[idx] = data_mean
        data_grand_mean = np.nanmean(data_accumulator, axis=0)
        return data_grand_mean

    def _iterative_workflow(self, data):
        """
        Iterative data of each frame of a video, and calcualte the mean and std of relative euclidean distance between keypoints,
        as well as the asymmetry
        Args:
            data: Numpy array with shape (num_frames, 25, 2)
        Returns:
            features: Numpy array with shape (666, )
        """
        features_all_frames = []

        for data_each_frame in data:  # data_each_frame ~ (25, 2)

            # Relative euclidean distance
            pair_dist = pairwise_distances(data_each_frame, metric='euclidean')
            upper_tri_indices = np.triu_indices(n=self.keyps_shape[0], k=1)
            only_relative_dist = pair_dist[upper_tri_indices[0], upper_tri_indices[1]]  # Shape = (300,)

            # Asymmetry
            asymmetry = self._asymmetry_measure(data_each_frame)  # Shape = (33, )

            features_each_frame = np.append(only_relative_dist, asymmetry)  # Shape = (333, )
            features_all_frames.append(features_each_frame)

        features_all_frames_np = np.array(features_all_frames)  # Shape = (num_frames, 333)
        mean, std = np.mean(features_all_frames_np, axis=0), np.std(features_all_frames_np,
                                                                    axis=0)  # Shape = (333,) for both
        features_flattened = np.append(mean, std)  # Shape = (666, )
        return features_flattened

    def _asymmetry_measure(self, data_each_frame):
        """
        Calculate the RATIO of  Euclidean distance of left keypoints to the right keypoints.
        Args:
            data_each_frame: Numpy array with shape (25, 2)
        Returns:
            asymmetry_features: Numpy array with shape (33, )
        """

        def degree_of_asymmetry(L_keyps, R_keyps, anchor):
            """
            Args:
                L_keyps: Numpy array with shape (M, 2)
                R_keyps: Numpy array with shape (M, 2)
                anchor: Numpy array with shape (1, 2), accepting only one anchor vector in each function call.

            """
            L_eu_dist = np.linalg.norm(L_keyps - anchor, axis=1)  # Shape = (M, )
            R_eu_dist = np.linalg.norm(R_keyps - anchor, axis=1)  # Shape = (M, )
            asy_degree = L_eu_dist / (R_eu_dist + 0.00001)  # Shape = (M,)
            return asy_degree

        L_keyps = data_each_frame[openpose_L_indexes, :]  # Shape = (11, 2)
        R_keyps = data_each_frame[openpose_R_indexes, :]  # Shape = (11, 2)
        anchor_points = data_each_frame[openpose_central_indexes, :]  # Shape = (3, 2)
        all_anchors_degree_list = []
        for i in range(anchor_points.shape[0]):
            each_anchor_point = anchor_points[[i], :]  # Shape = (1, 2)
            degrees_each_anchor = degree_of_asymmetry(L_keyps, R_keyps, each_anchor_point)  # Shape = (11,)
            all_anchors_degree_list.append(degrees_each_anchor)

        asymmetry_features = np.array(all_anchors_degree_list).flatten()  # Shape = (11*3, ) = (33, )
        return asymmetry_features

    @staticmethod
    def _mean_single_imputation(data, mean):
        imputor = CustomMeanImputator(data, mean)
        data_imputed = imputor.mean_imputation()
        return data_imputed

    @staticmethod
    def _clipping_rescaling(data, min_val=0, max_val=250, mul_factor=1 / 250):
        data = np.clip(data, min_val, max_val)
        data = data * mul_factor
        return data
