
from glob import glob
from abc import ABC, abstractmethod
from .utils import LabelsReader, fullfile
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
            if stop-start > 0:
                duration_indices.append((start, stop))
                start = stop
        random.shuffle(self.all_data_paths)
        
        for start, stop in duration_indices:
            sampled_data = self._convert_paths_to_data(start,stop)
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
            vid_base_name = fullfile(batch_data_path)[1][1]+ ".mp4" # vid1065_xxxx.mp4
            
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
            vid_base_name = fullfile(path_each)[1][1]+ ".mp4"
            if vid_base_name in all_available_vid_base_names:
                available_data_paths.append(path_each)
        self.all_data_paths = available_data_paths
        self.num_files = len(self.all_data_paths)