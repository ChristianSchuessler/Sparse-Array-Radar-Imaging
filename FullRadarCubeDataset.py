
import os
import random
from typing import Dict
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py

class FullRadarCubeDatasetConfig:
    def __init__(self):
        self.shuffle = True
        self.input_load_callback = None
        self.target_load_callback = None
        self.target_filename = None
        self.input_filename = None
        self.mode = "train"
        self.data_set_size = 1000

        self.number_train_samples = None
        self.number_valid_samples = None
        self.number_test_samples = None
        
class FullRadarCubeDataset(Dataset):
    def __init__(self, data_set_config: FullRadarCubeDatasetConfig):

        self.config = data_set_config
        self.rng = np.random.default_rng(seed=0)

        if self.config.number_test_samples is None:
            number_test_samples = int(np.ceil(self.config.data_set_size*0.10)) # 10 % test samples
        else:
            number_test_samples = self.config.number_test_samples

        if self.config.number_valid_samples is None:
            number_valid_samples = int(np.ceil(self.config.data_set_size*0.10)) # 10 % validation samples
        else:
            number_valid_samples = self.config.number_valid_samples
        
        number_train_samples = self.config.number_train_samples
        number_test_samples = number_test_samples

        if self.config.mode == "train":
            self.idx_offset = 0
            self.dataset_size = number_train_samples
            self.data_indices = np.arange(self.idx_offset, number_train_samples + self.idx_offset)
        elif self.config.mode == "valid":
            self.idx_offset = number_train_samples
            self.dataset_size = number_valid_samples
            self.data_indices = np.arange(self.idx_offset, number_valid_samples + self.idx_offset)
        elif self.config.mode == "test":
            self.idx_offset = number_train_samples + number_valid_samples
            self.data_indices = np.arange(self.idx_offset, self.idx_offset + number_test_samples)
            self.dataset_size = number_test_samples
        else:
            raise Exception(f"Load mode {self.load_mode} is not supported. Supported modes are: train, test, all")
        
        if self.config.shuffle:
            random.seed(1)
            random.shuffle(self.data_indices)

    def shuffle_data(self, seed=1):
        random.seed(seed)
        random.shuffle(self.data_indices)

    def __getitem__(self, idx):
        with h5py.File(self.config.input_filename, 'r') as input_h5:
            if self.config.input_load_callback is None:
                x_ra = np.array(input_h5.get(f'ra_data_{self.data_indices[idx]:06d}'))
                x_rd = np.array(input_h5.get(f'rd_data_{self.data_indices[idx]:06d}'))
                x = (x_ra, x_rd)
            else:
                x = self.config.input_load_callback(input_h5, self.data_indices[idx])

        with h5py.File(self.config.target_filename, 'r') as target_h5:
            if self.config.target_load_callback is None:
                y = np.array(target_h5.get(f'ra_data_{self.data_indices[idx]:06d}'))
            else:
                y = self.config.target_load_callback(target_h5, self.data_indices[idx])

        return x,y

    def __len__(self):
        return self.dataset_size
