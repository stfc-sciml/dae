#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:10:11 2023

@author: pearl025
"""
from torch.utils.data import Dataset
import torch

import numpy as np
import os
import random

class XYC(Dataset):
    def __init__(self, path_dir):
        self.data_path = os.path.join(path_dir, "xyc.npz")
        self.data_zip = np.load(self.data_path)
        self.data = self.data_zip['imgs'] 
        self.labels = self.data_zip['labs']
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
                    
        sample = self.data[idx]
        label = self.labels[idx]

        sample = torch.from_numpy(np.expand_dims(sample, axis = 0))
        label = torch.from_numpy(np.expand_dims(label, axis = 0))

        sample = {'x':sample,
                  'y':label}
            
        return sample
