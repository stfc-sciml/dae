#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 13:53:54 2023

@author: pearl025
"""
import os
import numpy as np
from tqdm import tqdm
from itertools import product


def circle(img, center, radius, color):
    m,n = img.shape
    center = np.array(center, dtype = np.float32)
    x = np.arange(0, m)

    coords = product(x, x)
    coords = np.array(list(coords), dtype = np.float32)

    in_circle = np.where(np.linalg.norm(coords-center, axis = -1) < radius)[0]
    img[coords[in_circle].astype(np.uint8)[:,0], coords[in_circle].astype(np.uint8)[:,1]] = color
    
    return img


datasets = []
labels = []


for i in tqdm(range(15, 84-15-1)):
    for j in range(15, 84-15-1):
            for c in [0.2, 0.4, 0.6, 0.8, 1.0]:
                template = np.zeros((84,84), dtype = np.float32)
                datasets.append(circle(template, (j, i), 15, c))
                labels.append(np.array([j, i, c]))

n_samples = len(datasets)
print(n_samples)

datasets = np.stack(datasets)
labels = np.stack(labels)

dataset_folder_name = 'datasets'

try:
    os.mkdir(dataset_folder_name)
except OSError:
    pass 

np.savez(os.path.join(dataset_folder_name, 'xyc.npz'), imgs = datasets, labs = labels)