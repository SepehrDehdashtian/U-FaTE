# folk_feature_loader.py

import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader

import numpy as np
import os
import torch
from random import shuffle
import dgl
from scipy.sparse import csgraph
import h5py
from sklearn.decomposition import PCA
import scipy.sparse as sp

from sklearn.model_selection import train_test_split

__all__ = ['FeatureLoaderFolk']



class FeatureLoaderFolk:
    def __init__(self, opts):
        self.opts = opts

    def load_data(self):
        dataset_path = self.opts.dataset_options["path"]
        sample_ratio = self.opts.dataset_options["dataset"]["sample_ratio"]
        sensitive_attr = self.opts.dataset_options["sens_attr"]

        features = torch.from_numpy(np.loadtxt(os.path.join(dataset_path, 'features_embedded.out'))).float()
        labels = torch.from_numpy(np.loadtxt(os.path.join(dataset_path, 'label.out')))
        if sensitive_attr == 'race':
            sensitive = torch.from_numpy(np.loadtxt(os.path.join(dataset_path, 'group_race.out')))
            # sensitive: 1 to 9 -> 0 to 8
            sensitive -= 1
        else:
            sensitive = torch.from_numpy(np.loadtxt(os.path.join(dataset_path, 'group.out'))).unsqueeze(1)


        # Combine Employment classes from 7 classes to 4 classes :: 0, 1, {2,3,4,5}, 6 
        if labels.max() == 6:
            labels[labels == 3] = 2 
            labels[labels == 4] = 2 
            labels[labels == 5] = 2 
            
            labels[labels == 6] = 3



        # Sample Data
        org_idx = torch.arange(len(features))
        idx_sampled, _ = train_test_split(org_idx, train_size=sample_ratio)
        
        features  = features[idx_sampled]
        labels    = labels[idx_sampled]
        sensitive = sensitive[idx_sampled]

        return features, labels, sensitive
