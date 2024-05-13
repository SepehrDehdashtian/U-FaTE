# celeba_feature_loader.py

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

__all__ = ['FeatureLoaderCelebA']



class FeatureLoaderCelebA:
    def __init__(self, opts):
        self.opts = opts

    def load_data(self):
        dataset_path = self.opts.dataset_options["path"]
        sample_ratio = self.opts.dataset_options["dataset"]["sample_ratio"]

        self.z = dict()
        self.y = dict()
        self.s = dict()

        features  = torch.from_numpy(np.loadtxt(os.path.join(dataset_path, 'z_train.out')))
        labels    = torch.from_numpy(np.loadtxt(os.path.join(dataset_path, 'y_train.out')))
        sensitive = torch.from_numpy(np.loadtxt(os.path.join(dataset_path, 's_train.out')))

        # map y = -1 to y = 0
        labels[labels == -1] = 0

        # The file has 2 sensitive attribute per sample, we are going to remove the second one (Wearing_Necktie) and only keep the first one (High_Cheekbones)
        sensitive = (sensitive[:, 0] + 1) / 2


        # Sample Data
        org_idx = torch.arange(len(features))
        idx_sampled, _ = train_test_split(org_idx, train_size=sample_ratio)
        
        features  = features[idx_sampled]
        labels    = labels[idx_sampled]
        sensitive = sensitive[idx_sampled]

        print('#Samples = ', len(sensitive))
        # import pdb; pdb.set_trace()
        return features, labels, sensitive
