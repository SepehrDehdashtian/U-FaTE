# celeba_feature_loader.py

import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import os
import torch
from random import shuffle
import h5py
import scipy.sparse as sp
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

__all__ = ['CelebAFeatureLoaderYS']



class PrepareCelebA:
    def __init__(self, opts):
        self.opts = opts

    
    def load_data(self):
        device = 'cuda' if self.opts.ngpu else 'cpu'

        self.dataset_path    = self.opts.dataset_options["path"]
        self.images_path     = self.opts.dataset_options["imgs"]
        attrs_filename       = self.opts.dataset_options["attr_filename"]

        target_attr          = self.opts.dataset_options["target_attr"]
        sensitive_attr       = self.opts.dataset_options["sensitive_attr"]


        csv_data = pd.read_csv(os.path.join(self.dataset_path, attrs_filename))


        # Map -1 to 0
        y = torch.tensor(csv_data[target_attr].values)#, device=device)
        y[y == -1] = 0
        s = torch.tensor(csv_data[sensitive_attr].values)#, device=device)
        s[s == -1] = 0
        
        # Combine two or more binary sensitive attributes to create one integer
        if len(s.squeeze().shape) > 1:
            s = s[:, 0] * 2 + s[:, 1]
            
        data = dict()
        data['train'] = {'x': torch.cat((s[:162770].unsqueeze(-1), y[:162770].unsqueeze(-1)), dim=1),          'y': y[:162770],         's': s[:162770]}
        data['val']   = {'x': torch.cat((s[162770:182637].unsqueeze(-1), y[162770:182637].unsqueeze(-1)), dim=1),    'y': y[162770:182637],   's': s[162770:182637]}
        data['test']  = {'x': torch.cat((s[182637:].unsqueeze(-1), y[182637:].unsqueeze(-1)), dim=1),          'y': y[182637:],         's': s[182637:]}
        return data

class CelebADataloader:
    def __init__(self, data, opts):
        self.opts = opts
        self.x      = data['x']
        self.y      = data['y']
        self.s      = data['s']
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x       = self.x[index]
        y       = self.y[index].long()
        s       = self.s[index]
        return x, y, s

# 0  5_o_Clock_Shadow      88.83, 90.01
# 1  Arched_Eyebrows       73.41, 71.55
# 2  Attractive            51.36, 50.41  ***
# 3  Bags_Under_Eyes       79.55, 79.74
# 4  Bald                  97.72, 97.88
# 5  Bangs                 84.83, 84.42
# 6  Big_Lips              75.91, 67.30
# 7  Big_Nose              76.44, 78.80
# 8  Black_Hair            76.10, 72.84
# 9  Blond_Hair            85.10, 86.67
# 10  Blurry               94.86, 94.94
# 11  Brown_Hair           79.61, 82.04
# 12  Bushy_Eyebrows       85.63, 87.04
# 13  Chubby               94.23, 94.70
# 14  Double_Chin          95.35, 95.43
# 15  Eyeglasses           93.54, 93.55
# 16  Goatee               93.65, 95.42
# 17  Gray_Hair            95.76, 96.81
# 18  Heavy_Makeup         61.57, 59.50  **
# 19  High_Cheekbones      54.76, 51.82  ***
# 20  Male                 58.06, 61.35  ***
# 21  Mouth_Slightly_Open  51.78, 50.49  ***
# 22  Mustache             95.92, 96.13
# 23  Narrow_Eyes          88.41, 85.13
# 24  No_Beard             83.42, 85.37
# 25  Oval_Face            71.68, 70.44
# 26  Pale_Skin            95.70, 95.80
# 27  Pointy_Nose          72.45, 71.42
# 28  Receding_Hairline    91.99, 91.51
# 29  Rosy_Cheeks          99.53, 92.83
# 30  Sideburns            94.37, 95.36
# 31  Smiling              52.03, 50.03  ***
# 32  Straight_Hair        79.14, 79.01
# 33  Wavy_Hair            68.06, 63.59  *
# 34  Wearing_Earrings     81.35, 79.33
# 35  Wearing_Hat          95.06, 95.80
# 36  Wearing_Lipstick     53.04, 52.19 ***
# 37  Wearing_Necklace     87.86, 86.21
# 38  Wearing_Necktie      92.70, 92.99
# 39  Young                77.89, 75.71

# (18, 19):  0.0729, 0.0951 *
# (18, 20):  0.4439, 0.4227 ****
# (18, 21):  0.0106, 0.0130
# (18, 31):  0.0308, 0.0382
# (18, 33):  0.1041, 0.1072 **
# (18, 36):  0.6434, 0.5791 ****

# (19, 20):  0.0615, 0.0781
# (19, 21):  0.1747, 0.1741 ***
# (19, 31):  0.4662, 0.4582 ****
# (19, 33):  0.0131, 0.0130
# (19, 36):  0.0793, 0.0936

# (21, 20)  DEO = 0.05
 
class CelebAFeatureLoaderYS(pl.LightningDataModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        if opts.ngpu == 0:
            self.pin_memory = False
        else:
            self.pin_memory = True

        pre = PrepareCelebA(opts)
        self.data = pre.load_data()

    def train_dataloader(self):
        dataset = CelebADataloader(self.data['train'], self.opts)

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.opts.batch_size_train,
            shuffle=True,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory
        )
        return loader

    def val_dataloader(self):
        dataset = CelebADataloader(self.data['val'], self.opts)

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.opts.batch_size_test,
            shuffle=False,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory
        )
        return loader

    def test_dataloader(self):
        dataset = CelebADataloader(self.data['test'], self.opts)

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.opts.batch_size_test,
            shuffle=False,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory
        )
        return loader


    def train_kernel_dataloader(self):
        idx_sampled = torch.randperm(len(self.data['train']['y']))[:self.opts.dataset_options["kernel_numSamples"]]
        x      = self.data['train']['x'][idx_sampled].cuda()
        y      = self.data['train']['y'][idx_sampled].cuda()
        s      = self.data['train']['s'][idx_sampled].cuda()
        return x, y, s


