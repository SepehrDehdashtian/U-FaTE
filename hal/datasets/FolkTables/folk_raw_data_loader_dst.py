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

__all__ = ['FolkTablesRawLoaderDST']



class PrepareFolkTables:
    def __init__(self, opts):
        self.opts = opts
                
    def load_data(self):
        device = 'cuda' if self.opts.ngpu else 'cpu'

        dataset_path = self.opts.dataset_options["path"]
        sensitive_attr = self.opts.dataset_options["sensitive_attr"]

        features = torch.from_numpy(np.loadtxt(os.path.join(dataset_path, 'features.out'))).int()
        labels = torch.from_numpy(np.loadtxt(os.path.join(dataset_path, 'label.out')))
        if sensitive_attr.lower() == 'race':
            sensitive = torch.from_numpy(np.loadtxt(os.path.join(dataset_path, 'group_race.out')))
            # sensitive: 1 to 9 -> 0 to 8
            sensitive -= 1
        elif sensitive_attr.lower() == 'age':
            sensitive = torch.from_numpy(np.loadtxt(os.path.join(dataset_path, 'group.out'))).unsqueeze(1)

        # import pdb; pdb.set_trace()

        # Combine Employment classes from 7 classes to 4 classes :: 0, 1, {2,3,4,5}, 6 
        if labels.max() == 6:
            labels[labels == 3] = 2 
            labels[labels == 4] = 2 
            labels[labels == 5] = 2 
            
            labels[labels == 6] = 3

        # Spliting data
        idx = torch.arange(len(features))
        
        if ('train_size' in self.opts.dataset_options.keys() and not self.opts.dataset_options["train_size"] == 1) or not 'train_size' in self.opts.dataset_options.keys():
            if not 'train_size' in self.opts.dataset_options.keys():
                self.opts.dataset_options["train_size"] = 0.7

            idx_train, idx_val_test = train_test_split(idx, train_size=self.opts.dataset_options["train_size"])
            # Using custom ratio between validation and test sets
            if "TestVal_ratio" in self.opts.dataset_options.keys():
                TestVal_ratio = self.opts.dataset_options['TestVal_ratio']
            else:
                TestVal_ratio = 0.5

            idx_test, idx_val = idx_val_test[:int(len(idx_val_test) * TestVal_ratio)], idx_val_test[int(len(idx_val_test) * TestVal_ratio):] 

            # Combining train and val data to make DST
            idx_train = torch.cat((idx_train, idx_val), 0)

        else:
            idx_train = idx
            idx_val, idx_test = torch.Tensor(), torch.Tensor()


        data = dict()

        #----------------------------------- Print Data Imbalance --------------------------------------------#
        txt = str()
        txt += '********************\n'
        txt += 'Data Imbalance \n'
        max_y = dict()
        max_s = dict()

        for split, idx in zip(['train', 'val', 'test'], [idx_train, idx_val, idx_test]):
            if not len(idx) == 0:
                txt += f'{split.capitalize()}: '
                y_uniques, y_counts = torch.unique(labels[idx.to(int)], return_counts=True)
                s_uniques, s_counts = torch.unique(sensitive[idx.to(int)], return_counts=True)
                
                max_y[split] = 0
                max_s[split] = 0

                for y, count in zip(y_uniques, y_counts):
                    p = count / sum(y_counts)
                    txt += f'P(y={y}) = {p:.3f}; '
                    
                    if p > max_y[split]:
                        max_y[split] = p

                for s, count in zip(s_uniques, s_counts):
                    p = count / sum(s_counts)
                    # txt += f'P(s={s}) = {p:.3f}; '

                    if p > max_s[split]:
                        max_s[split] = p

                txt += '\n' 
            else:
                max_y[split] = 0 
                max_s[split] = 0

        txt += '********************\n'
        print(txt)
        print('*******************************')
        print('Random Chance:')
        print(f'\tTrain\t,\tVal\t,\tTest')
        print(f"Y\t{max_y['train']:.5f}\t,\t{max_y['val']:.5f}\t,\t{max_y['test']:.5f}")
        print(f"S\t{max_s['train']:.5f}\t,\t{max_s['val']:.5f}\t,\t{max_s['test']:.5f}")
        print(f"#\t{len(idx_train)}\t,\t{len(idx_val)}\t,\t{len(idx_test)}")
        print('*******************************\n')

        self.whole_data = dict()
        self.whole_data['x'] = features
        self.whole_data['y'] = labels
        self.whole_data['s'] = sensitive

        data = dict()
        data['train']  = {'x': features[idx_train],                     'y': labels[idx_train],                     's': sensitive[idx_train]}
        data['val']    = {'x': features[idx_val.numpy().tolist()],      'y': labels[idx_val.numpy().tolist()],      's': sensitive[idx_val.numpy().tolist()]}
        data['test']   = {'x': features[idx_test.numpy().tolist()],     'y': labels[idx_test.numpy().tolist()],     's': sensitive[idx_test.numpy().tolist()]}
        
        print(len(idx_train))
        return data

class FolkDataLoader:
    def __init__(self, data, opts):
        self.opts = opts
        self.x      = data['x']
        self.y      = data['y']
        self.s      = data['s']

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index): 
        x       = self.x[index].int()
        y       = self.y[index].long()
        s       = self.s[index].squeeze().long()
        return x, y, s

 
class FolkTablesRawLoaderDST(pl.LightningDataModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        if opts.ngpu == 0:
            self.pin_memory = False
        else:
            self.pin_memory = True

        pre = PrepareFolkTables(opts)
        self.data = pre.load_data()

    def train_dataloader(self):
        dataset = FolkDataLoader(self.data['train'], self.opts)

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.opts.batch_size_train,
            shuffle=True,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory
        )

        return loader

    def val_dataloader(self):
        dataset = FolkDataLoader(self.data['val'], self.opts)

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.opts.batch_size_test,
            shuffle=False,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory
        )
        
        return loader

    def test_dataloader(self):
        dataset = FolkDataLoader(self.data['test'], self.opts)

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
        x = self.data['train']['x'][idx_sampled]
        y = self.data['train']['y'][idx_sampled]
        s = self.data['train']['s'][idx_sampled]
        
        return x.cuda(), y.cuda(), s.cuda()


