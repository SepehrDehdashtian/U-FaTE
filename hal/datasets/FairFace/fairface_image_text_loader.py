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

__all__ = ['FairFaceImageTextLoader']



class PrepareFairFace:
    def __init__(self, opts):
        self.opts = opts
        if opts.dataset_options['transform']:
            self.transform = transforms.Compose([
                    transforms.Resize((self.opts.dataset_options['resolution_high'], self.opts.dataset_options['resolution_wide'])),
                    transforms.ToTensor(),
                ])
        else:
            self.transform = None

    def to_uint8(self, img):
        img = 255 * (img + 1) / 2 
        return img.to(dtype=torch.uint8)

    def from_unit8(self, img):
        return 2 * (img.to(dtype=torch.float32) / 255) - 1

    
    def image_loader(self, img_name):
        path = os.path.join(self.dataset_path, self.images_path, img_name)
        if self.transform is None:
            if not ('uint8' in self.opts.dataset_options['dataset'].keys() and self.opts.dataset_options['dataset']['uint8']):
                return Image.open(path).convert('RGB')
            else:
                img = Image.open(path).convert('RGB')
                img = self.to_uint8(img)
                return img
                
        else:
            if not ('uint8' in self.opts.dataset_options['dataset'].keys() and self.opts.dataset_options['dataset']['uint8']):
                return self.transform(Image.open(path).convert('RGB'))
            else:
                img = self.transform(Image.open(path).convert('RGB'))
                img = self.to_uint8(img)
                return img

    
    def load_data(self):
        device = 'cuda' if self.opts.ngpu else 'cpu'

        self.dataset_path    = self.opts.dataset_options["path"]
        self.images_path     = self.opts.dataset_options["imgs"]
        train_attrs_filename       = self.opts.dataset_options["train_attr_filename"]
        val_attrs_filename       = self.opts.dataset_options["val_attr_filename"]

        target_attr          = self.opts.dataset_options["target_attr"]
        sensitive_attr_list       = self.opts.dataset_options["sensitive_attr"]

        if target_attr == 'gender':
            text_prompts = [
                'This is a photo of a female person.',
                'This is a photo of a male person.'
                ]

        train_csv_data = pd.read_csv(os.path.join(self.dataset_path, train_attrs_filename))
        val_csv_data   = pd.read_csv(os.path.join(self.dataset_path, val_attrs_filename))

        # Converting attributes to torch tensors
        train_s_list = list()
        val_s_list = list()
        num_classes_list = list()
        sensDict_list = list()
        for sensitive_attr in sensitive_attr_list:
            if sensitive_attr == 'age':
                raise ValueError(f'{sensitive_attr} is not implemented yet!')
            elif sensitive_attr == 'gender':
                sensDict = {'Female': 0, 'Male': 1}
            elif sensitive_attr == 'race':
                sensDict = {'White': 0, 'Black': 1, 'Indian': 2, 'Latino_Hispanic': 3, 'East Asian': 4, 'Southeast Asian': 5, 'Middle Eastern': 6}
            else:
                raise ValueError(f'{sensitive_attr} is not a in attributes list. Choose one from age, gender, and race')

            train_s_list.append(torch.tensor(list(map(lambda x: sensDict[x], train_csv_data[sensitive_attr].values.tolist()))))#, device=device)
            val_s_list.append(torch.tensor(list(map(lambda x: sensDict[x], val_csv_data[sensitive_attr].values.tolist()))))#, device=device)
            num_classes_list.append(len(sensDict))
            sensDict_list.append(sensDict)
        
        # Combine labels (race and gender) into multiple classes
        train_s = num_classes_list[0] * train_s_list[1] + train_s_list[0]
        val_s = num_classes_list[0] * val_s_list[1] + val_s_list[0]

        if target_attr == 'age':
            raise ValueError(f'{target_attr} is not implemented yet!')
        elif target_attr == 'gender':
            tgtDict = {'Female': 0, 'Male': 1}
        elif target_attr == 'race':
            tgtDict = {'White': 0, 'Black': 1, 'Indian': 2, 'Latino_Hispanic': 3, 'East Asian': 4, 'Southeast Asian': 5, 'Middle Eastern': 6}
        else:
            raise ValueError(f'{target_attr} is not a in attributes list. Choose one from age, gender, and race')

        train_y = torch.tensor(list(map(lambda x: tgtDict[x], train_csv_data[target_attr].values.tolist())))

        val_y = torch.tensor(list(map(lambda x: tgtDict[x], val_csv_data[target_attr].values.tolist())))

        print("Loading images ...")
        tic = time.time()

        # Deciding how to load images (Batch by Batch or all at once)
        self.opts.load_all = not ('loadAll' in self.opts.dataset_options['dataset'].keys() and not self.opts.dataset_options['dataset']['loadAll'])

        if not self.opts.load_all:
            train_images = train_csv_data["file"].values
            val_images = val_csv_data["file"].values
        else:
            val_images = list(map(self.image_loader, tqdm(val_csv_data["file"].values)))
            val_images = torch.stack(val_images)
        
        toc = time.time()
        print(f"Loading Completed in {toc - tic:.3f} seconds\n")
        
        data = dict()

        data['val']   = {'imgs': val_images,   'y': val_y,   's': val_s}
        data['text-Y'] = text_prompts



        return data

class FairFaceDataloader:
    def __init__(self, data, opts):
        self.opts = opts
        self.images = data['imgs']
        self.y      = data['y']
        self.s      = data['s']
        if opts.dataset_options['transform']:
            self.transform = transforms.Compose([
                    transforms.Resize((self.opts.dataset_options['resolution_high'], self.opts.dataset_options['resolution_wide'])),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        else:
            self.transform = None

    def image_loader(self, img_name):
        dataset_path    = self.opts.dataset_options["path"]
        images_path     = self.opts.dataset_options["imgs"]

        path = os.path.join(dataset_path, images_path, img_name)
        if self.transform is None:
            return Image.open(path).convert('RGB')
        else:
            return self.transform(Image.open(path).convert('RGB'))
    
    def from_unit8(self, img):
        return 2 * (img.to(dtype=torch.float32) / 255) - 1

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        if not self.opts.load_all:
            # Load batch by batch
            image_name  = self.images[index]
            image       = self.image_loader(image_name)
        else:
            # Load All
            if not ('uint8' in self.opts.dataset_options['dataset'].keys() and self.opts.dataset_options['dataset']['uint8']):
                print(index)
                image   = self.images[index]
            else:
                image   = self.from_unit8(self.images[index])    

        y       = self.y[index].long()
        s       = self.s[index]
        return image, y, s


class FairFaceImageTextLoader(pl.LightningDataModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.pin_memory = False
        
        pre = PrepareFairFace(opts)
        self.data = pre.load_data()

        self.text_y = self.data['text-Y']

        if opts.dataset_options['transform']:
            self.transform = transforms.Compose([
                    transforms.Resize((self.opts.dataset_options['resolution_high'], self.opts.dataset_options['resolution_wide'])),
                    transforms.ToTensor(),
                ])
        else:
            self.transform = None

    def train_dataloader(self):
        dataset = FairFaceDataloader(self.data['train'], self.opts)
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.opts.batch_size_train,
            shuffle=True,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory
        )

        return loader

    def val_dataloader(self):
        dataset = FairFaceDataloader(self.data['val'], self.opts)

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.opts.batch_size_test,
            shuffle=False,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory
        )
        
        return loader

    def test_dataloader(self):
        dataset = FairFaceDataloader(self.data['test'], self.opts)

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.opts.batch_size_test,
            shuffle=False,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory
        )
        return loader

    def image_loader(self, img_name):
        dataset_path    = self.opts.dataset_options["path"]
        images_path     = self.opts.dataset_options["imgs"]

        path = os.path.join(dataset_path, images_path, img_name)
        if self.transform is None:
            return Image.open(path).convert('RGB')
        else:
            return self.transform(Image.open(path).convert('RGB'))

    def train_kernel_dataloader(self):
        idx_sampled = torch.randperm(len(self.data['train']['y']))[:self.opts.dataset_options["kernel_numSamples"]]
        if not ('uint8' in self.opts.dataset_options['dataset'].keys() and self.opts.dataset_options['dataset']['uint8']):
            if self.opts.load_all:
                images = self.data['train']['imgs'][idx_sampled].cuda()
            else:
                images = list(map(self.image_loader, self.data['train']['imgs'][idx_sampled]))
                images = torch.stack(images).cuda()
        else:    
            images = list(map(self.from_unit8, torch.Tensor(self.data['train']['imgs'])[idx_sampled]))
            images = torch.stack(images).cuda()

        y      = self.data['train']['y'][idx_sampled].cuda()
        s      = self.data['train']['s'][idx_sampled].cuda()
        return images, y, s


