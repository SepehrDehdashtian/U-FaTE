# tmlr.py

import torch
from collections import OrderedDict

import hal.kernels as kernels
import hal.models as models
import hal.utils.misc as misc
from control.base import BaseClass
import control.build_kernel as build_kernel
from tqdm import tqdm
import numpy as np


__all__ = ['TMLR']

class TMLR(BaseClass):

    def __init__(self, opts, dataloader):
        super().__init__(opts, dataloader)

        self.rff_flag = self.hparams.rff_flag
        self.kernel_x       = getattr(kernels, self.hparams.kernel_x)(**self.hparams.kernel_x_options)
        self.kernel_y       = getattr(kernels, self.hparams.kernel_y)(**self.hparams.kernel_y_options)
        self.kernel_s       = getattr(kernels, self.hparams.kernel_s)(**self.hparams.kernel_s_options)


        self.dataloader = dataloader

        self.model_device = 'cuda' if self.hparams.ngpu else 'cpu'
        # Initializing the kernel
        print('Initializing the kernel ...')

        self.compute_kernel(init=True)

    def save_features(self):
        # Loading data
        data = self.dataloader.data

        checkpoint = torch.load(self.hparams.pretrained_checkpoint)
        state_dict = checkpoint['state_dict']
        new_state_dict = OrderedDict()


        for k, v in state_dict.items():
            model_name = k.split('.')[0]
            if model_name == 'feature_extractor':
                new_state_dict[k.split('feature_extractor.')[-1]] = v

        self.feature_extractor.load_state_dict(new_state_dict)
        self.feature_extractor.eval()

        self.feature_extractor = self.feature_extractor.to(device=self.model_device)
        self.turn_off_grad(self.feature_extractor)
        with torch.no_grad():
            self.feature_extractor.eval()
            for split in ['train', 'val', 'test']:
                print(split)
                x = data[split]['imgs'].cuda() 
                y = data[split]['y'] 
                s = data[split]['s']

                feat = list()
                for i in range(len(x)//1000):
                    if i == (len(x) // 1000) - 1:
                        feat.append(self.feature_extractor(x[i*1000:]))
                    else:
                        feat.append(self.feature_extractor(x[i*1000:(i+1)*1000]))

                feat = torch.cat(feat, dim=0)
                np.savetxt('/research/hal-datastage/datasets/processed/CelebA/Heavy_Makeup' + f'/z_{split}.out', feat.cpu().numpy(), fmt='%10.5f')
                np.savetxt('/research/hal-datastage/datasets/processed/CelebA/Heavy_Makeup' + f'/y_{split}.out', y.cpu().numpy(), fmt='%10.5f')
                np.savetxt('/research/hal-datastage/datasets/processed/CelebA/Heavy_Makeup' + f'/s_{split}.out', s.cpu().numpy(), fmt='%10.5f')   
        exit()

    def save_features_folk(self):
        # Loading data
        data = self.dataloader.data

        checkpoint = torch.load(self.hparams.pretrained_checkpoint)
        state_dict = checkpoint['state_dict']
        new_state_dict = OrderedDict()


        for k, v in state_dict.items():
            model_name = k.split('.')[0]
            if model_name == 'feature_extractor':
                new_state_dict[k.split('feature_extractor.')[-1]] = v

        print(new_state_dict)
        self.feature_extractor.load_state_dict(new_state_dict)
        self.feature_extractor.eval()

        self.feature_extractor = self.feature_extractor.to(device=self.model_device)
        self.turn_off_grad(self.feature_extractor)
        with torch.no_grad():
            self.feature_extractor.eval()
            feat = list()
            y_list = list()
            s_list = list()
            for split in ['train', 'val', 'test']:
                print(split)
                x = data[split]['x'].cuda() 
                y = data[split]['y'] 
                s = data[split]['s']

                feat.append(self.feature_extractor(x))
                y_list.append(y)
                s_list.append(s)

            feat = torch.cat(feat, dim=0)
            y = torch.cat(y_list, dim=0)
            s = torch.cat(s_list, dim=0)
            np.savetxt('/research/hal-datastage/datasets/processed/folktables/states/WA/Age/Emp/' + '/emb_features.out', feat.cpu().numpy(), fmt='%10.5f')
            np.savetxt('/research/hal-datastage/datasets/processed/folktables/states/WA/Age/Emp/' + '/emb_label.out', y.cpu().numpy(), fmt='%10.5f')
            np.savetxt('/research/hal-datastage/datasets/processed/folktables/states/WA/Age/Emp/' + '/emb_group.out', s.cpu().numpy(), fmt='%10.5f')   
        exit()
                

    def save_predicts_folk(self):
        # Loading data
        data = self.dataloader.data

        checkpoint = torch.load(self.hparams.pretrained_checkpoint)
        state_dict = checkpoint['state_dict']
        new_state_dict = OrderedDict()


        for k, v in state_dict.items():
            model_name = k.split('.')[0]
            if model_name == 'feature_extractor':
                new_state_dict[k.split('feature_extractor.')[-1]] = v

        self.feature_extractor.load_state_dict(new_state_dict)
        self.feature_extractor.eval()

        self.feature_extractor = self.feature_extractor.to(device=self.model_device)
        self.turn_off_grad(self.feature_extractor)
        with torch.no_grad():
            self.feature_extractor.eval()
            feat = list()
            y_list = list()
            s_list = list()
            for split in ['train', 'val', 'test']:
                print(split)
                x = data[split]['x'].cuda() 
                y = data[split]['y'] 
                s = data[split]['s']

                feat.append(self.feature_extractor(x))
                y_list.append(y)
                s_list.append(s)

            feat = torch.cat(feat, dim=0)
            y = torch.cat(y_list, dim=0)
            s = torch.cat(s_list, dim=0)
            np.savetxt('/research/hal-datastage/datasets/processed/folktables/states/WA/Age/Emp/' + '/emb_features.out', feat.cpu().numpy(), fmt='%10.5f')
            np.savetxt('/research/hal-datastage/datasets/processed/folktables/states/WA/Age/Emp/' + '/emb_label.out', y.cpu().numpy(), fmt='%10.5f')
            np.savetxt('/research/hal-datastage/datasets/processed/folktables/states/WA/Age/Emp/' + '/emb_group.out', s.cpu().numpy(), fmt='%10.5f')   
        exit()
                

    def compute_kernel_iter(self, features=None, Y=None, S=None, init=False):
        self.encoder = None

        if init:
            print('Initializing the kernel ...', end='\r')
        else:
            print('Computing the kernel ...', end='\r')


        checkpoint = torch.load(self.hparams.pretrained_checkpoint)
        state_dict = checkpoint['state_dict']
        new_state_dict = OrderedDict()


        for k, v in state_dict.items():
            model_name = k.split('.')[0]
            if model_name == 'feature_extractor':
                new_state_dict[k.split('feature_extractor.')[-1]] = v
        self.feature_extractor.load_state_dict(new_state_dict)
        self.feature_extractor.eval()

        self.feature_extractor = self.feature_extractor.to(device=self.model_device).eval()

        with torch.no_grad():
            if features is None or Y is None or S is None:
                features_list = list()
                Y_list = list()
                S_list = list()
                Y = list()
                S = list()
                for X, Y, S in tqdm(self.dataloader.train_kernel_dataloader()): 
                    X = X.to(device=self.model_device)   

                    features_list.append(self.feature_extractor(X))
                    Y_list.append(Y)
                    S_list.append(S)

                features = torch.cat(features_list, dim=0)
                Y = torch.cat(Y_list, dim=0).to(device=self.model_device)  
                S = torch.cat(S_list, dim=0).to(device=self.model_device)  

        self.encoder = getattr(build_kernel, self.hparams.build_kernel)(self, features, Y, S)

        if init:
            print('Initializing the kernel is done!', end='\r')
        else:
            print('Computing the kernel is done!', end='\r')

    def compute_kernel(self, features=None, Y=None, S=None, init=False):
        self.encoder = None
        if init:
            print('Initializing the kernel ...')
        else:
            print('Computing the kernel ...')

        features, Y, S = self.dataloader.train_kernel_dataloader()

        self.encoder = getattr(build_kernel, self.hparams.build_kernel)(self, features, Y, S)

        if init:
            print('Initializing the kernel is done!')
        else:
            print('Computing the kernel is done!')



    def training_step(self, batch, batch_idx, *args, **kwargs):     
        x, y, s = batch
        
        opt = self.optimizers()
        with torch.no_grad():
            z = self.encoder(x) # Kernel

        y_hat = self.target(z)

        loss_tgt = self.criterion['target'](y_hat, y)

        opt.zero_grad()
        self.manual_backward(loss_tgt)
        opt.step()
        self.used_optimizers[0] = True

        output = OrderedDict({
            'loss': loss_tgt.detach(),
            'loss_tgt': {'value': loss_tgt.detach(), 'numel': len(x)},
            'y_hat': y_hat.detach(),
            'x': x,
            'z': z.detach(),
            'y': y.detach(),
            's': s.detach(),
        })
        return output


    def validation_step(self, batch, _):
        x, y, s = batch

        z = self.encoder(x)

        y_hat = self.target(z)

        loss_tgt = self.criterion['target'](y_hat, y)

        output = OrderedDict({
            'loss': loss_tgt.detach(),
            'x': x,
            's': s,
            'y_hat': y_hat,
            'y': y,
            'z': z.detach(),
            'loss_tgt': {'value': loss_tgt.detach(), 'numel': len(x)},
        })
        return output

    def test_step(self, batch, _):
        x, y, s = batch

        z = self.encoder(x)

        y_hat = self.target(z)

        loss_tgt = self.criterion['target'](y_hat, y)

        output = OrderedDict({
            'loss': loss_tgt.detach(),
            'x': x,
            's': s,
            'y_hat': y_hat,
            'y': y,
            'z': z.detach(),
            'loss_tgt': {'value': loss_tgt.detach(), 'numel': len(x)},
        })
        return output

    def format_y_onehot(self, y):
        if len(y.shape) == 1:
            y = y.unsqueeze(1)
        y_onehot = torch.zeros(y.size(0), self.hparams.model_options['target']['nout'], device=y.device).scatter_(1, y.type(torch.int64), 1)
        return y_onehot
        
        
    def format_s_onehot(self, s):
        # int -> one-hot
        if len(s.shape) > 2:
            s = s.squeeze(-1)
        elif len(s.shape) == 1:
            s = s.unsqueeze(1)
            
        s_onehot = torch.zeros(s.size(0), self.hparams.metric_control_options['SP']['num_s_classes'], device=s.device).scatter_(1, s.long(), 1)
        return s_onehot
