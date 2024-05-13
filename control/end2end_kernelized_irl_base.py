import torch
from collections import OrderedDict

import hal.kernels as kernels
import hal.models as models
import hal.utils.misc as misc
from control.base import BaseClass
import control.build_kernel as build_kernel

from tqdm import tqdm

__all__ = ['EndToEndKIRL']

class EndToEndKIRL(BaseClass):
    def __init__(self, opts, dataloader):
        super().__init__(opts, dataloader)

        self.rff_flag = self.hparams.rff_flag
        self.kernel_x       = getattr(kernels, self.hparams.kernel_x)(**self.hparams.kernel_x_options)
        self.kernel_y       = getattr(kernels, self.hparams.kernel_y)(**self.hparams.kernel_y_options)
        self.kernel_s       = getattr(kernels, self.hparams.kernel_s)(**self.hparams.kernel_s_options)

        self.dataloader = dataloader

        self.model_device = 'cuda' if self.hparams.ngpu else 'cpu'

        self.compute_kernel(init=True)

    def compute_kernel0(self, features=None, Y=None, S=None, init=False):
        self.encoder = None

        if init:
            print('Initializing the kernel ...', end='\r')
        else:
            print('Computing the kernel ...', end='\r')

        self.feature_extractor = self.feature_extractor.to(device=self.model_device).eval()
        with torch.no_grad():
            if features is None or Y is None or S is None:
                X, Y, S = self.dataloader.train_kernel_dataloader()
                features = self.feature_extractor(X)


        self.encoder = getattr(build_kernel, self.hparams.build_kernel)(self, features, Y, S)
        self.feature_extractor = self.feature_extractor.to(device=self.model_device).train()

        if init:
            print('Initializing the kernel is done!', end='\r')
        else:
            print('Computing the kernel is done!', end='\r')


    def compute_kernel(self, features=None, Y=None, S=None, init=False):
        self.encoder = None

        if init:
            print('Initializing the kernel ...', end='\r')
        else:
            print('Computing the kernel ...', end='\r')

        self.feature_extractor = self.feature_extractor.to(device=self.model_device).eval()
        with torch.no_grad():
            if features is None or Y is None or S is None:
                data = self.dataloader.train_kernel_dataloader()
                if type(data) is torch.utils.data.dataloader.DataLoader:
                    features = list()
                    Y        = list()
                    S        = list() 
                    for batch in tqdm(data):
                        x, y, s = batch
                        features.append(self.feature_extractor(x.to(device=self.model_device)))
                        Y.append(y)
                        S.append(s)
                    features = torch.cat(features, dim=0)
                    Y        = torch.cat(Y, dim=0)
                    S        = torch.cat(S, dim=0)

                else:
                    X, Y, S = data
                    features = self.feature_extractor(X)


        self.encoder = getattr(build_kernel, self.hparams.build_kernel)(self, features, Y, S)
        self.feature_extractor = self.feature_extractor.to(device=self.model_device).train()

        if init:
            print('Initializing the kernel is done!', end='\r')
        else:
            print('Computing the kernel is done!', end='\r')


    def training_step(self, batch, batch_idx):     
        x, y, s = batch

        opt = self.optimizers()


        if self.current_epoch <= self.hparams.pretrain_epochs:
            features = self.feature_extractor(x)

            z = self.encoder(features)

            if 'DEP_ZS' in self.hparams.loss_type.keys() and self.hparams.loss_type['DEP_ZS'] in ['DEPLoss_old', 'EOODEPLossBinary', 'EODEPLoss', 'EODEPLossLinear', 'RFFEODEPLoss']:
                dep_zy = self.criterion['DEP_ZY'](z, y, self.norm2_b_y)
            else:
                dep_zy = self.criterion['DEP_ZY'](z, y)

            if 'DEP_ZS' in self.hparams.loss_type.keys() and self.hparams.loss_type['DEP_ZS'] in ['EODEPLoss', 'EOODEPLossBinary', 'EODEPLossLinear', 'RFFEODEPLoss']:
                dep_zs = self.criterion['DEP_ZS'](z, s, y, self.norm2_b_s)
            elif 'DEP_ZS' in self.hparams.loss_type.keys() and self.hparams.loss_type['DEP_ZS'] in ['DEPLoss_old']:
                dep_zs = self.criterion['DEP_ZS'](z, s, self.norm2_b_s)
            else:
                dep_zs = self.criterion['DEP_ZS'](z, s)

            loss = - dep_zy + self.hparams.tau / (1 - self.hparams.tau) * dep_zs
            loss_tgt = torch.zeros_like(loss)

            if y.max() == 1:
                y_hat = torch.Tensor(len(y) * [[1,0]]).to(device=y.device)
            elif y.max() == 3:
                y_hat = torch.Tensor(len(y) * [[0, 1, 0, 0]]).to(device=y.device)

            # turn off require_grad in the encoder
            for p in self.encoder.parameters():
                p.requires_grad = False

            opt[0].zero_grad()
            self.manual_backward(loss)
            opt[0].step()
            self.used_optimizers[0] = True


            output = OrderedDict({
                'loss'      : loss.detach(),
                'loss_tgt'  : loss_tgt.detach(),
                'dep_zy'    : dep_zy.detach(),
                'dep_zs'    : dep_zs.detach(),
                'x'         : features.detach(),
                'z'         : z.detach(),
                'y'         : y.detach(),
                'y_hat'     : y_hat.detach(),
                's'         : s.detach(),
            })
            return output

        else:
            # turn off require_grad in the encoder and feature extractor
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

            features = self.feature_extractor(x)

            z = self.encoder(features)

            y_hat = self.target(z)

            loss_tgt = self.criterion['target'](y_hat, y)

            if 'DEP_ZS' in self.hparams.loss_type.keys() and self.hparams.loss_type['DEP_ZS'] in ['DEPLoss_old', 'EOODEPLossBinary', 'EODEPLoss', 'EODEPLossLinear', 'RFFEODEPLoss']:
                dep_zy = self.criterion['DEP_ZY'](z, y, self.norm2_b_y)
            else:
                dep_zy = self.criterion['DEP_ZY'](z, y)


            if 'DEP_ZS' in self.hparams.loss_type.keys() and self.hparams.loss_type['DEP_ZS'] in ['EODEPLoss', 'EOODEPLossBinary', 'EODEPLossLinear', 'RFFEODEPLoss']:
                dep_zs = self.criterion['DEP_ZS'](z, s, y, self.norm2_b_s)
            elif 'DEP_ZS' in self.hparams.loss_type.keys() and self.hparams.loss_type['DEP_ZS'] in ['DEPLoss_old']:
                dep_zs = self.criterion['DEP_ZS'](z, s, self.norm2_b_s)
            else:
                dep_zs = self.criterion['DEP_ZS'](z, s)

            loss = - dep_zy + self.hparams.tau / (1 - self.hparams.tau) * dep_zs

            opt[1].zero_grad()
            self.manual_backward(loss_tgt)
            opt[1].step()
            self.used_optimizers[1] = True

            output = OrderedDict({
                'loss'      : loss.detach(),
                'loss_tgt'  : loss_tgt.detach(),
                'dep_zy'    : dep_zy.detach(),
                'dep_zs'    : dep_zs.detach(),
                'x'         : features.detach(),
                'z'         : z.detach(),
                'y'         : y.detach(),
                'y_hat'     : y_hat.detach(),
                's'         : s.detach(),
            })
            return output


    def validation_step(self, batch, _):
        x, y, s = batch

        if self.current_epoch <= self.hparams.pretrain_epochs:
            features = self.feature_extractor(x)

            z = self.encoder(features)


            if 'DEP_ZS' in self.hparams.loss_type.keys() and self.hparams.loss_type['DEP_ZS'] in ['DEPLoss_old', 'EOODEPLossBinary', 'EODEPLoss', 'EODEPLossLinear', 'RFFEODEPLoss']:
                dep_zy = self.criterion['DEP_ZY'](z, y, self.norm2_b_y)
            else:
                dep_zy = self.criterion['DEP_ZY'](z, y)

            if 'DEP_ZS' in self.hparams.loss_type.keys() and self.hparams.loss_type['DEP_ZS'] in ['EODEPLoss', 'EOODEPLossBinary', 'EODEPLossLinear', 'RFFEODEPLoss']:
                dep_zs = self.criterion['DEP_ZS'](z, s, y, self.norm2_b_s)
            elif 'DEP_ZS' in self.hparams.loss_type.keys() and self.hparams.loss_type['DEP_ZS'] in ['DEPLoss_old']:
                dep_zs = self.criterion['DEP_ZS'](z, s, self.norm2_b_s)
            else:
                dep_zs = self.criterion['DEP_ZS'](z, s)

            loss = - dep_zy + self.hparams.tau / (1 - self.hparams.tau) * dep_zs
            loss_tgt = torch.zeros_like(loss)
            
            if y.max() == 1:
                y_hat = torch.Tensor(len(y) * [[1,0]]).to(device=y.device)
            elif y.max() == 3:
                y_hat = torch.Tensor(len(y) * [[0, 1, 0, 0]]).to(device=y.device)

            output = OrderedDict({
                'loss'  : loss.detach(),
                'loss_tgt'  : loss_tgt.detach(),
                'dep_zy': dep_zy.detach(),
                'dep_zs': dep_zs.detach(),
                'x'     : features.detach(),
                's'     : s,
                'y'     : y,
                'y_hat' : y_hat.detach(),
                'z'     : z.detach(),
            })
            return output

        else:
            # turn off require_grad in the encoder and feature extractor
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

            features = self.feature_extractor(x)

            z = self.encoder(features)

            y_hat = self.target(z)

            loss_tgt = self.criterion['target'](y_hat, y)

            if 'DEP_ZS' in self.hparams.loss_type.keys() and self.hparams.loss_type['DEP_ZS'] in ['DEPLoss_old', 'EOODEPLossBinary', 'EODEPLoss', 'EODEPLossLinear', 'RFFEODEPLoss']:
                dep_zy = self.criterion['DEP_ZY'](z, y, self.norm2_b_y)
            else:
                dep_zy = self.criterion['DEP_ZY'](z, y)


            if 'DEP_ZS' in self.hparams.loss_type.keys() and self.hparams.loss_type['DEP_ZS'] in ['EODEPLoss', 'EOODEPLossBinary', 'EODEPLossLinear', 'RFFEODEPLoss']:
                dep_zs = self.criterion['DEP_ZS'](z, s, y, self.norm2_b_s)
            elif 'DEP_ZS' in self.hparams.loss_type.keys() and self.hparams.loss_type['DEP_ZS'] in ['DEPLoss_old']:
                dep_zs = self.criterion['DEP_ZS'](z, s, self.norm2_b_s)
            else:
                dep_zs = self.criterion['DEP_ZS'](z, s)
                
            loss = - dep_zy + self.hparams.tau / (1 - self.hparams.tau) * dep_zs


            output = OrderedDict({
                'loss'      : loss.detach(),
                'loss_tgt'  : loss_tgt.detach(),
                'dep_zy'    : dep_zy.detach(),
                'dep_zs'    : dep_zs.detach(),
                'x'         : features.detach(),
                'z'         : z.detach(),
                'y'         : y.detach(),
                'y_hat'     : y_hat.detach(),
                's'         : s.detach(),
            })
            return output

    def test_step(self, batch, _):
        x, y, s = batch

        features = self.feature_extractor(x)

        z = self.encoder(features)
        
        dep_zy = self.criterion['DEP_ZY'](z, y)
        dep_zs = self.criterion['DEP_ZS'](z, s)

        loss = - dep_zy + self.hparams.tau / (1 - self.hparams.tau) * dep_zs

        output = OrderedDict({
            'loss'  : loss.detach(),
            'dep_zy': dep_zy.detach(),
            'dep_zs': dep_zs.detach(),
            'x'     : features.detach(),
            's'     : s,
            'y'     : y,
            'z'     : z.detach(),
        })
        return output

    def format_y_onehot(self, y):
        if len(y.shape) == 1:
            y = y.unsqueeze(1)
        y_onehot = torch.zeros(y.size(0), self.hparams.loss_options['DEP_ZY']['onehot_num_classes'], device=y.device).scatter_(1, y.type(torch.int64), 1)
        return y_onehot
        
        
    def format_s_onehot(self, s):
        # int -> one-hot
        if len(s.shape) > 2:
            s = s.squeeze(-1)
        elif len(s.shape) == 1:
            s = s.unsqueeze(1)

        s_onehot = torch.zeros(s.size(0), self.hparams.loss_options['DEP_ZS']['onehot_num_classes'], device=s.device).scatter_(1, s.long(), 1)
        return s_onehot
