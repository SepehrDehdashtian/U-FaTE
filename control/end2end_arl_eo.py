# adversarial_representation_learning.py

import torch
from collections import OrderedDict

import hal.losses as losses
from control.base import BaseClass

__all__ = ['EndToEndARLEO']


class EndToEndARLEO(BaseClass):
    def __init__(self, opts, dataloader):
        super().__init__(opts, dataloader)

    def training_step(self, batch, batch_idx):
        x, y, s = batch
        
        mask = (y == 1)


        features = self.feature_extractor(x)

        z = self.encoder(features)

        y_hat = self.target(z)

        loss_tgt = self.criterion['target'](y_hat, y)
        
        opt = self.optimizers()

        if self.current_epoch >= self.hparams.control_epoch:
            s_hat = self.adversary(z[mask])
            
            if self.hparams.loss_type['adversary'] == 'Classification':
                loss_ctl = self.criterion['adversary'](s_hat.squeeze(), s[mask].squeeze())
            else:
                loss_ctl = self.criterion['adversary'](s_hat.squeeze(), s[mask].float().squeeze())
        else:
            loss_ctl = torch.zeros_like(loss_tgt)

        loss = loss_tgt - self.hparams.tau / (1 - self.hparams.tau) * loss_ctl

        opt[0].zero_grad()
        opt[1].zero_grad()
        opt[2].zero_grad()
        self.manual_backward(loss)
        opt[0].step()
        opt[1].step()
        opt[2].step()
        self.used_optimizers[0] = True
        self.used_optimizers[1] = True
        self.used_optimizers[2] = True

        
        if self.current_epoch >= self.hparams.control_epoch:
            for _ in range(self.hparams.num_adv_train_iters):
                features = self.feature_extractor(x)
                z = self.encoder(features)
                s_hat = self.adversary(z[mask])
                
                if self.hparams.loss_type['adversary'] == 'Classification':
                    loss_ctl = self.criterion['adversary'](s_hat.squeeze(), s[mask].squeeze())
                else:
                    loss_ctl = self.criterion['adversary'](s_hat.squeeze(), s[mask].float().squeeze())

                opt[3].zero_grad()
                self.manual_backward(loss_ctl)
                opt[3].step()
                self.used_optimizers[3] = True
                
        else:
            s_hat = self.adversary(z)



        output = OrderedDict({
            'loss': loss.detach(),
            'loss_tgt': {'value': loss_tgt.detach(), 'numel': len(x)},
            'loss_ctl': {'value': loss_ctl.detach(), 'numel': len(x)},
            'y_hat': y_hat.detach(),
            'x' : features.detach(),
            'z' : z.detach(),
            'y': y.detach(),
            's': s.detach(),
            's_hat': s_hat.detach()
        })

        return output
        

    def validation_step(self, batch, _):
        x, y, s = batch


        mask = (y == 1)

        features = self.feature_extractor(x)

        z = self.encoder(features)

        y_hat = self.target(z)
        loss_tgt = self.criterion['target'](y_hat, y)

        s_hat = self.adversary(z)

        loss_ctl = self.criterion['adversary'](s_hat.squeeze(), s.squeeze())

        output = OrderedDict({
            'loss': loss_tgt.detach(),
            'x' : features.detach(),
            's': s,
            's_hat': s_hat,
            'y_hat': y_hat,
            'y': y,
            'z' : z.detach(),
            'loss_tgt': {'value': loss_tgt.detach(), 'numel': len(x)},
            'loss_ctl': {'value': loss_ctl.detach(), 'numel': len(x)},
        })

        return output

    def test_step(self, batch, _):
        x, y, s = batch

        features = self.feature_extractor(x)

        z = self.encoder(features)

        y_hat = self.target(z)
        loss_tgt = self.criterion['target'](y_hat, y)

        s_hat = self.adversary(z)
        loss_ctl = self.criterion['adversary'](s_hat, s)


        output = OrderedDict({
            'loss': loss_tgt.detach(),
            # 'x': x,
            's': s,
            's_hat': s_hat,
            'y_hat': y_hat,
            'y': y,
            'z' : z.detach(),
            'loss_tgt': {'value': loss_tgt.detach(), 'numel': len(x)},
            'loss_ctl': {'value': loss_ctl.detach(), 'numel': len(x)},
        })

        return output

