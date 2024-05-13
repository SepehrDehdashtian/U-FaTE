import torch
from collections import OrderedDict

import hal.losses as losses
from control.base import BaseClass

__all__ = ['EndToEndOptNet']


class EndToEndOptNet(BaseClass):
    def __init__(self, opts, dataloader):
        super().__init__(opts, dataloader)

    def training_step(self, batch, batch_idx):
        x, y, s = batch
        
        opt = self.optimizers()

        if self.current_epoch < self.hparams.control_epoch_optnet:
            features = self.feature_extractor(x)
            z = self.encoder(features)

            loss_ctl, proj_s, proj_y = self.criterion['adversary'](z, s, y, self.hparams.tau, self.hparams.gamma, 1.0)

            opt[0].zero_grad()
            opt[1].zero_grad()
            self.manual_backward(loss_ctl)
            opt[0].step()
            opt[1].step()
            self.used_optimizers[0] = True
            self.used_optimizers[1] = True



            y_hat = torch.Tensor(len(y) * [[1,0]]).to(device=y.device)

            output = OrderedDict({
                'loss': loss_ctl.detach(),
                'loss_tgt': {'value': torch.Tensor([0]), 'numel': len(x)},
                'loss_ctl': {'value': loss_ctl.detach(), 'numel': len(x)},
                'proj_s'  : proj_s,
                'proj_y'  : proj_y,
                'y_hat': y_hat.detach(),
                'x' : features.detach(),
                'z' : z.detach(),
                'y': y.detach(),
                's': s.detach()
            })

        else:
            self.turn_off_grad(self.feature_extractor)
            self.turn_off_grad(self.encoder)
            self.feature_extractor.eval()
            self.encoder.eval()

            with torch.no_grad():
                features = self.feature_extractor(x)
                z = self.encoder(features)
            
            y_hat = self.target(z)

            loss_tgt = self.criterion['target'](y_hat, y)

            opt[2].zero_grad()
            self.manual_backward(loss_tgt)
            opt[2].step()
            self.used_optimizers[2] = True


            output = OrderedDict({
                'loss': loss_tgt.detach(),
                'loss_tgt': {'value': loss_tgt.detach(), 'numel': len(x)},
                'loss_ctl': {'value': torch.Tensor([0]), 'numel': len(x)},
                'proj_s'  : torch.Tensor([0]),
                'proj_y'  : torch.Tensor([0]),
                'y_hat': y_hat.detach(),
                'x' : features.detach(),
                'z' : z.detach(),
                'y': y.detach(),
                's': s.detach()
            })

        return output
        

    def validation_step(self, batch, _):
        x, y, s = batch

        if self.current_epoch < self.hparams.control_epoch_optnet:
            features = self.feature_extractor(x)
            z = self.encoder(features)
            loss_ctl, proj_s, proj_y = self.criterion['adversary'](z, s, y, self.hparams.tau, self.hparams.gamma, 1.0)

            y_hat = torch.Tensor(len(y) * [[1,0]]).to(device=y.device)

            output = OrderedDict({
                'loss': loss_ctl.detach(),
                'loss_ctl': {'value': loss_ctl.detach(), 'numel': len(x)},
                'proj_s'  : proj_s,
                'proj_y'  : proj_y,
                'y_hat': y_hat.detach(),
                'x' : features.detach(),
                'z' : z.detach(),
                'y': y.detach(),
                's': s.detach()
            })

        else:

            features = self.feature_extractor(x)
            z = self.encoder(features)
            
            y_hat = self.target(z)

            loss_tgt = self.criterion['target'](y_hat, y)

            output = OrderedDict({
                'loss': loss_tgt.detach(),
                'loss_tgt': {'value': loss_tgt.detach(), 'numel': len(x)},
                'loss_ctl': {'value': torch.Tensor([0]), 'numel': len(x)},
                'proj_s'  : torch.Tensor([0]),
                'proj_y'  : torch.Tensor([0]),
                'y_hat': y_hat.detach(),
                'x' : features.detach(),
                'z' : z.detach(),
                'y': y.detach(),
                's': s.detach()
            })

        return output

    def test_step(self, batch, _):
        x, y, s = batch

        features = self.feature_extractor(x)

        z = self.encoder(features)

        y_hat = self.target(z)
        loss_tgt = self.criterion['target'](y_hat, y)

        loss_ctl = self.criterion['adversary'](z, s)


        output = OrderedDict({
            'loss': loss_tgt.detach(),
            'x' : features.detach(),
            's': s,
            'y_hat': y_hat,
            'y': y,
            'z' : z.detach(),
            'loss_tgt': {'value': loss_tgt.detach(), 'numel': len(x)},
            'loss_ctl': {'value': loss_ctl.detach(), 'numel': len(x)},
        })

        return output

