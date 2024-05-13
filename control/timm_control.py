import torch
from collections import OrderedDict
import numpy as np
import hal.losses as losses
from control.base import BaseClass
from torchvision import transforms

__all__ = ['TIMM']


class TIMM(BaseClass):
    def __init__(self, opts, dataloader):
        super().__init__(opts, dataloader)

    def training_step(self, batch, batch_idx):
        image, y, s = batch
        
        z = self.timm(image)

        loss_tgt = torch.tensor([0]).float()
        y_hat    = torch.tensor(len(y)*[[0, 0]], device='cuda').float()

        output = OrderedDict({
            'loss': loss_tgt.detach(),
            'z': z.cpu(),
            'y': y,
            'y_hat': y_hat,
            's': s,
        })

        return output

    def test_step(self, batch, _):
        image, y, s = batch
        
        z = self.timm(image)

        loss_tgt = torch.tensor([0]).float()

        y_hat = self.clf.predict(z.cpu().detach())
        
        # to one-hot
        y_hat = torch.eye(2)[y_hat].cuda()


        output = OrderedDict({
            'loss': loss_tgt.detach(),
            'z': z.cpu(),
            'y': y.detach(),
            'y_hat': y_hat,
            's': s,
        })

        return output

