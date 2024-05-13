# adversarial_representation_learning.py

import torch
from collections import OrderedDict

import hal.losses as losses
from control.base import BaseClass

__all__ = ['ZeroShot']


class ZeroShot(BaseClass):
    def __init__(self, opts, dataloader):
        super().__init__(opts, dataloader)
        
        self.text = dataloader.text_y


    def training_step(self, batch, batch_idx):
        return None

    def test_step(self, batch, _):
        image, y, s = batch

        
        y_hat, _ = self.vlm(image, self.text)

        hate = self.vlm.hate_evaluation(image)

        loss_tgt = torch.tensor([0], dtype=torch.float32)

        output = OrderedDict({
            'loss': loss_tgt.detach(),
            'y_hat': y_hat,
            'y': y,
            's': s,
            'hate': hate
        })

        return output

