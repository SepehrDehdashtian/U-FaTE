# nncc.py

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
import torchmetrics.metric as tm
from pytorch_lightning.core.lightning import LightningModule

from collections import OrderedDict
from typing import Any, Callable, Optional

import hal.models as models

__all__ = ['DepNNCC']


class NNCC(LightningModule):

    def __init__(self, ndim_x, ndim_y, hdl, adjust=False):
        super(NNCC, self).__init__()

        self.val_stack = {'sum_x': [], 'sum_y': [], 'norm_x': [], 'norm_y': [], 'inner_xy': [], 'n': []}
        self.test_stack = {'sum_x': [], 'sum_y': [], 'norm_x': [], 'norm_y': [], 'inner_xy': [], 'n': []}
        self.train_stack = {'sum_x': [], 'sum_y': [], 'norm_x': [], 'norm_y': [], 'inner_xy': [],'n': []}

        self.adjust = adjust

        self.encoder_x = getattr(models, self.hparams.model_type['model'])(**self.hparams.model_options['model'])
        self.encoder_y = getattr(models, self.hparams.model_type['model'])(**self.hparams.model_options['model'])

    def forward(self, x, y):
        x_0 = self.encoder_x(x)
        y_0 = self.encoder_y(y)
        return x_0, y_0

    def training_step(self, x, y):
        sum_x = torch.sum(x)
        sum_y = torch.sum(y)

        norm_x = torch.sum(x ** 2)
        norm_y = torch.sum(y ** 2)
        inner_xy = torch.sum(x * y)

        vx = x - torch.mean(x, dim=0)
        vy = y - torch.mean(y, dim=0)
        numerator = torch.sum(vx * vy)
        denominator = torch.sqrt(torch.sum(vx ** 2)) *  torch.sqrt(torch.sum(vy ** 2))
        corr_xs =  numerator / (denominator + 1e-16)

        output = OrderedDict({
            'loss': -corr_xs,
            'sum_x': sum_x,
            'sum_y': sum_y,
            'norm_x': norm_x,
            'norm_y': norm_y,
            'inner_xy': inner_xy,
            'n': x.shape[0],
        })

        return output

    def training_epoch_end(self, outputs):
        sum_x = torch.stack([x['sum_x'] for x in outputs])
        sum_y = torch.stack([x['sum_y'] for x in outputs])
        norm_x = torch.stack([x['norm_x'] for x in outputs])
        norm_y = torch.stack([x['norm_y'] for x in outputs])
        inner_xy = torch.stack([x['inner_xy'] for x in outputs])
        n = torch.stack([x['n'] for x in outputs]).sum()

        cov = torch.sum(inner_xy) / n - torch.sum(sum_x) * torch.sum(sum_y) / (n**2)
        var_x = torch.sum(norm_x) / n - torch.sum(sum_x) ** 2 / (n ** 2)  # var(x) = E{x^2} - E^2{x}
        var_y = torch.sum(norm_y) / n - torch.sum(sum_y) ** 2 / (n ** 2)  # var(x) = E{x^2} - E^2{x}

        epoch_dep = torch.abs(cov / torch.sqrt(var_x * var_y))
        if self.adjust:
            epoch_dep = self.adjust_corr(epoch_dep, n)

        return torch.sqrt(epoch_dep)

    def validation_step(self, x, y):
        sum_x = torch.sum(x)
        sum_y = torch.sum(y)

        norm_x = torch.sum(x ** 2)
        norm_y = torch.sum(y ** 2)
        inner_xy = torch.sum(x * y)

        output = OrderedDict({
            'sum_x': sum_x,
            'sum_y': sum_y,
            'norm_x': norm_x,
            'norm_y': norm_y,
            'inner_xy': inner_xy,
            'n': x.shape[0],
        })

        return output

    def validation_epoch_end(self, outputs):
        sum_x = torch.stack([x['sum_x'] for x in outputs])
        sum_y = torch.stack([x['sum_y'] for x in outputs])
        norm_x = torch.stack([x['norm_x'] for x in outputs])
        norm_y = torch.stack([x['norm_y'] for x in outputs])
        inner_xy = torch.stack([x['inner_xy'] for x in outputs])
        n = torch.stack([x['n'] for x in outputs]).sum()

        cov = torch.sum(inner_xy) / n - torch.sum(sum_x) * torch.sum(sum_y) / (n**2)
        var_x = torch.sum(norm_x) / n - torch.sum(sum_x) ** 2 / (n ** 2)  # var(x) = E{x^2} - E^2{x}
        var_y = torch.sum(norm_y) / n - torch.sum(sum_y) ** 2 / (n ** 2)  # var(x) = E{x^2} - E^2{x}

        dep = torch.abs(cov / torch.sqrt(var_x * var_y))
        if self.adjust:
            dep = self.adjust_corr(dep, n)

        return torch.sqrt(dep)

    def test_step(self, x, y):
        sum_x = torch.sum(x)
        sum_y = torch.sum(y)

        norm_x = torch.sum(x ** 2)
        norm_y = torch.sum(y ** 2)
        inner_xy = torch.sum(x * y)

        output = OrderedDict({
            'sum_x': sum_x,
            'sum_y': sum_y,
            'norm_x': norm_x,
            'norm_y': norm_y,
            'inner_xy': inner_xy,
            'n': x.shape[0],
        })

        return output

    def test_epoch_end(self, outputs):
        sum_x = torch.stack([x['sum_x'] for x in outputs])
        sum_y = torch.stack([x['sum_y'] for x in outputs])
        norm_x = torch.stack([x['norm_x'] for x in outputs])
        norm_y = torch.stack([x['norm_y'] for x in outputs])
        inner_xy = torch.stack([x['inner_xy'] for x in outputs])
        n = torch.stack([x['n'] for x in outputs]).sum()

        var_x = torch.sum(norm_x) / n - torch.sum(sum_x) ** 2 / (n ** 2)  # var(x) = E{x^2} - E^2{x}
        var_y = torch.sum(norm_y) / n - torch.sum(sum_y) ** 2 / (n ** 2)  # var(x) = E{x^2} - E^2{x}

        cov = torch.sum(inner_xy) / n - torch.sum(sum_x) * torch.sum(sum_y) / (n ** 2)
        dep = torch.abs(cov / torch.sqrt(var_x * var_y))
        if self.adjust:
            dep = self.adjust_corr(dep, n)

        return torch.sqrt(dep)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=1e-4)

    def adjust_corr(self, r, n):
        r_adj = 1 - ((1 - r**2) * (n - 1)) / (n - 2)
        return r_adj

    def compute_dep(self, x, y):
        x_mean = torch.mean(x, dim=0)
        y_mean = torch.mean(y, dim=0)
        vx = x - x_mean
        vy = y - y_mean

        numerator = torch.sum(vx * vy)
        denominator = torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))
        dep = numerator / denominator

        n = x.shape[0]
        if self.adjust:
            dep = torch.abs(self.adjust_corr(dep, n))

        return torch.sqrt(torch.abs(dep))


class DataPrepare(pl.LightningDataModule):
    def __init__(self, x, s):
        self.x = x
        self.s = s

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x, s = self.x[index], self.s[index]
        return x, s


class NNCCDataLoader(pl.LightningDataModule):
    def __init__(self, x, s, batch_size):
        super().__init__()
        self.x = x
        self.s = s
        self.shuffle = True
        self.batch_size = batch_size
        self.pin_memory = True

    def train_dataloader(self):
        dataset = DataPrepare(self.x, self.s)
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )
        return loader

class DepNNCC(tm.Metric):
    def __init__(self,
        batch_size,
        num_epochs,
        model_options,
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__(
                compute_on_step=compute_on_step,
                dist_sync_on_step=dist_sync_on_step,
                process_group=process_group,
                dist_sync_fn=dist_sync_fn,
            )

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model_options = model_options

        self.add_state("x", default=[], dist_reduce_fx=None)
        self.add_state("y", default=[], dist_reduce_fx=None)

    def update(self, x, y):
        self.x.append(x)
        self.s.append(y)

    def compute(self):
        x = torch.cat(self.x, dim=0)
        y = torch.cat(self.y, dim=0)

        x_mean = torch.mean(x, dim=0)
        x_std = torch.std(x, dim=0)
        x_m = ((x - x_mean) / (x_std + 1e-16))

        y_mean = torch.mean(y, dim=0)
        y_std = torch.std(y, dim=0)
        y_m = ((y - y_mean) / (y_std + 1e-16))

        dataloader = NNCCDataLoader(x_m, y_m, batch_size=self.batch_size)
        model = NNCC(**self.model_options)

        trainer = pl.Trainer(
            gpus=1,
            min_epochs=1,
            max_epochs=self.num_epochs,
        )
        trainer.fit(model, dataloader)
        dep = model.compute_dep(x_m, y_m)
        return dep