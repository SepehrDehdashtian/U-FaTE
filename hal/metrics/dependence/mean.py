# mean.py

import torch
from cmath import nan
import torchmetrics.metric as tm
from typing import Any, Callable, Optional

__all__ = ['DepMean']

class DepMean(tm.Metric):
    def __init__(self,
        alpha_x = torch.nn.Identity,
        alpha_y = torch.nn.Identity,
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
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y
        self.add_state("xx", default=[], dist_reduce_fx=None)
        self.add_state("yy", default=[], dist_reduce_fx=None)

    def update(self, x, y):
        xx = x
        yy = y

        self.xx.append(xx)
        self.yy.append(yy)

    def normalize(self, x):
        x_mean = torch.mean(x, dim=0)
        x_std = torch.std(x, dim=0)
        xx = (x - x_mean) / (x_std + 1e-16)
        return xx

    def compute(self):
        x = torch.cat(self.xx, dim=0)
        x = self.normalize(x)

        y = torch.cat(self.yy, dim=0)
        y = self.normalize(y)

        if self.alpha_x is torch.nn.Identity:
            kernel_y = self.alpha_y(y)
            kernel = kernel_y + 0.05 * torch.eye(kernel_y.shape[0], device=kernel_y.device)
            model = torch.mm(kernel_y, torch.linalg.inv(kernel))
            x_hat = torch.mm(model, x)
            output = 1 - torch.norm(x_hat - x) ** 2 / torch.norm(x) ** 2

        elif self.alpha_y is torch.nn.Identity:
            kernel_x = self.alpha_x(x)
            kernel = kernel_x + 0.05 * torch.eye(kernel_x.shape[0], device=kernel_x.device)
            model = torch.mm(kernel_x, torch.linalg.inv(kernel))
            y_hat = torch.mm(model, y)
            output = 1 - torch.norm(y_hat - y) ** 2 / torch.norm(y) ** 2
        else:
            print('something is wrong')
            return nan

        return output