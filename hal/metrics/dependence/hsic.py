# hsic.py

import torch
from hal import kernels
import torchmetrics.metric as tm
from typing import Any, Callable, Optional

__all__ = ['DepHSIC']

class DepHSIC(tm.Metric):
    def __init__(self,
        alpha_x = kernels.Linear,
        alpha_y = kernels.Linear,
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
        self.xx.append(x)
        self.yy.append(y)

    def normalize(self, x):
        x_mean = torch.mean(x, dim=0)
        x_std = torch.std(x, dim=0)
        xx = (x - x_mean) / (x_std + 1e-16)
        return xx

    def compute(self):
        x = torch.cat(self.xx, dim=0)
        y = torch.cat(self.yy, dim=0)
        
        x = self.normalize(x)
        y = self.normalize(y)

        kernel_x = self.alpha_x(x)
        kernel_y = self.alpha_y(y)
        n = kernel_x.shape[0]

        H = torch.eye(n, device=y.device) - torch.ones(n, device=y.device) / n
        kernel_xm = torch.mm(H, torch.mm(kernel_x, H))
        kernel_ym = torch.mm(H, torch.mm(kernel_y, H))

        num = torch.trace(torch.mm(kernel_ym, kernel_xm))
        den = torch.sqrt(torch.trace(torch.mm(kernel_ym, kernel_ym)) * torch.trace(torch.mm(kernel_xm, kernel_xm)))
        hsic = num / den

        return hsic