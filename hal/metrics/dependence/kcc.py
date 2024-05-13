# kcc.py

import torch
import torchmetrics.metric as tm
from typing import Any, Callable, Optional

from hal import kernels

__all__ = ['DepKCC']

class DepKCC(tm.Metric):
    def __init__(self,
        lam = 1e-3,
        rff_dim = 200,
        use_rff = False,
        kernel_x = kernels.Gaussian,
        kernel_y = None,
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
        self.add_state("xx", default=[], dist_reduce_fx=None)
        self.add_state("yy", default=[], dist_reduce_fx=None)

        self.lam = lam
        self.rff_dim = rff_dim
        self.use_rff = use_rff
        self.kernel_x = kernel_x

        if self.kernel_y is None:
            self.kernel_y = kernel_x
        else:
            self.kernel_y = kernel_y

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

        indices = torch.randperm(x.shape[0])[:10000]
        x, y = x[indices, :], y[indices, :]

        if self.use_rff:
            phi_x = self.kernel_x(x)
            phi_y = self.kernel_y(y)

            num_x, num_y = x.shape[0], y.shape[0]
            assert num_x == num_y

            C_xx = torch.mm(phi_x.t(), phi_x) / num_x
            C_yy = torch.mm(phi_y.t(), phi_y) / num_y
            C_xy = torch.mm(phi_x.t(), phi_y) / num_y

            A_1 = torch.cat((torch.zeros(self.rff_dim, self.rff_dim), torch.mm(torch.linalg.inv(C_xx + self.lam * torch.eye(self.rff_dim)), C_xy)), dim=1)
            A_2 = torch.cat((torch.mm(torch.linalg.inv(C_yy + self.lam * torch.eye(self.rff_dim)), C_xy.t()), torch.zeros(self.rff_dim, self.rff_dim)), dim=1)
        else:
            C_xx = self.kernel_x(x)
            C_yy = self.kernel_x(y)
            C_xy = self.kernel_x(x, y)
            num_x, num_y = C_xx.shape[0], C_yy.shape[0]
            A_1 = torch.cat((torch.zeros(num_x, num_y), torch.mm(torch.linalg.inv(C_xx + self.lam * torch.eye(num_x)), C_xy)), dim=1)
            A_2 = torch.cat((torch.mm(torch.linalg.inv(C_yy + self.lam * torch.eye(num_y)), C_xy.t()), torch.zeros(num_y, num_x)), dim=1)
            
        A = torch.cat((A_1, A_2), dim=0)
        kcc = torch.max(torch.real(torch.linalg.eig(A)[0]))

        return kcc