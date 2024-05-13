# closed_form.py

"""
This class implements the closed form solver from ICCV 2019 paper.
"""

from ast import Param
from numpy import require
from torch.nn.parameter import Parameter
import torch
import torch.nn as nn

import hal.kernels as kernels
import hal.utils.misc as misc

__all__ = ["ClosedFormTgt", "ClosedFormAdv"]

class ClosedFormTgt(nn.Module):
    def __init__(self,
                 lam: float,
                 kernel_z: str,
                 kernel_z_opts: dict,
                 y_dim: int):
        super().__init__()
        self.lam = lam
        self.kernel_z = getattr(kernels, kernel_z)(**kernel_z_opts)
        rff_dim = kernel_z_opts.get("rff_dim")
        self.target_closed = Parameter(data=torch.zeros(rff_dim, y_dim), requires_grad=False)
    
    def forward(self, z, y=None):
        if y is not None:
            self.y_m = torch.mean(y, axis=0)
            phi_z = self.kernel_z(z)
            phi_z = misc.mean_center(phi_z, dim=0)
            # import pdb; pdb.set_trace()
            calc_target_closed = torch.mm(torch.mm(torch.linalg.inv(torch.mm(phi_z.t(), phi_z)
                + self.lam * torch.eye(phi_z.shape[1], device=z.device)), phi_z.t()), y-self.y_m)
            self.target_closed = Parameter(data=calc_target_closed, requires_grad=False)
            y_hat = torch.mm(phi_z, self.target_closed) + self.y_m
        else:
            phi_z = torch.sqrt(torch.tensor([2./self.kernel_z.w.size(0)], device=z.device)) * \
                        torch.cos(torch.mm(z, self.kernel_z.w.t()) + self.kernel_z.b)
            phi_z = misc.mean_center(phi_z, dim=0)
            y_hat = torch.mm(phi_z, self.target_closed) + self.y_m
        
        return y_hat


class ClosedFormAdv(nn.Module):
    def __init__(self,
                 lam: float,
                 kernel_s: str,
                 kernel_s_opts: dict,
                 s_dim: int):
        super().__init__()
        self.lam = lam
        self.kernel_s = getattr(kernels, kernel_s)(**kernel_s_opts)
        rff_dim = kernel_s_opts.get("rff_dim")
        self.target_closed = Parameter(data=torch.zeros(rff_dim, s_dim), requires_grad=False)

    def forward(self, z, s=None):
        if s is not None:
            self.s_m = torch.mean(s, axis=0)
            phi_z = self.kernel_s(z)
            phi_z = misc.mean_center(phi_z, dim=0)
            calc_target_closed = torch.mm(torch.mm(torch.linalg.inv(torch.mm(phi_z.t(), phi_z)
                                                                    + self.lam * torch.eye(phi_z.shape[1],
                                                                                             device=z.device)),
                                                   phi_z.t()), s - self.s_m)
            self.target_closed = Parameter(data=calc_target_closed, requires_grad=False)
            s_hat = torch.mm(phi_z, self.target_closed) + self.s_m
        else:
            phi_z = torch.sqrt(torch.tensor([2. / self.kernel_s.w.size(0)], device=z.device)) * \
                    torch.cos(torch.mm(z, self.kernel_s.w.t()) + self.kernel_s.b)
            phi_z = misc.mean_center(phi_z, dim=0)
            s_hat = torch.mm(phi_z, self.target_closed) + self.s_m

        return s_hat