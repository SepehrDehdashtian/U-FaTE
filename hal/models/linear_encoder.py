# kernelized_encoder.py

import torch
import math
from torch import nn
from torch.nn.parameter import Parameter

import hal.kernels as kernels

__all__ = ['LinearEncoder']

class LinearEncoder(nn.Module):
    def __init__(self, U):
        super().__init__()
        self.dtype = U.dtype

        if isinstance(U, list):
            self.U = Parameter(torch.zeros(*tuple(U)), requires_grad=False)
        else:
            self.U = Parameter(U)

    def forward(self, x):
        x = x.to(dtype=self.dtype)
        z = torch.mm(x, self.U)
        return z.float()