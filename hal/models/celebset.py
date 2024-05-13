# CelebSET.py
import torch
from torch import nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch import GraphConv

__all__ = ['EncCelebSET', 'TgtCelebSET', 'LinearTgtCelebSET', 'AdvCelebSET', 'LinearAdvCelebSET', 'RepCelebSETHGR', 'AdvCelebSETHGR']

class EncCelebSET(nn.Module):
    def __init__(self, ndim, r, hdl):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(ndim, 2*hdl),
            nn.PReLU(),
            nn.Linear(2*hdl, hdl),
            nn.PReLU(),
            # nn.Linear(hdl, hdl//2),
            # nn.PReLU(),
            # nn.Linear(hdl//2, r, bias=False)
            nn.Linear(hdl, r, bias=False)
        )

    def forward(self, x, y=None):
        z = self.encoder(x)
        ###### Normalization #####
        # z = z / (torch.norm(z, dim=1, keepdim=True) + 1e-16)
        return z


class TgtCelebSET(nn.Module):
    def __init__(self, nout, r, hdl):
        super().__init__()
        self.arl_target = nn.Sequential(
            nn.Linear(r, 2*hdl),
            nn.PReLU(),
            # nn.Linear(2*hdl, hdl),
            # nn.PReLU(),
            nn.Linear(2*hdl, nout),
        )

    def forward(self, z, y=None):
        out = self.arl_target(z)
        return out

class LinearTgtCelebSET(nn.Module):
    def __init__(self, nout, r, hdl):
        super().__init__()
        self.arl_linear_target = nn.Sequential(
            nn.Linear(r, nout)
        )

    def forward(self, z, y=None):
        out = self.arl_linear_target(z)
        return out

class AdvCelebSET(nn.Module):

    def __init__(self, nout, r, hdl):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(r, 2*hdl),
            nn.PReLU(),
            nn.Linear(2*hdl, hdl),
            nn.PReLU(),
            nn.Linear(hdl, nout),
        )

    def forward(self, x, y=None):
        out = self.decoder(x)
        return out

class LinearAdvCelebSET(nn.Module):
    def __init__(self, nout, r, hdl):
        super().__init__()
        self.linear_decoder = nn.Sequential(
            nn.Linear(r, nout)
        )

    def forward(self, x, y=None):
        out = self.linear_decoder(x)
        return out

class RepCelebSETHGR(nn.Module):

    def __init__(self, nout, r, hdl):
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(r, hdl),
            nn.PReLU(),
            nn.Linear(hdl, hdl),
            nn.PReLU(),
            nn.Linear(hdl, nout),
        )

    def forward(self, z, y=None):
        out = self.target(z)
        return out


class AdvCelebSETHGR(nn.Module):

    def __init__(self, nout, ndim, hdl):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(ndim, hdl),
            nn.PReLU(),
            nn.Linear(hdl, hdl),
            nn.PReLU(),
            nn.Linear(hdl, nout),
        )

    def forward(self, x, y=None):
        out = self.decoder(x)
        return out



