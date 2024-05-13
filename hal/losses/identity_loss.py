import torch
from torch import nn

__all__ = ['IdentityLoss']

class IdentityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, *args):
        return torch.tensor(0.0)
