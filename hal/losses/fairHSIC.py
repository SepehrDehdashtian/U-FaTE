# classification.py

import torch
from torch import nn
import hal.kernels as kernels
import hal.utils.misc as misc

__all__ = ['FairHSICAdversaryLoss', 'HSICAdversaryLoss']

class FairHSICAdversaryLoss(nn.Module):
    def __init__(self, kernel_s, kernel_s_opts,
                 kernel_z, kernel_z_opts,
                 hsic_ss, num_classes):
        """
        Loss function that calculates the correlation between the output
        of adversary_z and the output of adversary_s.
        """
        super().__init__()
        self.eps = 1e-4
        self.num_classes = num_classes
        self.kernel_s = getattr(kernels, kernel_s)(**kernel_s_opts)
        self.kernel_z = getattr(kernels, kernel_z)(**kernel_z_opts)
        self.hsic_ss = hsic_ss

    def forward(self, z, s, y):
        phi_s = self.kernel_s(s)
        phi_s = misc.mean_center(phi_s, dim=0)

        phi_z = self.kernel_z(z)
        phi_z = misc.mean_center(phi_z, dim=0)

        n, _ = phi_z.shape  # batch size

        loss = 0

        for l in range(self.num_classes):
            mask = (torch.argmax(y, 1) == l).reshape(-1)
            phi_z_mask = phi_z[mask]
            phi_s_mask = phi_s[mask]
            hsic_zz = torch.norm(torch.mm(phi_z_mask.t(), phi_z_mask), p='fro') / n
            
            hsic_zs = torch.norm(torch.mm(phi_z_mask.t(), phi_s_mask), p='fro') / n

            loss += -hsic_zs ** 2 / (hsic_zz * self.hsic_ss)

        return loss


class HSICAdversaryLoss(nn.Module):
    def __init__(self, num_classes, kernel_s, kernel_s_opts,
                 kernel_z, kernel_z_opts):
        """
        Loss function that calculates the correlation between the output
        of adversary_z and the output of adversary_s.
        """
        super().__init__()
        self.eps      = 1e-4
        self.num_classes = num_classes
        self.kernel_s = getattr(kernels, kernel_s)(**kernel_s_opts)
        self.kernel_z = getattr(kernels, kernel_z)(**kernel_z_opts)

    def forward(self, z, s):
        torch.set_default_tensor_type(torch.DoubleTensor)

        s = self.format_s_onehot(s)

        z, s = z.double(), s.double()

        K_s = self.kernel_s(s)
        K_s = misc.mean_center(K_s, dim=0)
        K_sm = misc.mean_center(K_s.t(), dim=0)

        K_z = self.kernel_z(z)
        K_z = misc.mean_center(K_z, dim=0)
        K_zm = misc.mean_center(K_z.t(), dim=0)

        try:
            hsic = torch.trace(torch.mm(K_zm.t(), K_sm)) / torch.sqrt(
                torch.trace(torch.mm(K_sm.t(), K_sm)) * torch.trace(torch.mm(K_zm.t(), K_zm)))
        except:
            import pdb; pdb.set_trace()

        if torch.isnan(hsic):
            hsic = torch.zeros_like(hsic).to(device=hsic.device)

        torch.set_default_tensor_type(torch.FloatTensor)

        return hsic

    def format_s_onehot(self, s):
        # int -> one-hot
        if len(s.shape) > 2:
            s = s.squeeze(-1)
        s_onehot = torch.zeros(s.size(0), self.num_classes, device=s.device).scatter_(1, s.unsqueeze(1).long(), 1)
        return s_onehot
