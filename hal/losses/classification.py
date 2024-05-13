# classification.py

import torch
from torch import nn
import hal.kernels as kernels
import hal.utils.misc as misc
import math

__all__ = ['Logistic', 'Classification', 'OneHotCrossEntropy', 'HGRAdversaryLoss', 'NPHGRAdversaryLoss',
            'HybAdversaryLoss', 'NPHybAdversaryLoss', 'NPSymHybAdversaryLoss', 'HSICAdversaryLoss']


class Logistic(nn.Module):
    def __init__(self):
        super(Logistic, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def __call__(self, inputs, targets):
        loss = self.loss(inputs, targets)
        return loss


class Classification(nn.Module):
    def __init__(self):
        super(Classification, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, inputs, targets):
        loss = self.loss(inputs, targets)
        return loss

class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()
        self.softmax = nn.Softmax

    def __call__(self, inputs):
        inputs = self.softmax(inputs)
        loss = -(inputs * inputs.log()).sum(dim=1)
        return loss

class OneHotCrossEntropy(nn.Module):
    def __init__(self):
        """
        This class calculates cross entropy loss when y is provided as
        one-hot encoding instead of class index.
        """
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.kl_div = nn.KLDivLoss(log_target=False, reduction="batchmean")

    def forward(self, x, y):
        return self.kl_div(self.log_softmax(x), y)

class HGRAdversaryLoss(nn.Module):
    def __init__(self, gamma):
        """
        Loss function that calculates the correlation between the output
        of adversary_z and the output of adversary_s.
        """
        super().__init__()
        self.eps = gamma

    def forward(self, adv_out, s=None):
        z_hat, s_hat = adv_out
        s_hat -= s_hat.mean()
        z_hat -= z_hat.mean()

        cov = torch.mean(s_hat * z_hat)

        z_var = torch.var(z_hat)
        s_var = torch.var(s_hat)
        return -torch.abs(cov**2 / (z_var * s_var + self.eps))

class NPHGRAdversaryLoss(nn.Module):
    def __init__(self, kernel_s, kernel_s_opts, kernel_z, kernel_z_opts, gamma):
        """
        Loss function that calculates the correlation between the output
        of adversary_z and the output of adversary_s.
        """
        super().__init__()
        self.lam = gamma
        self.kernel_s = getattr(kernels, kernel_s)(**kernel_s_opts)
        self.kernel_z = getattr(kernels, kernel_z)(**kernel_z_opts)
        self.rff = 'rff_dim' in kernel_s_opts

    def forward(self, z, s):
        ######################### Z-normalization #########################
        # z_mean = torch.mean(z, dim=0)
        # z_std = torch.std(z, dim=0)
        # z = (z - z_mean) / (z_std + 1e-16)
        #
        # s_mean = torch.mean(s, dim=0)
        # s_std = torch.std(s, dim=0)
        # s = (s - s_mean) / (s_std + 1e-16)

        ######################### Norm-normalization #######################

        # z = z / (z.norm() / math.sqrt(z.shape[0]))
        # s = s / (s.norm() / math.sqrt(s.shape[0]))
        ###################################################################

        phi_s = self.kernel_s(s)
        phi_s = misc.mean_center(phi_s, dim=0)
        if not self.rff:
            phi_s = misc.mean_center(phi_s.t(), dim=0)

        phi_z = self.kernel_z(z)
        phi_z = misc.mean_center(phi_z, dim=0)
        if not self.rff:
            phi_z = misc.mean_center(phi_z.t(), dim=0)

        n = phi_z.shape[0]

        m_z = phi_z.shape[1]
        m_s = phi_s.shape[1]

        if self.rff:
            c_zz = torch.mm(phi_z.t(), phi_z) / n
            c_ss = torch.mm(phi_s.t(), phi_s) / n
            c_zs = torch.mm(phi_z.t(), phi_s) / n
        else:
            c_zz = phi_z / n
            c_ss = phi_z / n


        # print('rank_z', torch.linalg.matrix_rank(c_zz + self.lam * torch.eye(m_z, device=c_zz.device)))
        # print('rank_s', torch.linalg.matrix_rank(c_ss + self.lam * torch.eye(m_s, device=c_zz.device)))

        psd_z = torch.linalg.inv(c_zz + self.lam * torch.eye(m_z, device=c_zz.device))
        psd_s = torch.linalg.inv(c_ss + self.lam * torch.eye(m_s, device=c_ss.device))

        zeros_z = torch.zeros(m_z, m_z, device=c_zz.device)
        zeros_s = torch.zeros(m_s, m_s, device=c_zz.device)

        if self.rff:
            a_zs = torch.cat((zeros_z, torch.mm(psd_z, c_zs)), dim=1)
            a_sz = torch.cat((torch.mm(psd_s, c_zs.t()), zeros_s), dim=1)
        else:
            a_zs = torch.cat((zeros_z, torch.mm(psd_z, c_ss)), dim=1)
            a_sz = torch.cat((torch.mm(psd_s, c_zz), zeros_s), dim=1)

        b_zs = torch.cat((a_zs, a_sz), dim=0)

        a_zz = torch.cat((zeros_z, torch.mm(psd_z, c_zz)), dim=1)
        a_zzt = torch.cat((torch.mm(psd_z, c_zz.t()), zeros_z), dim=1)
        b_zz = torch.cat((a_zz, a_zzt), dim=0)

        a_ss = torch.cat((zeros_s, torch.mm(psd_s, c_ss)), dim=1)
        a_sst = torch.cat((torch.mm(psd_s, c_ss.t()), zeros_s), dim=1)
        b_ss = torch.cat((a_ss, a_sst), dim=0)

        kcc_zs = torch.max(torch.real(torch.linalg.eigvals(b_zs))) / n
        kcc_zz = torch.max(torch.linalg.eigvalsh(b_zz)) / n
        kcc_ss = torch.max(torch.linalg.eigvalsh(b_ss)) / n
        # print("kcc_zz", kcc_zz, "kcc_zs", kcc_zs, "kcc_ss", kcc_ss)

        return -kcc_zs ** 2 / (kcc_zz * kcc_ss)

class HybAdversaryLoss(nn.Module):
    def __init__(self, kernel_s, kernel_s_opts, norm_s, gamma):
        """
        Loss function that calculates the correlation between the output
        of adversary_z and the output of adversary_s.
        """
        super().__init__()
        self.eps = gamma
        self.kernel_s = getattr(kernels, kernel_s)(**kernel_s_opts)
        self.norm_s = norm_s
        self.rff = 'rff_dim' in kernel_s_opts


    def forward(self, f_z, s):
        ######################### Z-normalization #########################
        #
        # s_mean = torch.mean(s, dim=0)
        # s_std = torch.std(s, dim=0)
        # s = (s - s_mean) / (s_std + 1e-16)

        ######################### Norm-normalization #######################
        # s = s / (s.norm() / math.sqrt(s.shape[0]))
        ###################################################################
        phi_s = self.kernel_s(s)
        phi_s = misc.mean_center(phi_s, dim=0)
        if not self.rff:
            phi_s = misc.mean_center(phi_s.t(), dim=0)

        f_z -= f_z.mean()

        hyb = torch.mm(f_z.t(), phi_s)

        n, _ = phi_s.shape  # batch size
        if self.rff:
            hyb = torch.mm(hyb, hyb.t()).squeeze(0).squeeze(0) / (n ** 2)
            # norm_2 = torch.linalg.norm(torch.mm(phi_s.t(), phi_s), 2) / n
        else:
            hyb = torch.mm(hyb, f_z).squeeze(0).squeeze(0) / (n**2)
            # norm_s = torch.linalg.norm(phi_s, 2) / n

        fz_var = torch.var(f_z)
        return -torch.abs(hyb / (fz_var * self.norm_s + self.eps))

class NPHybAdversaryLoss(nn.Module):
    def __init__(self, kernel_s, kernel_s_opts, kernel_z,
                 kernel_z_opts, norm_s, gamma):
        """
        Loss function that calculates the correlation between the output
        of adversary_z and the output of adversary_s.
        """
        super().__init__()
        self.eps = gamma
        self.kernel_s = getattr(kernels, kernel_s)(**kernel_s_opts)
        self.kernel_z = getattr(kernels, kernel_z)(**kernel_z_opts)
        self.norm_s = norm_s
        self.rff = 'rff_dim' in kernel_s_opts

    def forward(self, z, s):
        ######################### Z-normalization #########################
        # z_mean = torch.mean(z, dim=0)
        # z_std = torch.std(z, dim=0)
        # z = (z - z_mean) / (z_std + 1e-16)
        #
        # s_mean = torch.mean(s, dim=0)
        # s_std = torch.std(s, dim=0)
        # s = (s - s_mean) / (s_std + 1e-16)

        ######################### Norm-normalization #######################

        # z = z / (z.norm() / math.sqrt(z.shape[0]))
        # s = s / (s.norm() / math.sqrt(s.shape[0]))
        ###################################################################

        phi_s = self.kernel_s(s)
        n = phi_s.shape[0]
        phi_s = misc.mean_center(phi_s, dim=0)

        if not self.rff:
            phi_s = misc.mean_center(phi_s.t(), dim=0)

        phi_z = self.kernel_z(z)

############################################################################3
        # phi_z = (phi_z+phi_z.t())/2
        # v_z, sig, _ = torch.linalg.svd(phi_z)
        # d = 2*max(1, int(torch.linalg.matrix_rank(phi_z).item()))
        # phi_z = torch.mm(v_z[:, 0:d], torch.pow(torch.diag(sig[0:d]), 0.5))
############################################################################
        phi_z = misc.mean_center(phi_z, dim=0)
        b = torch.mm(phi_z.t(), phi_s)

        if self.rff:
            b = torch.mm(b, b.t())
        else:
            b = torch.mm(b, phi_z)

        a = torch.mm(phi_z.t(), phi_z)
        a += self.eps * n * torch.eye(a.shape[0], device=a.device)

        b /= n
        c = torch.mm(torch.linalg.inv(a), b)
        eigs_12 = torch.max(torch.real(torch.linalg.eigvals(c)))
        # eigs, _ = torch.lobpcg(c, k=1, method='ortho', largest=True)
        # import pdb; pdb.set_trace()
        # eigs_12 = eigs[0]
        # eigs_12 = torch.max(torch.linalg.eigvalsh(c / 2 + c.t() / 2))

        return -eigs_12 / self.norm_s
        # return -eigs_12 / norm_2

class NPSymHybAdversaryLoss(nn.Module):
    def __init__(self, kernel_s, kernel_s_opts, kernel_z,
                 kernel_z_opts, norm_s, gamma):
        """
        Loss function that calculates the correlation between the output
        of adversary_z and the output of adversary_s.
        """
        super().__init__()
        self.eps = gamma
        self.kernel_s = getattr(kernels, kernel_s)(**kernel_s_opts)
        self.kernel_z = getattr(kernels, kernel_z)(**kernel_z_opts)
        self.norm_s = norm_s
        self.rff = 'rff_dim' in kernel_s_opts

    def forward(self, z, s):
        ######################### Z-normalization #########################
        # z_mean = torch.mean(z, dim=0)
        # z_std = torch.std(z, dim=0)
        # z = (z - z_mean) / (z_std + 1e-16)
        #
        # s_mean = torch.mean(s, dim=0)
        # s_std = torch.std(s, dim=0)
        # s = (s - s_mean) / (s_std + 1e-16)
        ###################################################################

        phi_s = self.kernel_s(s)
        n = phi_s.shape[0]
        phi_s = misc.mean_center(phi_s, dim=0)

        phi_z = self.kernel_z(z)
        phi_z = misc.mean_center(phi_z, dim=0)

        b12 = torch.mm(phi_z.t(), phi_s)
        b12 = torch.mm(b12, b12.t())

        a12 = torch.mm(phi_z.t(), phi_z)
        a12 += self.eps * n * torch.eye(a12.shape[0], device=a12.device)

        b12 /= n
        c12 = torch.mm(torch.linalg.inv(a12), b12)
        eigs_12 = torch.max(torch.real(torch.linalg.eigvals(c12)))
        # eigs_12 = torch.max(torch.linalg.eigvalsh(c12 / 2 + c12.t() / 2))

        ###################################################################
        b21 = torch.mm(phi_s.t(), phi_z)
        b21 = torch.mm(b21, b21.t())

        a21 = torch.mm(phi_s.t(), phi_s)
        a21 += self.eps * n * torch.eye(a21.shape[0], device=a21.device)

        b21 /= n
        c21 = torch.mm(torch.linalg.inv(a21), b21)
        eigs_21 = torch.max(torch.real(torch.linalg.eigvals(c21)))
        # eigs_21 = torch.max(torch.linalg.eigvalsh(c21 / 2 + c21.t() / 2))
        self.norm_z = torch.linalg.norm(torch.mm(phi_z.t(), phi_z), 2) / n
        ########################################################################

        return torch.min(-eigs_12 / self.norm_s, -eigs_21 / self.norm_z)
        # return -eigs_21 / self.norm_z

class HSICAdversaryLoss(nn.Module):
    def __init__(self, kernel_s, kernel_s_opts, kernel_z, kernel_z_opts, hsic_ss):
        """
        Loss function that calculates the correlation between the output
        of adversary_z and the output of adversary_s.
        """
        super().__init__()
        self.eps = 1e-4
        self.kernel_s = getattr(kernels, kernel_s)(**kernel_s_opts)
        self.kernel_z = getattr(kernels, kernel_z)(**kernel_z_opts)
        self.hsic_ss = hsic_ss

    def forward(self, z, s):

        phi_s = self.kernel_s(s)
        phi_s = misc.mean_center(phi_s, dim=0)

        phi_z = self.kernel_z(z)
        phi_z = misc.mean_center(phi_z, dim=0)

        n, _ = phi_z.shape  # batch size
        hsic_zz = torch.norm(torch.mm(phi_z.t(), phi_z), p='fro') / n
        # hsic_ss = torch.norm(torch.mm(phi_s.t(), phi_s), p='fro') / n
        # import pdb; pdb.set_trace()
        hsic_zs = torch.norm(torch.mm(phi_z.t(), phi_s), p='fro') / n

        return -hsic_zs ** 2 / (hsic_zz * self.hsic_ss)
