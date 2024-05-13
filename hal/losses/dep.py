# classification.py

import torch
from torch import nn
import hal.kernels as kernels
import hal.utils.misc as misc

__all__ = ['DEPLoss', 'DEPLossLinear', 'DEPLoss_old', 'EODEPLoss', 'EODEPLossLinear', 'RFFEODEPLoss', 'EODEPLoss2', 'EOODEPLossBinary']


class DEPLoss_old(nn.Module):
    def __init__(self, onehot_num_classes, kernel_s, kernel_s_opts,
                 kernel_z, kernel_z_opts):
        """
        Loss function that calculates the correlation between the representation (Z) and 
        the sensitive attribute (S) or the target attribute (Y)
        """
        super().__init__()
        self.eps      = 1e-4
        self.num_classes    = onehot_num_classes
        self.kernel_s       = getattr(kernels, kernel_s)(**kernel_s_opts)
        self.kernel_z       = getattr(kernels, kernel_z)(**kernel_z_opts)

    def forward(self, z, s, norm=None):
        torch.set_default_tensor_type(torch.DoubleTensor)

        s = self.format_onehot(s)

        z, s = z.double(), s.double()

        K_s = self.kernel_s(s)
        K_s = misc.mean_center(K_s, dim=0)
        K_sm = misc.mean_center(K_s.t(), dim=0)

        K_z = self.kernel_z(z)
        K_z = misc.mean_center(K_z, dim=0)
        K_zm = misc.mean_center(K_z.t(), dim=0)

        dep = torch.trace(torch.mm(K_zm.t(), K_sm))

        dep = dep / (norm * len(z)**2)
        torch.set_default_tensor_type(torch.FloatTensor)

        if torch.isnan(dep):
            import pdb; pdb.set_trace()

        return dep

    def format_onehot(self, s):
        # int -> one-hot
        if len(s.shape) > 2:
            s = s.squeeze(-1)
        s_onehot = torch.zeros(s.size(0), self.num_classes, device=s.device).scatter_(1, s.unsqueeze(1).long(), 1)
        return s_onehot


class DEPLoss(nn.Module):
    def __init__(self, onehot_num_classes, kernel_s, kernel_s_opts,
                 kernel_z, kernel_z_opts, one_hot_s):
        """
        Loss function that calculates the correlation between the representation (Z) and 
        the sensitive attribute (S) or the target attribute (Y)
        """
        super().__init__()
        self.eps      = 1e-4
        self.num_classes    = onehot_num_classes
        self.kernel_s       = getattr(kernels, kernel_s)(**kernel_s_opts)
        self.kernel_z       = getattr(kernels, kernel_z)(**kernel_z_opts)
        self.one_hot_s      = one_hot_s

    def forward(self, z, s, norm=None):
        torch.set_default_tensor_type(torch.DoubleTensor)

        if self.one_hot_s:
            s = self.format_onehot(s)
        else:
            s = s.reshape(-1,1).float()


        z, s = z.double(), s.double()

        K_s = self.kernel_s(s)
        K_sm = misc.mean_center(K_s, dim=0)
        K_sm = misc.mean_center(K_sm.t(), dim=0)

        K_z = self.kernel_z(z)
        K_zm = misc.mean_center(K_z, dim=0)
        # K_zm = misc.mean_center(K_zm.t(), dim=0)


        if norm is None:
            dep = torch.trace(torch.mm(torch.mm(K_zm.t(), K_sm), K_zm))
        else:
            # dep = torch.trace(torch.mm(torch.mm(K_zm.t(), K_sm), K_zm)) / (norm * len(s)**2)
            dep = torch.trace(torch.mm(torch.mm(K_zm.t(), K_sm), K_zm)) / (norm * len(s))

        torch.set_default_tensor_type(torch.FloatTensor)
        # import pdb; pdb.set_trace()

        return dep

    def format_onehot(self, s):
        # int -> one-hot
        if len(s.shape) > 2:
            s = s.squeeze(-1)
        s_onehot = torch.zeros(s.size(0), self.num_classes, device=s.device).scatter_(1, s.unsqueeze(1).long(), 1)
        return s_onehot


class DEPLossLinear(nn.Module):
    def __init__(self, onehot_num_classes, kernel_s, kernel_s_opts,
                 kernel_z, kernel_z_opts, one_hot_s):
        """
        Loss function that calculates the correlation between the representation (Z) and 
        the sensitive attribute (S) or the target attribute (Y)
        """
        super().__init__()
        self.eps      = 1e-4
        self.num_classes    = onehot_num_classes
        self.kernel_s       = getattr(kernels, kernel_s)(**kernel_s_opts)
        self.kernel_z       = getattr(kernels, kernel_z)(**kernel_z_opts)
        self.one_hot_s      = one_hot_s

    def forward(self, z, s, norm=None):
        torch.set_default_tensor_type(torch.DoubleTensor)

        if self.one_hot_s:
            s = self.format_onehot(s)
        else:
            s = s.reshape(-1,1).float()


        z, s = z.double(), s.double()

        K_s = self.kernel_s(s)
        K_sm = misc.mean_center(K_s, dim=0)
        # K_sm = misc.mean_center(K_sm.t(), dim=0)

        K_z = self.kernel_z(z)
        K_zm = misc.mean_center(K_z, dim=0)
        # K_zm = misc.mean_center(K_zm.t(), dim=0)


        if norm is None:
            dep = torch.mm(K_zm.t(), K_sm).norm()
        else:
            # dep = torch.trace(torch.mm(torch.mm(K_zm.t(), K_sm), K_zm)) / (norm * len(s)**2)
            dep = torch.mm(K_zm.t(), K_sm).norm() / (norm * len(s))

        torch.set_default_tensor_type(torch.FloatTensor)
        # import pdb; pdb.set_trace()

        return dep

    def format_onehot(self, s):
        # int -> one-hot
        if len(s.shape) > 2:
            s = s.squeeze(-1)
        s_onehot = torch.zeros(s.size(0), self.num_classes, device=s.device).scatter_(1, s.unsqueeze(1).long(), 1)
        return s_onehot

class EODEPLoss(nn.Module):
    def __init__(self, onehot_num_classes, kernel_s, kernel_s_opts,
                 kernel_z, kernel_z_opts, one_hot_s):
        """
        Loss function that calculates the correlation between the representation (Z) and 
        the sensitive attribute (S) or the target attribute (Y)
        """
        super().__init__()
        self.eps      = 1e-4
        self.num_classes    = onehot_num_classes
        self.kernel_s       = getattr(kernels, kernel_s)(**kernel_s_opts)
        self.kernel_z       = getattr(kernels, kernel_z)(**kernel_z_opts)
        self.one_hot_s      = one_hot_s

    def forward(self, z, s, y, norm=None):
        torch.set_default_tensor_type(torch.DoubleTensor)

        mask = (y == 1)

        z = z[mask]
        s = s[mask]

        if self.one_hot_s:
            s = self.format_onehot(s)
        else:
            s = s.reshape(-1,1).float()


        z, s = z.double(), s.double()

        K_s = self.kernel_s(s)
        K_sm = misc.mean_center(K_s, dim=0)
        K_sm = misc.mean_center(K_sm.t(), dim=0)

        K_z = self.kernel_z(z)
        K_zm = misc.mean_center(K_z, dim=0)
        # K_zm = misc.mean_center(K_zm.t(), dim=0)


        if norm is None:
            dep = torch.trace(torch.mm(torch.mm(K_zm.t(), K_sm), K_zm))
        else:
            # dep = torch.trace(torch.mm(torch.mm(K_zm.t(), K_sm), K_zm)) / (norm * len(s)**2)
            dep = torch.trace(torch.mm(torch.mm(K_zm.t(), K_sm), K_zm)) / (norm * sum(mask))

        torch.set_default_tensor_type(torch.FloatTensor)

        if len(s) == 0:
            import pdb; pdb.set_trace()
        
        if torch.isnan(dep):
            import pdb; pdb.set_trace()
        return dep

    def format_onehot(self, s):
        # int -> one-hot
        if len(s.shape) > 2:
            s = s.squeeze(-1)
        s_onehot = torch.zeros(s.size(0), self.num_classes, device=s.device).scatter_(1, s.unsqueeze(1).long(), 1)
        return s_onehot

class EODEPLossLinear(nn.Module):
    def __init__(self, onehot_num_classes, kernel_s, kernel_s_opts,
                 kernel_z, kernel_z_opts, one_hot_s):
        """
        Loss function that calculates the correlation between the representation (Z) and 
        the sensitive attribute (S) or the target attribute (Y)
        """
        super().__init__()
        self.eps      = 1e-4
        self.num_classes    = onehot_num_classes
        self.kernel_s       = getattr(kernels, kernel_s)(**kernel_s_opts)
        self.kernel_z       = getattr(kernels, kernel_z)(**kernel_z_opts)
        self.one_hot_s      = one_hot_s

    def forward(self, z, s, y, norm=None):
        torch.set_default_tensor_type(torch.DoubleTensor)

        mask = (y == 1)

        z = z[mask]
        s = s[mask]

        if self.one_hot_s:
            s = self.format_onehot(s)
        else:
            s = s.reshape(-1,1).float()


        z, s = z.double(), s.double()

        K_s = self.kernel_s(s)
        K_sm = misc.mean_center(K_s, dim=0)
        # K_sm = misc.mean_center(K_sm.t(), dim=0)

        K_z = self.kernel_z(z)
        K_zm = misc.mean_center(K_z, dim=0)
        # K_zm = misc.mean_center(K_zm.t(), dim=0)


        if norm is None:
            dep = torch.mm(K_zm.t(), K_sm).norm()
        else:
            # dep = torch.trace(torch.mm(torch.mm(K_zm.t(), K_sm), K_zm)) / (norm * len(s)**2)
            dep = torch.mm(K_zm.t(), K_sm).norm() / (norm * sum(mask))

        torch.set_default_tensor_type(torch.FloatTensor)

        if len(s) == 0:
            import pdb; pdb.set_trace()
        
        if torch.isnan(dep):
            import pdb; pdb.set_trace()
        return dep

    def format_onehot(self, s):
        # int -> one-hot
        if len(s.shape) > 2:
            s = s.squeeze(-1)
        s_onehot = torch.zeros(s.size(0), self.num_classes, device=s.device).scatter_(1, s.unsqueeze(1).long(), 1)
        return s_onehot

class EOODEPLossBinary(nn.Module):
    def __init__(self, onehot_num_classes, kernel_s, kernel_s_opts,
                 kernel_z, kernel_z_opts, one_hot_s):
        """
        Loss function that calculates the correlation between the representation (Z) and 
        the sensitive attribute (S) or the target attribute (Y)
        """
        super().__init__()
        self.eps      = 1e-4
        self.num_classes    = onehot_num_classes
        self.kernel_s       = getattr(kernels, kernel_s)(**kernel_s_opts)
        self.kernel_z       = getattr(kernels, kernel_z)(**kernel_z_opts)
        self.one_hot_s      = one_hot_s

    def forward(self, z, s, y, norm=None):
        torch.set_default_tensor_type(torch.DoubleTensor)

        mask = (y == 1)

        if self.one_hot_s:
            s = self.format_onehot(s)
        else:
            s = s.reshape(-1,1).float()

        z0 = z[~mask].double()
        s0 = s[~mask].double()

        z1 = z[mask].double()
        s1 = s[mask].double()


        K_s0 = self.kernel_s(s0)
        K_sm0 = misc.mean_center(K_s0, dim=0)
        K_sm0 = misc.mean_center(K_sm0.t(), dim=0)

        K_z0 = self.kernel_z(z0)
        K_zm0 = misc.mean_center(K_z0, dim=0)


        K_s1 = self.kernel_s(s1)
        K_sm1 = misc.mean_center(K_s1, dim=0)
        K_sm1 = misc.mean_center(K_sm1.t(), dim=0)

        K_z1 = self.kernel_z(z1)
        K_zm1 = misc.mean_center(K_z1, dim=0)

        # import pdb; pdb.set_trace()

        if norm is None:
            dep = torch.trace(torch.mm(torch.mm(K_zm0.t(), K_sm0), K_zm0)) + torch.trace(torch.mm(torch.mm(K_zm1.t(), K_sm1), K_zm1))
        else:
            # dep = torch.trace(torch.mm(torch.mm(K_zm0.t(), K_sm0), K_zm0)) / (norm[0] * sum(~mask)) + torch.trace(torch.mm(torch.mm(K_zm1.t(), K_sm1), K_zm1)) / (norm[1] * sum(mask))
            # dep = ( torch.trace(torch.mm(torch.mm(K_zm0.t(), K_sm0), K_zm0)) * (sum(~mask)) + torch.trace(torch.mm(torch.mm(K_zm1.t(), K_sm1), K_zm1)) * (sum(mask)) ) / len(mask)
            dep = ( torch.trace(torch.mm(torch.mm(K_zm0.t(), K_sm0), K_zm0)) + torch.trace(torch.mm(torch.mm(K_zm1.t(), K_sm1), K_zm1)) ) / len(mask)
            # import pdb; pdb.set_trace()

        torch.set_default_tensor_type(torch.FloatTensor)

        if len(s) == 0:
            import pdb; pdb.set_trace()
        
        if torch.isnan(dep):
            import pdb; pdb.set_trace()
        return dep


    def format_onehot(self, s):
        # int -> one-hot
        if len(s.shape) > 2:
            s = s.squeeze(-1)
        s_onehot = torch.zeros(s.size(0), self.num_classes, device=s.device).scatter_(1, s.unsqueeze(1).long(), 1)
        return s_onehot


class RFFEODEPLoss(nn.Module):
    def __init__(self, onehot_num_classes, kernel_s, kernel_s_opts,
                 kernel_z, kernel_z_opts, one_hot_s):
        """
        Use it with RFF kernels
        This is the EO version of Dep:
        Loss function that calculates the correlation between the representation (Z) and 
        the sensitive attribute (S) or the target attribute (Y) 
        """
        super().__init__()
        self.eps            = 1e-4
        self.num_classes    = onehot_num_classes
        self.kernel_s       = getattr(kernels, kernel_s)(**kernel_s_opts)
        self.kernel_z       = getattr(kernels, kernel_z)(**kernel_z_opts)
        self.one_hot_s      = one_hot_s

    def forward(self, z, s, y, norm=None):
        torch.set_default_tensor_type(torch.DoubleTensor)

        mask = (y == 1)

        if self.one_hot_s:
            s = self.format_onehot(s)
        else:
            s = s.reshape(-1,1).float()

        phi_s = self.kernel_s(s[mask])
        phi_s_c = misc.mean_center(phi_s, dim=0)
        
        phi_z = self.kernel_z(z)
        phi_z_c = misc.mean_center(phi_z, dim=0)
        
        n, _ = phi_z.shape  # batch size
        
        if norm is None:
            hsic_zs = torch.norm(torch.mm(phi_z_c, phi_s_c.to(dtype=phi_z.dtype).t()), p='fro')
        else:
            hsic_zs = torch.norm(torch.mm(phi_z_c, phi_s_c.to(dtype=phi_z.dtype).t()), p='fro') / (norm * sum(mask)**2)

        torch.set_default_tensor_type(torch.FloatTensor)
        return hsic_zs

    def format_onehot(self, s):
        # int -> one-hot
        if len(s.shape) > 2:
            s = s.squeeze(-1)
        s_onehot = torch.zeros(s.size(0), self.num_classes, device=s.device).scatter_(1, s.unsqueeze(1).long(), 1)
        return s_onehot



class EODEPLoss_old(nn.Module):
    def __init__(self, onehot_num_classes, kernel_s, kernel_s_opts,
                 kernel_z, kernel_z_opts, one_hot_s):
        """
        Use it with RFF kernels
        This is the EO version of Dep:
        Loss function that calculates the correlation between the representation (Z) and 
        the sensitive attribute (S) or the target attribute (Y) 
        """
        super().__init__()
        self.eps            = 1e-4
        self.num_classes    = onehot_num_classes
        self.kernel_s       = getattr(kernels, kernel_s)(**kernel_s_opts)
        self.kernel_z       = getattr(kernels, kernel_z)(**kernel_z_opts)
        self.one_hot_s      = one_hot_s

    def forward(self, z, s, y, norm=None):
        torch.set_default_tensor_type(torch.DoubleTensor)

        mask = (y == 1)

        if self.one_hot_s:
            s = self.format_onehot(s)
        else:
            s = s.reshape(-1,1).float()

        phi_s = self.kernel_s(s[mask])
        phi_z = self.kernel_z(z[mask])
        
        n, _ = phi_z.shape  # batch size
        
        # if norm is None:
        #     hsic_zs = torch.norm(torch.mm(phi_z, phi_s.to(dtype=phi_z.dtype).t()), p='fro')
        # else:
        #     hsic_zs = torch.norm(torch.mm(phi_z, phi_s.to(dtype=phi_z.dtype).t()), p='fro') / (norm * sum(mask)**2)

        phi_z_c = misc.mean_center(phi_z, dim=0)

        phi_s_c = misc.mean_center(phi_s, dim=0) 
        phi_s_c = misc.mean_center(phi_s_c.t(), dim=0) 

        if norm is None:
            hsic_zs = torch.trace(torch.mm(torch.mm(phi_z_c.t(), phi_s_c), phi_z_c))
        else:
            # hsic_zs = torch.trace(torch.mm(torch.mm(phi_z_c.t(), phi_s_c), phi_z_c)) / (norm * sum(mask)**2)
            hsic_zs = torch.trace(torch.mm(torch.mm(phi_z_c.t(), phi_s_c), phi_z_c)) / (norm * sum(mask) ** 2)

        torch.set_default_tensor_type(torch.FloatTensor)
        return hsic_zs

    def format_onehot(self, s):
        # int -> one-hot
        if len(s.shape) > 2:
            s = s.squeeze(-1)
        s_onehot = torch.zeros(s.size(0), self.num_classes, device=s.device).scatter_(1, s.unsqueeze(1).long(), 1)
        return s_onehot



class EODEPLoss2(nn.Module):
    def __init__(self, onehot_num_classes, kernel_s, kernel_s_opts,
                 kernel_z, kernel_z_opts, one_hot_s):
        """
        Use it with RFF kernels
        This is the EO version of Dep:
        Loss function that calculates the correlation between the representation (Z) and 
        the sensitive attribute (S) or the target attribute (Y) 
        """
        super().__init__()
        self.eps            = 1e-4
        self.num_classes    = onehot_num_classes
        self.kernel_s       = getattr(kernels, kernel_s)(**kernel_s_opts)
        self.kernel_z       = getattr(kernels, kernel_z)(**kernel_z_opts)
        self.one_hot_s      = one_hot_s

    def forward(self, z, s, y, norm=None):
        torch.set_default_tensor_type(torch.DoubleTensor)

        mask = (y == 1)

        if self.one_hot_s:
            s = self.format_onehot(s)
        else:
            s = s.reshape(-1,1).float()

        phi_s = self.kernel_s(s[mask])
        phi_z = self.kernel_z(z)
        
        n, _ = phi_z.shape  # batch size
        
        # if norm is None:
        #     hsic_zs = torch.norm(torch.mm(phi_z, phi_s.to(dtype=phi_z.dtype).t()), p='fro')
        # else:
        #     hsic_zs = torch.norm(torch.mm(phi_z, phi_s.to(dtype=phi_z.dtype).t()), p='fro') / (norm * sum(mask)**2)

        phi_s_c = phi_s 

        if norm is None:
            hsic_zs = torch.norm(torch.mm(phi_z.t(), phi_s.to(dtype=phi_z.dtype).t()), p='fro')
        else:
            hsic_zs = torch.norm(torch.mm(phi_z, phi_s.to(dtype=phi_z.dtype).t()), p='fro') / (norm * sum(mask)**2)

        torch.set_default_tensor_type(torch.FloatTensor)
        return hsic_zs

    def format_onehot(self, s):
        # int -> one-hot
        if len(s.shape) > 2:
            s = s.squeeze(-1)
        s_onehot = torch.zeros(s.size(0), self.num_classes, device=s.device).scatter_(1, s.unsqueeze(1).long(), 1)
        return s_onehot


