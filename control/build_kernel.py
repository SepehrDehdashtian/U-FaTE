# build_kernel.py

import torch
import hal.models as models
import hal.utils.misc as misc



def end2end_kernel(self, X, Y, S):
        device = 'cuda'
        # dtype  = torch.float
        dtype  = torch.double

        y = self.format_y_onehot(Y)
        s = self.format_s_onehot(S)

        n = len(X)

        if self.rff_flag:
            R_x       = self.kernel_x(X).to(dtype=dtype, device=device)
            R_x_c     = misc.mean_center(R_x, dim=0).to(dtype=dtype, device=device)
            dtype = R_x.dtype

            R_y = self.kernel_y(y).to(dtype=dtype, device=device)
            R_y_c = misc.mean_center(R_y, dim=0).to(dtype=dtype, device=device)

            R_s = self.kernel_s(s).to(dtype=dtype, device=device)
            R_s_c = misc.mean_center(R_s, dim=0).to(dtype=dtype, device=device)

            b_y = torch.mm(R_x.t(), R_y_c)
            b_y = torch.mm(b_y, b_y.t())

            b_s = torch.mm(R_x.t(), R_s_c)
            b_s = torch.mm(b_s, b_s.t())
            
            norm2_b_y = torch.linalg.norm(b_y, 2)
            norm2_b_s = torch.linalg.norm(b_s, 2)

            b = b_y / norm2_b_y - self.hparams.tau / (1. - self.hparams.tau)  * b_s / norm2_b_s

            self.norm2_b_y = norm2_b_y / n ** 2
            self.norm2_b_s = norm2_b_s / n ** 2

            b = (b + b.t()) / 2

            # H^2 = H
            c = torch.mm(R_x_c.t(), R_x_c) + n * self.hparams.gamma * torch.eye(R_x.shape[1], device=R_x.device)
            
            print(f'Gamma = {self.hparams.gamma}, Cond = {torch.linalg.cond(c)}')
            
            c = (c + c.t()) / 2

            eigs, V = torch.linalg.eig(torch.mm(torch.linalg.inv(c), b))
            eigs = torch.real(eigs)
            V = torch.real(V)

            sorted, indeces = torch.sort(eigs, descending=True)

            U = V[:, indeces[0:self.hparams.dim_z]]

            # import pdb; pdb.set_trace()

            #########################################
            r0 = self.hparams.dim_z
            if self.hparams.tau == 0:
                r = r0
            else:
                r1 = min((sorted > 0).sum(), r0)

            ###### Energy Thresholding ######
                if r1 > 0:
                    for k in range(1, r1+1):
                        if torch.linalg.norm(sorted[0:k])**2 / torch.linalg.norm(sorted[0:r1])**2 >= 0.95:
                            r = k
                            break
                else:
                    r = 0
            ######################################################
            if self.hparams.tau >= 0.999999999:
                r = 0
            ######################################################
            U[:, r:self.hparams.dim_z] = 0

            self.hparams.auto_dim_z = r
            
            encoder = models.KernelizedEncoder(U= n**0.5 * U, w=self.kernel_x.w, b=self.kernel_x.b)
        
        else:
            R_x       = self.kernel_x(X).to(dtype=dtype, device=device)
            R_x_c     = misc.mean_center(R_x, dim=0).to(dtype=dtype, device=device)
            dtype = R_x.dtype

            R_y = self.kernel_y(y).to(dtype=dtype, device=device)
            R_y_c = misc.mean_center(R_y, dim=0).to(dtype=dtype, device=device)
            R_y_c = misc.mean_center(R_y_c.t(), dim=0).to(dtype=dtype, device=device)

            R_s = self.kernel_s(s).to(dtype=dtype, device=device)
            R_s_c = misc.mean_center(R_s, dim=0).to(dtype=dtype, device=device)
            R_s_c = misc.mean_center(R_s_c, dim=0).to(dtype=dtype, device=device)

            b_y = torch.mm(torch.mm(R_x.t(), R_y_c), R_x)

            b_s = torch.mm(torch.mm(R_x.t(), R_s_c), R_x)

            b = b_y / torch.linalg.norm(b_y, 2) - self.hparams.tau / (1. - self.hparams.tau)  * b_s / torch.linalg.norm(b_s, 2)

            b = (b + b.t()) / 2

            # H^2 = H
            c = torch.mm(R_x_c.t(), R_x_c) + n * self.hparams.gamma * torch.eye(R_x.shape[1], device=R_x.device)
            c = (c + c.t()) / 2

            eigs, V = torch.linalg.eig(torch.mm(torch.linalg.inv(c), b))
            eigs = torch.real(eigs)
            V = torch.real(V)

            sorted, indeces = torch.sort(eigs, descending=True)


            U = V[:, indeces[0:self.hparams.dim_z]]

            #########################################
            r0 = self.hparams.dim_z
            if self.hparams.tau == 0:
                r = r0
            else:
                r1 = min((sorted > 0).sum(), r0)

            ###### Energy Thresholding ######
                if r1 > 0:
                    for k in range(1, r1+1):
                        if torch.linalg.norm(sorted[0:k])**2 / torch.linalg.norm(sorted[0:r1])**2 >= 0.92:
                            r = k
                            break
                else:
                    r = 0
            ######################################################
            if self.hparams.tau >= 0.99999:
                r = 0
            ###############################################################################
            U[:, r:self.hparams.dim_z] = 0

            self.hparams.auto_dim_z = r
            
            encoder = models.KernelizedEncoder(U=U, w=self.kernel_x.w, b=self.kernel_x.b)

        return encoder




def end2end_kernel_eo(self, X, Y, S):
        device = 'cuda'
        # dtype  = torch.float
        dtype  = torch.double

        y = self.format_y_onehot(Y)

        if self.hparams.dataset_options["onehot_s"]:
            s = self.format_s_onehot(S)
        else:
            s = S

        mask = (Y == 1)

        n = len(X)

        if self.rff_flag:
            R_x       = self.kernel_x(X).to(dtype=dtype, device=device)
            R_x_c     = misc.mean_center(R_x, dim=0).to(dtype=dtype, device=device)
            dtype = R_x.dtype

            R_x_y       = self.kernel_x(X[mask]).to(dtype=dtype, device=device)
            R_x_y_c     = misc.mean_center(R_x_y, dim=0).to(dtype=dtype, device=device)

            R_y = self.kernel_y(y).to(dtype=dtype, device=device)
            R_y_c = misc.mean_center(R_y, dim=0).to(dtype=dtype, device=device)

            R_s_y = self.kernel_s(s[mask]).to(dtype=dtype, device=device)
            R_s_y_c = misc.mean_center(R_s_y, dim=0).to(dtype=dtype, device=device)

            b_y = torch.mm(R_x.t(), R_y_c)
            b_y = torch.mm(b_y, b_y.t())

            b_s_y = torch.mm(R_x_y.t(), R_s_y_c)
            b_s_y = torch.mm(b_s_y, b_s_y.t())

            norm2_b_y = torch.linalg.norm(b_y, 2)
            norm2_b_s = torch.linalg.norm(b_s_y, 2)

            b = b_y / norm2_b_y - self.hparams.tau / (1. - self.hparams.tau)  * b_s_y / norm2_b_s
            
            self.norm2_b_y = norm2_b_y / n
            self.norm2_b_s = norm2_b_s / sum(mask)

            b = (b + b.t()) / 2

            # H^2 = H
            c = torch.mm(R_x_c.t(), R_x_c) + n * self.hparams.gamma * torch.eye(R_x.shape[1], device=R_x.device)
            c = (c + c.t()) / 2


            eigs, V = torch.linalg.eig(torch.mm(torch.linalg.inv(c), b))
            eigs = torch.real(eigs)
            V = torch.real(V)

            sorted, indeces = torch.sort(eigs, descending=True)

            # import pdb; pdb.set_trace()

            U = V[:, indeces[0:self.hparams.dim_z]]
            
            #########################################
            r0 = self.hparams.dim_z
            if self.hparams.tau == 0:
                r = r0
            else:
                r1 = min((sorted > 0).sum(), r0)

            ###### Energy Thresholding ######
                if r1 > 0:
                    for k in range(1, r1+1):
                        if torch.linalg.norm(sorted[0:k])**2 / torch.linalg.norm(sorted[0:r1])**2 >= 0.95:
                            r = k
                            break
                else:
                    r = 0
            ######################################################
            if self.hparams.tau >= 0.999999999:
                r = 0
            ######################################################
            U[:, r:self.hparams.dim_z] = 0

            self.hparams.auto_dim_z = r
                        
            encoder = models.KernelizedEncoder(U = n**0.5 * U, w=self.kernel_x.w, b=self.kernel_x.b)
        
        else:
            R_x       = self.kernel_x(X).to(dtype=dtype, device=device)
            if len(R_x.shape) == 1: R_x = R_x.unsqueeze(-1)
            R_x_c     = misc.mean_center(R_x, dim=0).to(dtype=dtype, device=device)
            dtype = R_x.dtype

            R_x_y       = self.kernel_x(X[mask]).to(dtype=dtype, device=device)
            if len(R_x_y.shape) == 1: R_x_y = R_x_y.unsqueeze(-1)
            R_x_y_c     = misc.mean_center(R_x_y, dim=0).to(dtype=dtype, device=device)

            R_y = self.kernel_y(y).to(dtype=dtype, device=device)
            if len(R_y.shape) == 1: R_y = R_y.unsqueeze(-1)
            R_y_c = misc.mean_center(R_y, dim=0).to(dtype=dtype, device=device)

            R_s_y = self.kernel_s(s[mask]).to(dtype=dtype, device=device)
            if len(R_s_y.shape) == 1: R_s_y = R_s_y.unsqueeze(-1)
            R_s_y_c = misc.mean_center(R_s_y, dim=0).to(dtype=dtype, device=device)

            b_y = torch.mm(R_x.t(), R_y_c)
            b_y = torch.mm(b_y, b_y.t())

            b_s_y = torch.mm(R_x_y.t(), R_s_y_c)

            b_s_y = torch.mm(b_s_y, b_s_y.t())

            norm2_b_y = torch.linalg.norm(b_y, 2)
            norm2_b_s = torch.linalg.norm(b_s_y, 2)

            b = b_y / norm2_b_y - self.hparams.tau / (1. - self.hparams.tau)  * b_s_y / norm2_b_s
            
            self.norm2_b_y = norm2_b_y / n
            self.norm2_b_s = norm2_b_s / sum(mask)

            b = (b + b.t()) / 2

            # H^2 = H
            c = torch.mm(R_x_c.t(), R_x_c) + n * self.hparams.gamma * torch.eye(R_x.shape[1], device=R_x.device)
            c = (c + c.t()) / 2

            eigs, V = torch.linalg.eig(torch.mm(torch.linalg.inv(c), b))
            eigs = torch.real(eigs)
            V = torch.real(V)

            sorted, indeces = torch.sort(eigs, descending=True)

            U = V[:, indeces[0:self.hparams.dim_z]]


            #########################################
            r0 = self.hparams.dim_z
            if self.hparams.tau == 0:
                r = r0
            else:
                r1 = min((sorted > 0).sum(), r0)

            ###### Energy Thresholding ######
                if r1 > 0:
                    for k in range(1, r1+1):
                        if torch.linalg.norm(sorted[0:k])**2 / torch.linalg.norm(sorted[0:r1])**2 >= 0.95:
                            r = k
                            break
                else:
                    r = 0
            ######################################################
            if self.hparams.tau >= 0.999999999:
                r = 0
            ######################################################
            U[:, r:self.hparams.dim_z] = 0

            self.hparams.auto_dim_z = r
            encoder = models.LinearEncoder(U = n**0.5 * U)

        return encoder



def end2end_kernel_eoo(self, X, Y, S):
        device = 'cuda'
        # dtype  = torch.float
        dtype  = torch.double

        y = self.format_y_onehot(Y)

        if self.hparams.dataset_options["onehot_s"]:
            s = self.format_s_onehot(S)
        else:
            s = S

        n = len(X)

        if self.rff_flag:
            R_x       = self.kernel_x(X).to(dtype=dtype, device=device)
            R_x_c     = misc.mean_center(R_x, dim=0).to(dtype=dtype, device=device)
            dtype = R_x.dtype

            R_y   = self.kernel_y(y).to(dtype=dtype, device=device)
            R_y_c = misc.mean_center(R_y, dim=0).to(dtype=dtype, device=device)


            b_y = torch.mm(R_x.t(), R_y_c)
            b_y = torch.mm(b_y, b_y.t())

            norm2_b_y = torch.linalg.norm(b_y, 2)

            # EOO Conditions
            norm2_b_s = dict()
            b_s_y = torch.zeros_like(b_y).to(dtype=b_y.dtype, device=b_y.device)
            for label in Y.unique():
                mask        = (Y == label)

                R_x_y       = self.kernel_x(X[mask]).to(dtype=dtype, device=device)
                R_x_y_c     = misc.mean_center(R_x_y, dim=0).to(dtype=dtype, device=device)

                R_s_y = self.kernel_s(s[mask]).to(dtype=dtype, device=device)
                R_s_y_c = misc.mean_center(R_s_y, dim=0).to(dtype=dtype, device=device)
                if len(R_s_y_c.shape) == 1:
                    R_s_y_c = R_s_y_c.unsqueeze(-1)
                else:
                    pass
                
                b_s_y_temp = torch.mm(R_x_y.t(), R_s_y_c)
                b_s_y_temp = torch.mm(b_s_y_temp, b_s_y_temp.t())

                norm2_b_s[label.item()] = torch.linalg.norm(b_s_y_temp, 2)

                b_s_y += b_s_y_temp 
                

            b = b_y / norm2_b_y - self.hparams.tau / (1. - self.hparams.tau)  * b_s_y

            self.norm2_b_y = norm2_b_y / n
            self.norm2_b_s = dict()
            for k in norm2_b_s.keys():
                self.norm2_b_s[k] = norm2_b_s[k] / sum(mask)

            b = (b + b.t()) / 2

            # H^2 = H
            c = torch.mm(R_x_c.t(), R_x_c) + n * self.hparams.gamma * torch.eye(R_x.shape[1], device=R_x.device)
            c = (c + c.t()) / 2

            eigs, V = torch.linalg.eig(torch.mm(torch.linalg.inv(c), b))
            eigs = torch.real(eigs)
            V = torch.real(V)

            sorted, indeces = torch.sort(eigs, descending=True)


            U = V[:, indeces[0:self.hparams.dim_z]]


            #########################################
            r0 = self.hparams.dim_z
            if self.hparams.tau == 0:
                r = r0
            else:
                r1 = min((sorted > 0).sum(), r0)

            ###### Energy Thresholding ######
                if r1 > 0:
                    for k in range(1, r1+1):
                        if torch.linalg.norm(sorted[0:k])**2 / torch.linalg.norm(sorted[0:r1])**2 >= 0.95:
                            r = k
                            break
                else:
                    r = 0
            ######################################################
            if self.hparams.tau >= 0.999999999:
                r = 0
            ######################################################
            U[:, r:self.hparams.dim_z] = 0

            self.hparams.auto_dim_z = r
                        
            encoder = models.KernelizedEncoder(U = n**0.5 * U, w=self.kernel_x.w, b=self.kernel_x.b)
        
        else:
            R_x       = self.kernel_x(X).to(dtype=dtype, device=device)
            R_x_c     = misc.mean_center(R_x, dim=0).to(dtype=dtype, device=device)
            dtype = R_x.dtype

            R_y = self.kernel_y(y).to(dtype=dtype, device=device)
            R_y_c = misc.mean_center(R_y, dim=0).to(dtype=dtype, device=device)
            R_y_c = misc.mean_center(R_y_c.t(), dim=0).to(dtype=dtype, device=device)

            R_s = self.kernel_s(s).to(dtype=dtype, device=device)
            R_s_c = misc.mean_center(R_s, dim=0).to(dtype=dtype, device=device)
            R_s_c = misc.mean_center(R_s_c, dim=0).to(dtype=dtype, device=device)

            b_y = torch.mm(torch.mm(R_x.t(), R_y_c), R_x)

            b_s = torch.mm(torch.mm(R_x.t(), R_s_c), R_x)

            b = b_y / torch.linalg.norm(b_y, 2) - self.hparams.tau / (1. - self.hparams.tau)  * b_s / torch.linalg.norm(b_s, 2)

            b = (b + b.t()) / 2

            # H^2 = H
            c = torch.mm(R_x_c.t(), R_x_c) + n * self.hparams.gamma * torch.eye(R_x.shape[1], device=R_x.device)
            c = (c + c.t()) / 2

            eigs, V = torch.linalg.eig(torch.mm(torch.linalg.inv(c), b))
            eigs = torch.real(eigs)
            V = torch.real(V)

            sorted, indeces = torch.sort(eigs, descending=True)


            U = V[:, indeces[0:self.hparams.dim_z]]

            #########################################
            r0 = self.hparams.dim_z
            if self.hparams.tau == 0:
                r = r0
            else:
                r1 = min((sorted > 0).sum(), r0)

            ###### Energy Thresholding ######
                if r1 > 0:
                    for k in range(1, r1+1):
                        if torch.linalg.norm(sorted[0:k])**2 / torch.linalg.norm(sorted[0:r1])**2 >= 0.92:
                            r = k
                            break
                else:
                    r = 0
            ######################################################
            if self.hparams.tau >= 0.99999:
                r = 0
            ###############################################################################
            U[:, r:self.hparams.dim_z] = 0

            self.hparams.auto_dim_z = r
            
            encoder = models.KernelizedEncoder(U=U, w=self.kernel_x.w, b=self.kernel_x.b)

        return encoder

