import numpy as np
import torch
import torch.nn as nn
import scipy.linalg as sl


class DEP:
    def __init__(self):
        pass
    def __call__(self, phi_z, phi_s, rff):
        if rff:
            n, _ = phi_z.shape  # batch size
            dep_zs = torch.norm(torch.mm(phi_z.t(), phi_s), p='fro') / n
            return dep_zs**2
        
        else:
            n, _ = phi_z.shape  # batch size
            dep_zs = torch.trace(torch.mm(phi_z, phi_s)) / n
            return torch.sqrt(dep_zs)

class HSIC:
    def __init__(self):
        pass

    def __call__(self, phi_z, phi_s, rff):
        if rff:
            n, _ = phi_z.shape  # batch size
            hsic_zz = torch.norm(torch.mm(phi_z.t(), phi_z), p='fro') / n
            hsic_ss = torch.norm(torch.mm(phi_s.t(), phi_s), p='fro') / n
            hsic_zs = torch.norm(torch.mm(phi_z.t(), phi_s), p='fro') / n
            # norm_2_z = torch.linalg.norm(torch.mm(phi_z.t(), phi_z), 2) / n
            # norm_2_s = torch.linalg.norm(torch.mm(phi_s.t(), phi_s), 2) / n

            return hsic_zs**2 / (hsic_zz * hsic_ss)
            # return hsic_zs**2 / (norm_2_z * norm_2_s)
        else:
            n, _ = phi_z.shape  # batch size
            hsic_zz = torch.trace(torch.mm(phi_z, phi_z)) / n
            hsic_ss = torch.trace(torch.mm(phi_s, phi_s)) / n
            hsic_zs = torch.trace(torch.mm(phi_z, phi_s)) / n

            return torch.sqrt(hsic_zs / torch.sqrt(hsic_zz * hsic_ss))

class NHSIC:
    def __init__(self, lam):
        self.lam = lam

    def __call__(self, phi_z, phi_s, rff):
        if rff:
            n, mz = phi_z.shape  # batch size
            _, ms = phi_s.shape  # batch size
            sigma_z_half = (torch.mm(phi_z.t(), phi_z) + self.lam * torch.eye(mz, device=phi_z.device, dtype=phi_z.dtype)).cpu().numpy()
            sigma_z_half = torch.linalg.inv(torch.from_numpy((sl.sqrtm(sigma_z_half)).real).cuda().double()).type(phi_z.type())
            sigma_s_half = (torch.mm(phi_s.t(), phi_s) + self.lam * torch.eye(ms, device=phi_s.device, dtype=phi_s.dtype)).cpu().numpy()
            sigma_s_half = torch.linalg.inv(torch.from_numpy((sl.sqrtm(sigma_s_half)).real).cuda().double()).type(phi_s.type())
            # import pdb; pdb.set_trace()
            nhsic_zs = torch.norm(torch.mm(torch.mm(sigma_z_half, phi_z.t()), torch.mm(phi_s, sigma_s_half)), p='fro') / n
            nhsic_zz = torch.norm(torch.mm(torch.mm(sigma_z_half, phi_z.t()), torch.mm(phi_z, sigma_z_half)), p='fro') / n
            nhsic_ss = torch.norm(torch.mm(torch.mm(sigma_s_half, phi_s.t()), torch.mm(phi_s, sigma_s_half)), p='fro') / n
            return nhsic_zs / torch.sqrt(nhsic_zz * nhsic_ss)
        else:
            return torch.Tensor([0.])

class KCC:
    def __init__(self, lam):
        self.lam = lam

    def __call__(self, phi_z, phi_s, rff):
        if rff:
            n = phi_z.shape[0]

            m_z = phi_z.shape[1]
            m_s = phi_s.shape[1]

            c_zz = torch.mm(phi_z.t(), phi_z) / n
            c_ss = torch.mm(phi_s.t(), phi_s) / n
            c_zs = torch.mm(phi_z.t(), phi_s) / n

            psd_z = torch.linalg.inv(c_zz + self.lam * torch.eye(m_z, device=c_zz.device))
            psd_s = torch.linalg.inv(c_ss + self.lam * torch.eye(m_s, device=c_ss.device))
            zeros_z = torch.zeros(m_z, m_z, device=c_zs.device)
            zeros_s = torch.zeros(m_s, m_s, device=c_zs.device)

            a_zs = torch.cat((zeros_z, torch.mm(psd_z, c_zs)), dim=1)
            a_sz = torch.cat((torch.mm(psd_s, c_zs.t()), zeros_s), dim=1)
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

            return kcc_zs / torch.sqrt(kcc_zz*kcc_ss)

        else:
            return torch.Tensor([0.])

class COCO:
    def __init__(self):
        pass

    def __call__(self, phi_z, phi_s, rff):
        if rff:
            n = phi_z.shape[0]
            coco_zs = torch.linalg.norm(torch.mm(phi_z.t(), phi_s), 2) / n
            coco_zz = torch.linalg.norm(torch.mm(phi_z.t(), phi_z), 2) / n
            coco_ss = torch.linalg.norm(torch.mm(phi_s.t(), phi_s), 2) / n
            return coco_zs**2 / (coco_zz*coco_ss)
        else:
            n = phi_z.shape[0]
            coco_zs = torch.linalg.norm(torch.mm(phi_z, phi_s), 2) / (n**2)
            coco_zz = torch.linalg.norm(torch.mm(phi_z, phi_z), 2) / (n**2)
            coco_ss = torch.linalg.norm(torch.mm(phi_s, phi_s), 2) / (n**2)
            return torch.sqrt(coco_zs / torch.sqrt(coco_zz*coco_ss))

class Hyb:
    def __init__(self, gamma, dir='sym'):
        self.gamma = gamma
        self.dir = dir

    def __call__(self, phi_z, phi_s, rff):
        if rff:
            #####################################################################################################
            def Hyb_12(phi_1, phi_2, gamma):
                n = phi_1.shape[0]
                b = torch.mm(phi_1.t(), phi_2)
                b = torch.mm(b, b.t())
                # b = (b + b.t()) / 2

                a = torch.mm(phi_1.t(), phi_1)
                a += gamma * n * torch.eye(a.shape[0], device=a.device)

                b /= n

                # if b.shape[0] > 2:
                #     eigs, _ = torch.lobpcg(b, B=a, k=1, method='ortho', largest=True)
                #     eigs_12 = eigs[0]
                # else:
                #     c = torch.mm(torch.linalg.inv(a), b)
                #     eigs_12 = torch.max(torch.real(torch.linalg.eigvals(c)))

                c = torch.mm(torch.linalg.inv(a), b)
                eigs_12 = torch.max(torch.real(torch.linalg.eigvals(c)))
                # temp3 = torch.max(torch.real(torch.linalg.eigvals(c/2+c.t()/2)))
                # temp4 = torch.max(torch.linalg.eigvalsh(c))
                # eigs_12 = torch.max(torch.linalg.eigvalsh(c / 2 + c.t() / 2))
                # import pdb; pdb.set_trace()

                norm_2 = torch.linalg.norm(torch.mm(phi_2.t(), phi_2), 2) / n
                return torch.sqrt(eigs_12 / norm_2)
                # return eigs_12

            #############################################################################################
            if self.dir == 'fw':
                return Hyb_12(phi_z, phi_s, self.gamma)
            elif self.dir == 'bw':
                return Hyb_12(phi_s, phi_z, self.gamma)
            else:
                raise ValueError(f"Unsupported dir {self.dir}")
        else:
            return torch.Tensor([0.])

class HybSym:
    def __init__(self, gamma):
        self.gamma = gamma
    def __call__(self, phi_z, phi_s, rff):
        if rff:
            #####################################################################################################
            def Hyb_12(phi_1, phi_2, gamma):
                n = phi_1.shape[0]
                b = torch.mm(phi_1.t(), phi_2)
                b = torch.mm(b, b.t())
                # b = (b + b.t()) / 2

                a = torch.mm(phi_1.t(), phi_1)
                a += gamma * n * torch.eye(a.shape[0], device=a.device)
                b /= n
                # if b.shape[0] > 2:
                #     eigs, _ = torch.lobpcg(b, B=a, k=1, method='ortho', largest=True)
                #     eigs_12 = eigs[0]
                # else:
                #     c = torch.mm(torch.linalg.inv(a), b)
                #     eigs_12 = torch.max(torch.real(torch.linalg.eigvals(c)))

                c = torch.mm(torch.linalg.inv(a), b)
                eigs_12 = torch.max(torch.real(torch.linalg.eigvals(c)))
                # eigs_12 = torch.max(torch.linalg.eigvalsh(c / 2 + c.t() / 2))

                norm_2 = torch.linalg.norm(torch.mm(phi_2.t(), phi_2), 2) / n
                return eigs_12 / norm_2
                # return eigs_12

            #############################################################################################
            hyb_zs = Hyb_12(phi_z, phi_s, self.gamma)
            hyb_sz = Hyb_12(phi_s, phi_z, self.gamma)
            return torch.sqrt(max(hyb_zs, hyb_sz))

        else:
            return torch.Tensor([0.])


class NHyb:
    def __init__(self, lam, gamma, dir='sym'):
        self.gamma = gamma
        self.lam = lam
        self.dir = dir
    def __call__(self, phi_z, phi_s, rff):
        if rff:
            #####################################################################################################
            def NHyb_12(phi_1, phi_2, lam, gamma):
                n = phi_1.shape[0]
                m2 = phi_2.shape[1]
                sigma_2_half = (torch.mm(phi_2.t(), phi_2) + lam * torch.eye(m2, device=phi_2.device,
                                                                            dtype=phi_2.dtype)).cpu().numpy()
                sigma_2_half = torch.linalg.inv(torch.from_numpy((sl.sqrtm(sigma_2_half)).real).double().to(device=phi_2.device)).float()
                b = torch.mm(torch.mm(phi_1.t(), phi_2), sigma_2_half)
                b = torch.mm(b, b.t())
                # b = (b + b.t()) / 2

                a = torch.mm(phi_1.t(), phi_1)
                a += gamma * n * torch.eye(a.shape[0], device=a.device)

                if b.shape[0] > 2:
                    eigs, _ = torch.lobpcg(b, B=a, k=1, method='ortho', largest=True)
                    eigs_12 = eigs[0]
                else:
                    c = torch.mm(torch.linalg.inv(a), b)
                    eigs_12 = torch.max(torch.linalg.eigvalsh(c))

                return eigs_12

            #############################################################################################
            if self.dir == 'fw':
                return NHyb_12(phi_z, phi_s, self.lam, self.gamma)

            elif self.dir == 'bw':
                return NHyb_12(phi_s, phi_z, self.lam, self.gamma)

            elif self.dir == 'sym':
                nhyb_zs = NHyb_12(phi_z, phi_s, self.lam, self.gamma)
                nhyb_sz = NHyb_12(phi_s, phi_z, self.lam, self.gamma)
                nhyb_zz = NHyb_12(phi_z, phi_z, self.lam, self.gamma)
                nhyb_ss = NHyb_12(phi_s, phi_s, self.lam, self.gamma)
                return max(nhyb_zs, nhyb_sz) / torch.sqrt(nhyb_zz*nhyb_ss)

        else:
            return torch.Tensor([0.])