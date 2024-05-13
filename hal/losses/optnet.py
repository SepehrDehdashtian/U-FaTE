import torch
from torch import nn

__all__ = ['Projection', 'Projection_poly', 'Projection_IMQ', 'ProjectionGaussian', 'ProjectionGaussianEO', 'ProjectionGaussianEOO', 'Projection_gauss_linear', 'Projection_ideal']

class Projection_ideal:
    def __init__(self):
        # self.eye = torch.eye(1).to(device)
        pass

    def __call__(self, S, Y, lam, reg, device):


        S_bar = S - torch.mean(S, dim=0)

        Y_bar = Y - torch.mean(Y, dim=0)

        # Project_Y = (torch.norm(torch.mm(P_M, Y_bar)) ** 2 )/batch_size

        B = lam*torch.mm(S_bar, torch.t(S_bar)) - (1-lam)*torch.mm(Y_bar, torch.t(Y_bar))
        U, Sig, V = torch.eig(B)
        # import pdb; pdb.set_trace()
        # loss = lam*Project_S / (1-lam)*Project_Y

        return  U


class Projection:
    def __init__(self):
        # self.eye = torch.eye(1).to(device)
        pass

    def __call__(self, Z, S, Y, lam, reg, device):
        # import pdb;
        # pdb.set_trace()
        # if lam>0.2:
        # Z = Z / torch.norm(Z, 'fro')
        # Z = Z / torch.max(Z)
        M = Z - torch.mean(Z, dim=0) # Z is already transposed
        # import pdb; pdb.set_trace()
        # M = M/torch.norm(M,'fro')

        batch_size = Z.size(0)

        P1 = torch.mm(torch.t(M), M)
        # self.eye.resize_(P1.shape[0])

        P2 = torch.inverse(P1+ reg*torch.eye(P1.shape[0]).to(device))
        # P2 = torch.inverse(P1+ reg*self.eye)
        P3 = torch.mm(P2, torch.t(M))
        P_M = torch.mm(M, P3)

        # U, Sigma, V = torch.svd(M)
        # import pdb; pdb.set_trace()
        # P_M = torch.mm(U, torch.t(U))

        S_bar = S - torch.mean(S, dim=0)
        Project_S = (torch.norm(torch.mm(P_M, S_bar)) ** 2 + reg*torch.norm(torch.mm(P3, S_bar)) ** 2)/torch.norm(S_bar) ** 2
        # Project_S = (torch.norm(torch.mm(P_M, S_bar)) ** 2)/batch_size

        Y_bar = Y - torch.mean(Y, dim=0)
        # import pdb; pdb.set_trace()
        Project_Y = (torch.norm(torch.mm(P_M, Y_bar)) ** 2  + reg*torch.norm(torch.mm(P3, Y_bar)) ** 2)/torch.norm(Y_bar) ** 2
        # Project_Y = (torch.norm(torch.mm(P_M, Y_bar)) ** 2 )/batch_size

        loss = lam*Project_S - (1-lam)*Project_Y
        # import pdb; pdb.set_trace()
        # loss = lam*Project_S / (1-lam)*Project_Y

        return  loss, Project_S, Project_Y


class Projection_poly:
    def __init__(self):
        # self.eye = torch.eye(1).to(device)
        pass

    def __call__(self, Z, S, Y, lam, reg, device, c, d):
        # import pdb; pdb.set_trace()

        # M = Z - torch.mean(Z, dim=0) # Z is already transposed
        # Z = Z / torch.norm(Z, 'fro')

        # Z = Z / torch.max(Z)
        K = torch.mm(Z,torch.t(Z)) # Z is already transposed


        # K = torch.matrix_power(K+c*torch.eye(K.shape[0]).to(device),d)

        # K = torch.pow(K+c*torch.eye(K.shape[0]).to(device),d)
        K = torch.pow(K+c,d)
        batch_size = Z.size(0)
        # M = M/torch.norm(M,'fro')

        D = torch.eye(batch_size) - torch.ones([batch_size, batch_size])/batch_size
        D = D.to(device)
        M = torch.mm(torch.mm(D,K), D)

        # import pdb; pdb.set_trace()

        P1 = torch.mm(torch.t(M), M)
        # self.eye.resize_(P1.shape[0])

        # import pdb; pdb.set_trace()
        P2 = torch.inverse(P1+ reg*torch.eye(P1.shape[0]).to(device))
        # P2 = torch.inverse(P1+ reg*self.eye)
        P3 = torch.mm(P2, torch.t(M))
        P_M = torch.mm(M, P3)

        # U, Sigma, V = torch.svd(M)
        # import pdb; pdb.set_trace()
        # P_M = torch.mm(U, torch.t(U))

        S_bar = S - torch.mean(S, dim=0)
        Project_S = (torch.norm(torch.mm(P_M, S_bar)) ** 2 + reg*torch.norm(torch.mm(P3, S_bar)) ** 2)/torch.norm(S_bar) ** 2
        # Project_S = (torch.norm(torch.mm(P_M, S_bar)) ** 2 )/batch_size

        Y_bar = Y - torch.mean(Y, dim=0)
        Project_Y = (torch.norm(torch.mm(P_M, Y_bar)) ** 2 + reg*torch.norm(torch.mm(P3, Y_bar)) ** 2)/torch.norm(Y_bar) ** 2
        # Project_Y = (torch.norm(torch.mm(P_M, Y_bar)) ** 2)/batch_size

        loss = lam*Project_S - (1-lam)*Project_Y
        # import pdb; pdb.set_trace()
        # loss = lam*Project_S / (1-lam)*Project_Y

        return  loss, Project_S, Project_Y

class Projection_IMQ:
    def __init__(self):
        # self.eye = torch.eye(1).to(device)
        pass

    def __call__(self, Z, S, Y, lam, reg, device, c):
        # import pdb; pdb.set_trace()

        # M = Z - torch.mean(Z, dim=0) # Z is already transposed
        Z = Z / torch.norm(Z, 'fro')
        # import pdb;
        # pdb.set_trace()
        # Z = Z / torch.max(Z)
        batch_size = Z.size(0)
        ONES = torch.ones([1,batch_size]).to(device)
        NORM = torch.norm(Z,dim=1).reshape([1,batch_size])
        NORM = NORM ** 2
        K = c* torch.pow(torch.mm(torch.t(NORM), ONES) + torch.mm(torch.t(ONES), NORM) + 2*torch.mm(Z, torch.t(Z))+c, -1)
        # import pdb; pdb.set_trace()

        # K = torch.matrix_power(K+c*torch.eye(K.shape[0]).to(device),d)
        # K = torch.pow(K+c,d)
        # M = M/torch.norm(M,'fro')



        D = torch.eye(batch_size) - torch.ones([batch_size, batch_size])/batch_size
        D = D.to(device)
        M = torch.mm(torch.mm(D,K), D)

        # import pdb; pdb.set_trace()

        P1 = torch.mm(torch.t(M), M)
        # self.eye.resize_(P1.shape[0])

        # import pdb; pdb.set_trace()
        P2 = torch.inverse(P1+ reg*torch.eye(P1.shape[0]).to(device))
        # P2 = torch.inverse(P1+ reg*self.eye)
        P3 = torch.mm(P2, torch.t(M))
        P_M = torch.mm(M, P3)

        # U, Sigma, V = torch.svd(M)
        # import pdb; pdb.set_trace()
        # P_M = torch.mm(U, torch.t(U))

        S_bar = S - torch.mean(S, dim=0)
        Project_S = (torch.norm(torch.mm(P_M, S_bar)) ** 2 + reg*torch.norm(torch.mm(P3, S_bar)) ** 2)/torch.norm(S_bar) ** 2
        # Project_S = torch.norm(torch.mm(P_M, S_bar)) ** 2 /torch.norm(S_bar) ** 2
        # Project_S = (torch.norm(torch.mm(P_M, S_bar)) ** 2 )/batch_size

        Y_bar = Y - torch.mean(Y, dim=0)
        Project_Y = (torch.norm(torch.mm(P_M, Y_bar)) ** 2 + reg*torch.norm(torch.mm(P3, Y_bar)) ** 2)/torch.norm(Y_bar) ** 2
        # Project_Y = torch.norm(torch.mm(P_M, Y_bar)) ** 2 /torch.norm(Y_bar) ** 2
        # Project_Y = (torch.norm(torch.mm(P_M, Y_bar)) ** 2)/batch_size

        loss = lam*Project_S - (1-lam)*Project_Y
        # import pdb; pdb.set_trace()
        # loss = lam*Project_S / (1-lam)*Project_Y

        return  loss, Project_S, Project_Y

class ProjectionGaussian(nn.Module):
    def __init__(self, num_classes_y, num_classes_s):
        super().__init__()
        self.num_classes_y = num_classes_y
        self.num_classes_s = num_classes_s

    def forward(self, Z, S, Y, lam, reg, sigma):
        Y = self.format_y_onehot(Y)
        S = self.format_s_onehot(S)

        # M = Z - torch.mean(Z, dim=0) # Z is already transposed
        Z = Z / torch.norm(Z, 'fro')

        # Z = Z / torch.max(Z)
        batch_size = Z.size(0)
        device = Z.device

        # sigma = rbfsigma(Z, batch_size)
        ONES = torch.ones([1,batch_size]).to(device)
        NORM = torch.norm(Z, dim=1).reshape([1 ,batch_size])
        NORM = NORM ** 2

        K = torch.exp((-torch.mm(torch.t(NORM), ONES)-torch.mm(torch.t(ONES), NORM) + 2 * torch.mm(Z, torch.t(Z))) / (2*sigma**2)) # Z is already transposed

        # K = torch.matrix_power(K+c*torch.eye(K.shape[0]).to(device),d)
        # K = torch.pow(K+c,d)
        # M = M/torch.norm(M,'fro')

        D = torch.eye(batch_size) - torch.ones([batch_size, batch_size]) / batch_size
        D = D.to(device)
        M = torch.mm(torch.mm(D,K), D)

        P1 = torch.mm(torch.t(M), M)
        # self.eye.resize_(P1.shape[0])


        P2 = torch.inverse(P1+ reg*torch.eye(P1.shape[0]).to(device))
        # P2 = torch.inverse(P1+ reg*self.eye)
        P3 = torch.mm(P2, torch.t(M))
        P_M = torch.mm(M, P3)

        # U, Sigma, V = torch.svd(M)
        
        # P_M = torch.mm(U, torch.t(U))

        S_bar = S - torch.mean(S, dim=0)
        Project_S = (torch.norm(torch.mm(P_M, S_bar)) ** 2 + reg*torch.norm(torch.mm(P3, S_bar)) ** 2)/torch.norm(S_bar) ** 2
        # Project_S = torch.norm(torch.mm(P_M, S_bar)) ** 2 /torch.norm(S_bar) ** 2
        # Project_S = (torch.norm(torch.mm(P_M, S_bar)) ** 2 )/batch_size

        Y_bar = Y - torch.mean(Y, dim=0)
        Project_Y = (torch.norm(torch.mm(P_M, Y_bar)) ** 2 + reg*torch.norm(torch.mm(P3, Y_bar)) ** 2)/torch.norm(Y_bar) ** 2
        # Project_Y = torch.norm(torch.mm(P_M, Y_bar)) ** 2 /torch.norm(Y_bar) ** 2
        # Project_Y = (torch.norm(torch.mm(P_M, Y_bar)) ** 2)/batch_size

        loss = lam * Project_S - (1-lam) * Project_Y
        # import pdb; pdb.set_trace()
        # loss = lam*Project_S / (1-lam)*Project_Y

        return loss, Project_S, Project_Y

    def format_y_onehot(self, y):
        if len(y.shape) == 1:
            y = y.unsqueeze(1)
        y_onehot = torch.zeros(y.size(0), self.num_classes_y, device=y.device).scatter_(1, y.type(torch.int64), 1)
        return y_onehot
        
        
    def format_s_onehot(self, s):
        # int -> one-hot
        if len(s.shape) > 2:
            s = s.squeeze(-1)
        elif len(s.shape) == 1:
            s = s.unsqueeze(1)

        s_onehot = torch.zeros(s.size(0), self.num_classes_s, device=s.device).scatter_(1, s.long(), 1)
        return s_onehot


class ProjectionGaussianEO(nn.Module):
    def __init__(self, num_classes_y, num_classes_s):
        super().__init__()
        self.num_classes_y = num_classes_y
        self.num_classes_s = num_classes_s

    def forward(self, Z, S, Y, lam, reg, sigma):
        mask = (Y == 1)


        Y = self.format_y_onehot(Y)
        S = self.format_s_onehot(S)


        Z = Z / torch.norm(Z, 'fro')


        batch_size = Z.size(0)
        device = Z.device


        ONES = torch.ones([1,batch_size]).to(device)
        NORM = torch.norm(Z, dim=1).reshape([1 ,batch_size])
        NORM = NORM ** 2

        K = torch.exp((-torch.mm(torch.t(NORM), ONES)-torch.mm(torch.t(ONES), NORM) + 2 * torch.mm(Z, torch.t(Z))) / (2*sigma**2)) # Z is already transposed


        D = torch.eye(batch_size) - torch.ones([batch_size, batch_size]) / batch_size
        D = D.to(device)
        M = torch.mm(torch.mm(D,K), D)

        P1 = torch.mm(torch.t(M), M)


        P2 = torch.inverse(P1+ reg*torch.eye(P1.shape[0]).to(device))
        P3 = torch.mm(P2, torch.t(M))
        P_M = torch.mm(M, P3)

        
        Y_bar = Y - torch.mean(Y, dim=0)
        Project_Y = (torch.norm(torch.mm(P_M, Y_bar)) ** 2 + reg*torch.norm(torch.mm(P3, Y_bar)) ** 2)/torch.norm(Y_bar) ** 2
        


        # Mask data for adversary branch
        Y = Y[mask]
        S = S[mask]
        Z = Z[mask]

        Z = Z / torch.norm(Z, 'fro')


        batch_size = Z.size(0)
        device = Z.device


        ONES = torch.ones([1,batch_size]).to(device)
        NORM = torch.norm(Z, dim=1).reshape([1 ,batch_size])
        NORM = NORM ** 2

        K = torch.exp((-torch.mm(torch.t(NORM), ONES)-torch.mm(torch.t(ONES), NORM) + 2 * torch.mm(Z, torch.t(Z))) / (2*sigma**2)) # Z is already transposed


        D = torch.eye(batch_size) - torch.ones([batch_size, batch_size]) / batch_size
        D = D.to(device)
        M = torch.mm(torch.mm(D,K), D)

        P1 = torch.mm(torch.t(M), M)


        P2 = torch.inverse(P1+ reg*torch.eye(P1.shape[0]).to(device))
        P3 = torch.mm(P2, torch.t(M))
        P_M = torch.mm(M, P3)

        S_bar = S - torch.mean(S, dim=0)
        Project_S = (torch.norm(torch.mm(P_M, S_bar)) ** 2 + reg*torch.norm(torch.mm(P3, S_bar)) ** 2)/torch.norm(S_bar) ** 2


        loss = lam * Project_S - (1-lam) * Project_Y
        
        return loss, Project_S, Project_Y

    def format_y_onehot(self, y):
        if len(y.shape) == 1:
            y = y.unsqueeze(1)
        y_onehot = torch.zeros(y.size(0), self.num_classes_y, device=y.device).scatter_(1, y.type(torch.int64), 1)
        return y_onehot
        
        
    def format_s_onehot(self, s):
        # int -> one-hot
        if len(s.shape) > 2:
            s = s.squeeze(-1)
        elif len(s.shape) == 1:
            s = s.unsqueeze(1)

        s_onehot = torch.zeros(s.size(0), self.num_classes_s, device=s.device).scatter_(1, s.long(), 1)
        return s_onehot


class ProjectionGaussianEOO(nn.Module):
    def __init__(self, num_classes_y, num_classes_s):
        super().__init__()
        self.num_classes_y = num_classes_y
        self.num_classes_s = num_classes_s

    def forward(self, Z, S, Y, lam, reg, sigma):
        mask = (Y == 1)


        Y = self.format_y_onehot(Y)
        S = self.format_s_onehot(S)


        Z = Z / torch.norm(Z, 'fro')


        batch_size = Z.size(0)
        device = Z.device


        ONES = torch.ones([1,batch_size]).to(device)
        NORM = torch.norm(Z, dim=1).reshape([1 ,batch_size])
        NORM = NORM ** 2

        K = torch.exp((-torch.mm(torch.t(NORM), ONES)-torch.mm(torch.t(ONES), NORM) + 2 * torch.mm(Z, torch.t(Z))) / (2*sigma**2)) # Z is already transposed


        D = torch.eye(batch_size) - torch.ones([batch_size, batch_size]) / batch_size
        D = D.to(device)
        M = torch.mm(torch.mm(D,K), D)

        P1 = torch.mm(torch.t(M), M)


        P2 = torch.inverse(P1+ reg*torch.eye(P1.shape[0]).to(device))
        P3 = torch.mm(P2, torch.t(M))
        P_M = torch.mm(M, P3)

        
        Y_bar = Y - torch.mean(Y, dim=0)
        Project_Y = (torch.norm(torch.mm(P_M, Y_bar)) ** 2 + reg*torch.norm(torch.mm(P3, Y_bar)) ** 2)/torch.norm(Y_bar) ** 2
        
        # Mask data for adversary branch
        Y0 = Y[~mask]
        S0 = S[~mask]
        Z0 = Z[~mask]
        Y1 = Y[mask]
        S1 = S[mask]
        Z1 = Z[mask]

        Z0 = Z0 / torch.norm(Z0, 'fro')
        Z1 = Z1 / torch.norm(Z1, 'fro')


        batch_size0 = Z0.size(0)
        batch_size1 = Z1.size(0)
        device = Z0.device


        ONES0 = torch.ones([1,batch_size0]).to(device)
        ONES1 = torch.ones([1,batch_size1]).to(device)
        
        NORM0 = (torch.norm(Z0, dim=1).reshape([1 ,batch_size0])) ** 2
        NORM1 = (torch.norm(Z1, dim=1).reshape([1 ,batch_size1])) ** 2

        K0 = torch.exp((-torch.mm(torch.t(NORM0), ONES0)-torch.mm(torch.t(ONES0), NORM0) + 2 * torch.mm(Z0, torch.t(Z0))) / (2*sigma**2)) # Z is already transposed
        K1 = torch.exp((-torch.mm(torch.t(NORM1), ONES1)-torch.mm(torch.t(ONES1), NORM1) + 2 * torch.mm(Z1, torch.t(Z1))) / (2*sigma**2)) # Z is already transposed


        D0 = torch.eye(batch_size0) - torch.ones([batch_size0, batch_size0]) / batch_size0
        D0 = D0.to(device)
        D1 = torch.eye(batch_size1) - torch.ones([batch_size1, batch_size1]) / batch_size1
        D1 = D1.to(device)

        M0 = torch.mm(torch.mm(D0,K0), D0)
        M1 = torch.mm(torch.mm(D1,K1), D1)

        P10 = torch.mm(torch.t(M0), M0)
        P11 = torch.mm(torch.t(M1), M1)


        P20 = torch.inverse(P10+ reg*torch.eye(P10.shape[0]).to(device))
        P21 = torch.inverse(P11+ reg*torch.eye(P11.shape[0]).to(device))
        P30 = torch.mm(P20, torch.t(M0))
        P31 = torch.mm(P21, torch.t(M1))
        P_M0 = torch.mm(M0, P30)
        P_M1 = torch.mm(M1, P31)

        S_bar0 = S0 - torch.mean(S0, dim=0)
        S_bar1 = S1 - torch.mean(S1, dim=0)
        Project_S0 = (torch.norm(torch.mm(P_M0, S_bar0)) ** 2 + reg*torch.norm(torch.mm(P30, S_bar0)) ** 2)/torch.norm(S_bar0) ** 2
        Project_S1 = (torch.norm(torch.mm(P_M1, S_bar1)) ** 2 + reg*torch.norm(torch.mm(P31, S_bar1)) ** 2)/torch.norm(S_bar1) ** 2

        Project_S = Project_S0 + Project_S1

        loss = lam * Project_S - (1-lam) * Project_Y
        
        return loss, Project_S, Project_Y

    def format_y_onehot(self, y):
        if len(y.shape) == 1:
            y = y.unsqueeze(1)
        y_onehot = torch.zeros(y.size(0), self.num_classes_y, device=y.device).scatter_(1, y.type(torch.int64), 1)
        return y_onehot
        
        
    def format_s_onehot(self, s):
        # int -> one-hot
        if len(s.shape) > 2:
            s = s.squeeze(-1)
        elif len(s.shape) == 1:
            s = s.unsqueeze(1)

        s_onehot = torch.zeros(s.size(0), self.num_classes_s, device=s.device).scatter_(1, s.long(), 1)
        return s_onehot


class Projection_gauss_linear:
    def __init__(self):
        # self.eye = torch.eye(1).to(device)
        pass

    def __call__(self, Z, S, Y, lam, reg, device, sigma):
        # import pdb; pdb.set_trace()

        # M = Z - torch.mean(Z, dim=0) # Z is already transposed
        # Z = Z / torch.norm(Z, 'fro')
        batch_size = Z.size(0)
        ONES = torch.ones([1,batch_size]).to(device)
        NORM = torch.norm(Z,dim=1).reshape([1,batch_size])
        K = torch.exp((-torch.mm(torch.t(NORM), ONES)-torch.mm(torch.t(ONES), NORM)+2*torch.mm(Z, torch.t(Z)))/sigma) # Z is already transposed

        D = torch.eye(batch_size) - torch.ones([batch_size, batch_size])/batch_size
        D = D.to(device)
        M = torch.mm(torch.mm(D,K), D)

        P1 = torch.mm(torch.t(M), M)

        P2 = torch.inverse(P1+ reg*torch.eye(P1.shape[0]).to(device))
        # P2 = torch.inverse(P1+ reg*self.eye)
        P3 = torch.mm(P2, torch.t(M))
        P_M = torch.mm(M, P3)

        S_bar = S - torch.mean(S, dim=0)
        Project_S = (torch.norm(torch.mm(P_M, S_bar)) ** 2 + reg*torch.norm(torch.mm(P3, S_bar)) ** 2)/torch.norm(S_bar) ** 2
        # Project_S = (torch.norm(torch.mm(P_M, S_bar)) ** 2 )/batch_size

################################################ Linear
        M1 = Z - torch.mean(Z, dim=0)  # Z is already transposed
        # import pdb; pdb.set_trace()
        # M = M/torch.norm(M,'fro')

        Q1 = torch.mm(torch.t(M1), M1)
        # self.eye.resize_(P1.shape[0])

        Q2 = torch.inverse(Q1 + reg * torch.eye(Q1.shape[0]).to(device))
        # P2 = torch.inverse(P1+ reg*self.eye)
        Q3 = torch.mm(Q2, torch.t(M1))
        P_M1 = torch.mm(M1, Q3)

        Y_bar = Y - torch.mean(Y, dim=0)
        Project_Y = (torch.norm(torch.mm(P_M1, Y_bar)) ** 2 + reg*torch.norm(torch.mm(Q3, Y_bar)) ** 2)/torch.norm(Y_bar) ** 2
        # Project_Y = (torch.norm(torch.mm(P_M, Y_bar)) ** 2)/batch_size
##################################################

        loss = lam*Project_S - (1-lam)*Project_Y
        # import pdb; pdb.set_trace()
        # loss = lam*Project_S / (1-lam)*Project_Y

        return  loss, Project_S, Project_Y
