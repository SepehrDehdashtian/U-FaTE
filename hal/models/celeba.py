# celeba.py

import torch
from torch import nn

# from torch_geometric.nn import GCNConv
# from dgl.nn.pytorch import GATConv
# from dgl.nn.pytorch import GraphConv
# # from torch_geometric.nn import GINConv
# from dgl.nn import GINConv

__all__ = ['EncCelebA', 'EncCelebAGIN', 'EncCelebAGCN', 'TgtCelebA', 'SmallTgtCelebA', 'SmallTgtCelebA2', 'AdvCelebA', 'CelebAEmbeddingYS']


class EncCelebA(nn.Module):  # hdl:128 r:1
    def __init__(self, ndim, r, hdl):
        super().__init__()

        self.enc_equal = nn.Sequential(
            nn.Linear(ndim, hdl),
            nn.PReLU(),
            nn.Linear(hdl, hdl // 2),
            nn.PReLU(),
            nn.Linear(hdl // 2, r)
        )

    def forward(self, x, y=None):
        z = self.enc_equal(x)
        return z

class EncCelebAGIN(nn.Module):
    def __init__(self, ndim, hdl, r, dropout): 
        super(EncCelebAGIN, self).__init__()

        self.mlp1 = nn.Sequential(
            nn.Linear(ndim, hdl), 
            nn.ReLU(),
            nn.BatchNorm1d(hdl),
            nn.Linear(hdl, hdl),
        )
        self.conv1 = GINConv(self.mlp1)
        self.fc = nn.Linear(hdl, r)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        
    def forward(self, x, dglG): 
        z = self.conv1(dglG, x)
        z = self.fc(z)
        return z


class EncCelebAGCN(nn.Module):
    def __init__(self, ndim, r, hdl=None):
        super().__init__()
        self.gc1 = GraphConv(ndim, hdl)
        self.gc2 = GraphConv(hdl, r)
        # self.gc3 = GraphConv(hdl//2, r)

    def forward(self, x, dglG, y=None):
        # import pdb; pdb.set_trace()
        z = self.gc1(dglG, x)
        z = self.gc2(dglG, z)
        # z = self.gc3(dglG, z)
        return z

class TgtCelebA(nn.Module):  # hdl:128
    def __init__(self, nout, r, hdl):
        super().__init__()

        self.dec_equal2 = nn.Sequential(
            nn.Linear(r, hdl),
            nn.PReLU(),
            nn.Linear(hdl, hdl // 2),
            nn.PReLU(),
            nn.Linear(hdl // 2, nout)
        )

    def forward(self, z, y=None):
        out = self.dec_equal2(z)
        return out

class SmallTgtCelebA(nn.Module):  # hdl:128
    def __init__(self, nout, r, hdl):
        super().__init__()

        self.dec_2Layer = nn.Sequential(
            nn.Linear(r, hdl),
            nn.PReLU(),
            nn.Linear(hdl, nout)
        )

    def forward(self, z, y=None):
        out = self.dec_2Layer(z)
        return out

class SmallTgtCelebA2(nn.Module):  # hdl:128
    def __init__(self, nout, r, hdl):
        super().__init__()

        self.dec_equal2 = nn.Sequential(
            nn.Linear(r, hdl),
            nn.PReLU(),
            nn.Linear(hdl, hdl // 2),
            nn.PReLU(),
            nn.Linear(hdl // 2, nout)
        )

    def forward(self, z, y=None):
        out = self.dec_equal2(z)
        return out


class AdvCelebA(nn.Module):
    def __init__(self, nout, r, hdl):
        super().__init__()

        self.adv = nn.Sequential(
            nn.Linear(r, hdl),
            nn.PReLU(),
            nn.Linear(hdl, nout),
        )

    def forward(self, z, y=None):
        out = self.adv(z)
        return out
    

class CelebAEmbeddingYS(nn.Module):
    def __init__(self):
        super(CelebAEmbeddingYS, self).__init__()
        
        self.embedd_y = nn.Embedding(2, 1)
        self.embedd_s = nn.Embedding(4, 2)

        # self.fc1 = nn.Linear(sum(num_embedding) + 1, 64)

    def forward(self, x):
        # Apply the embeddings to the features and concat. them
        x0 = self.embedd_s(x[:,0])
        x1 = self.embedd_y(x[:,1])

        # Add age attribute to the embeddings
        embedded = torch.cat((x0, x1), dim=1).float()

        # out = self.fc1(embedded)
        
        return embedded   