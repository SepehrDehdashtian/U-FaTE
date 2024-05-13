# folktables.py

import torch
from torch import nn

# from torch_geometric.nn import GCNConv
# from dgl.nn.pytorch import GATConv
# from dgl.nn.pytorch import GraphConv
# # from torch_geometric.nn import GINConv
# from dgl.nn import GINConv


__all__ = ['EncFolk', 'EncFolkGIN', 'EncFolkGCN', 'TgtFolk', 'SmallTgtFolk', 'SmallTgtFolk2', 
           'AdvFolk', 'FolkEmbedding', 'FolkEmbeddingXYS', 'FolkEmbeddingXY', 'FolkEmbeddingXYhat', 
           'FolkEmbeddingYS']


class EncFolk(nn.Module):  # hdl:128 r:1
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

        z = z / (torch.norm(z, dim=1, keepdim=True) + 1e-16)
        
        return z

class EncFolkGIN(nn.Module):
    def __init__(self, ndim, hdl, r, dropout): 
        super(EncFolkGIN, self).__init__()

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


class EncFolkGCN(nn.Module):
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

class TgtFolk(nn.Module):  # hdl:128
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

class SmallTgtFolk(nn.Module):  # hdl:128
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

class SmallTgtFolk2(nn.Module):  # hdl:128
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


class AdvFolk(nn.Module):
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


class FolkEmbedding(nn.Module):
    def __init__(self):
        super(FolkEmbedding, self).__init__()

        #'Emp' :: [SCHL, MAR, RELP, DIS, ESP, CIT, MIG, MIL, ANC, Nativity, DEAR, DEYE, DREM, SEX, RAC1P]
        attributes    = [25, 6, 18, 3, 9, 6, 4, 5, 5, 3, 3, 3, 3, 3, 10]
        num_embedding = [10, 3,  9, 3, 5, 3, 2, 3, 3, 2, 2, 2, 2, 2, 5]
        
        # self.embeddings = [nn.Embedding(num_classes, emb_dim, device='cuda') for (num_classes, emb_dim) in zip(attributes, num_embedding)]
        self.embedd1 = nn.Embedding(attributes[0], num_embedding[0])
        self.embedd2 = nn.Embedding(attributes[1], num_embedding[1])
        self.embedd3 = nn.Embedding(attributes[2], num_embedding[2])
        self.embedd4 = nn.Embedding(attributes[3], num_embedding[3])
        self.embedd5 = nn.Embedding(attributes[4], num_embedding[4])
        self.embedd6 = nn.Embedding(attributes[5], num_embedding[5])
        self.embedd7 = nn.Embedding(attributes[6], num_embedding[6])
        self.embedd8 = nn.Embedding(attributes[7], num_embedding[7])
        self.embedd9 = nn.Embedding(attributes[8], num_embedding[8])
        self.embedd10 = nn.Embedding(attributes[9], num_embedding[9])
        self.embedd11 = nn.Embedding(attributes[10], num_embedding[10])
        self.embedd12 = nn.Embedding(attributes[11], num_embedding[11])
        self.embedd13 = nn.Embedding(attributes[12], num_embedding[12])
        self.embedd14 = nn.Embedding(attributes[13], num_embedding[13])
        self.embedd15 = nn.Embedding(attributes[14], num_embedding[14])

    def forward(self, x):
        # Apply the embeddings to the features and concat. them
        # embedded = torch.cat([embedding(x[:, i+1]).reshape(-1,embedding.embedding_dim) for i, embedding in enumerate(self.embeddings)], dim=1)
        
        # x0  = x[:,0].reshape(-1, 1) / 95
        x0  = x[:,0].reshape(-1, 1)
        x1  = self.embedd1(x[:,1].int())
        x2  = self.embedd2(x[:,2].int())
        x3  = self.embedd3(x[:,3].int())
        x4  = self.embedd4(x[:,4].int())
        x5  = self.embedd5(x[:,5].int())
        x6  = self.embedd6(x[:,6].int())
        x7  = self.embedd7(x[:,7].int())
        x8  = self.embedd8(x[:,8].int())
        x9  = self.embedd9(x[:,9].int())
        x10 = self.embedd10(x[:,10].int())
        x11 = self.embedd11(x[:,11].int())
        x12 = self.embedd12(x[:,12].int())
        x13 = self.embedd13(x[:,13].int())
        x14 = self.embedd14(x[:,14].int())
        x15 = self.embedd15(x[:,15].int())

        # Add age attribute to the embeddings
        # embedded = torch.cat((x[:, 0].reshape(-1,1), embedded), dim=1)
        embedded = torch.cat((x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15), dim=1)

        return embedded


class FolkEmbeddingXYhat(nn.Module):
    def __init__(self):
        super(FolkEmbeddingXYhat, self).__init__()

        #'Emp' :: [SCHL, MAR, RELP, DIS, ESP, CIT, MIG, MIL, ANC, Nativity, DEAR, DEYE, DREM, SEX, RAC1P]
        attributes    = [25, 6, 18, 3, 9, 6, 4, 5, 5, 3, 3, 3, 3, 3, 10]
        num_embedding = [10, 3,  9, 3, 5, 3, 2, 3, 3, 2, 2, 2, 2, 2, 5]
        
        # self.embeddings = [nn.Embedding(num_classes, emb_dim, device='cuda') for (num_classes, emb_dim) in zip(attributes, num_embedding)]
        self.embedd1 = nn.Embedding(attributes[0], num_embedding[0])
        self.embedd2 = nn.Embedding(attributes[1], num_embedding[1])
        self.embedd3 = nn.Embedding(attributes[2], num_embedding[2])
        self.embedd4 = nn.Embedding(attributes[3], num_embedding[3])
        self.embedd5 = nn.Embedding(attributes[4], num_embedding[4])
        self.embedd6 = nn.Embedding(attributes[5], num_embedding[5])
        self.embedd7 = nn.Embedding(attributes[6], num_embedding[6])
        self.embedd8 = nn.Embedding(attributes[7], num_embedding[7])
        self.embedd9 = nn.Embedding(attributes[8], num_embedding[8])
        self.embedd10 = nn.Embedding(attributes[9], num_embedding[9])
        self.embedd11 = nn.Embedding(attributes[10], num_embedding[10])
        self.embedd12 = nn.Embedding(attributes[11], num_embedding[11])
        self.embedd13 = nn.Embedding(attributes[12], num_embedding[12])
        self.embedd14 = nn.Embedding(attributes[13], num_embedding[13])
        self.embedd15 = nn.Embedding(attributes[14], num_embedding[14])

    def forward(self, x):
        # Apply the embeddings to the features and concat. them
        # embedded = torch.cat([embedding(x[:, i+1]).reshape(-1,embedding.embedding_dim) for i, embedding in enumerate(self.embeddings)], dim=1)
        
        x1  = self.embedd1(x[:,0].int())
        x2  = self.embedd2(x[:,1].int())
        x3  = self.embedd3(x[:,2].int())
        x4  = self.embedd4(x[:,3].int())
        x5  = self.embedd5(x[:,4].int())
        x6  = self.embedd6(x[:,5].int())
        x7  = self.embedd7(x[:,6].int())
        x8  = self.embedd8(x[:,7].int())
        x9  = self.embedd9(x[:,8].int())
        x10 = self.embedd10(x[:,9].int())
        x11 = self.embedd11(x[:,10].int())
        x12 = self.embedd12(x[:,11].int())
        x13 = self.embedd13(x[:,12].int())
        x14 = self.embedd14(x[:,13].int())
        x15 = self.embedd15(x[:,14].int())

        x16 = x[:, 15:]

        # Add age attribute to the embeddings
        # embedded = torch.cat((x[:, 0].reshape(-1,1), embedded), dim=1)
        embedded = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16), dim=1)

        return embedded


class FolkEmbedding2(nn.Module):
    def __init__(self):
        super(FolkEmbedding, self).__init__()

        #'Emp' :: [SCHL, MAR, RELP, DIS, ESP, CIT, MIG, MIL, ANC, Nativity, DEAR, DEYE, DREM, SEX, RAC1P]
        attributes    = [25, 6, 18, 3, 9, 6, 4, 5, 5, 3, 3, 3, 3, 3, 10]
        num_embedding = [10, 3,  9, 3, 5, 3, 2, 3, 3, 2, 2, 2, 2, 2, 5]
        
        # self.embeddings = [nn.Embedding(num_classes, emb_dim, device='cuda') for (num_classes, emb_dim) in zip(attributes, num_embedding)]
        self.embedd1 = nn.Embedding(attributes[0], num_embedding[0])
        self.embedd2 = nn.Embedding(attributes[1], num_embedding[1])
        self.embedd3 = nn.Embedding(attributes[2], num_embedding[2])
        self.embedd4 = nn.Embedding(attributes[3], num_embedding[3])
        self.embedd5 = nn.Embedding(attributes[4], num_embedding[4])
        self.embedd6 = nn.Embedding(attributes[5], num_embedding[5])
        self.embedd7 = nn.Embedding(attributes[6], num_embedding[6])
        self.embedd8 = nn.Embedding(attributes[7], num_embedding[7])
        self.embedd9 = nn.Embedding(attributes[8], num_embedding[8])
        self.embedd10 = nn.Embedding(attributes[9], num_embedding[9])
        self.embedd11 = nn.Embedding(attributes[10], num_embedding[10])
        self.embedd12 = nn.Embedding(attributes[11], num_embedding[11])
        self.embedd13 = nn.Embedding(attributes[12], num_embedding[12])
        self.embedd14 = nn.Embedding(attributes[13], num_embedding[13])
        self.embedd15 = nn.Embedding(attributes[14], num_embedding[14])

        self.fc1 = nn.Linear(sum(num_embedding) + 1, sum(num_embedding) + 1 + 1, bias=False)

    def forward(self, x):
        # Apply the embeddings to the features and concat. them
        # embedded = torch.cat([embedding(x[:, i+1]).reshape(-1,embedding.embedding_dim) for i, embedding in enumerate(self.embeddings)], dim=1)
        
        x0  = x[:,0].reshape(-1, 1) / 95
        x1  = self.embedd1(x[:,1].int())
        x2  = self.embedd2(x[:,2].int())
        x3  = self.embedd3(x[:,3].int())
        x4  = self.embedd4(x[:,4].int())
        x5  = self.embedd5(x[:,5].int())
        x6  = self.embedd6(x[:,6].int())
        x7  = self.embedd7(x[:,7].int())
        x8  = self.embedd8(x[:,8].int())
        x9  = self.embedd9(x[:,9].int())
        x10 = self.embedd10(x[:,10].int())
        x11 = self.embedd11(x[:,11].int())
        x12 = self.embedd12(x[:,12].int())
        x13 = self.embedd13(x[:,13].int())
        x14 = self.embedd14(x[:,14].int())
        x15 = self.embedd15(x[:,15].int())

        # Add age attribute to the embeddings
        # embedded = torch.cat((x[:, 0].reshape(-1,1), embedded), dim=1)
        embedded = torch.cat((x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15), dim=1)

        out = self.fc1(embedded)

        out = out / (torch.norm(out, dim=1, keepdim=True) + 1e-16)

        return out

class FolkEmbeddingXYS(nn.Module):
    def __init__(self):
        super(FolkEmbeddingXYS, self).__init__()

        #'Emp' :: [SCHL, MAR, RELP, DIS, ESP, CIT, MIG, MIL, ANC, Nativity, DEAR, DEYE, DREM, SEX, RAC1P]
        attributes    = [25, 6, 18, 3, 9, 6, 4, 5, 5, 3, 3, 3, 3, 3, 10, 2]
        num_embedding = [10, 3,  9, 3, 5, 3, 2, 3, 3, 2, 2, 2, 2, 2, 5, 1]
        
        # self.embeddings = [nn.Embedding(num_classes, emb_dim, device='cuda') for (num_classes, emb_dim) in zip(attributes, num_embedding)]
        self.embedd1 = nn.Embedding(attributes[0], num_embedding[0])
        self.embedd2 = nn.Embedding(attributes[1], num_embedding[1])
        self.embedd3 = nn.Embedding(attributes[2], num_embedding[2])
        self.embedd4 = nn.Embedding(attributes[3], num_embedding[3])
        self.embedd5 = nn.Embedding(attributes[4], num_embedding[4])
        self.embedd6 = nn.Embedding(attributes[5], num_embedding[5])
        self.embedd7 = nn.Embedding(attributes[6], num_embedding[6])
        self.embedd8 = nn.Embedding(attributes[7], num_embedding[7])
        self.embedd9 = nn.Embedding(attributes[8], num_embedding[8])
        self.embedd10 = nn.Embedding(attributes[9], num_embedding[9])
        self.embedd11 = nn.Embedding(attributes[10], num_embedding[10])
        self.embedd12 = nn.Embedding(attributes[11], num_embedding[11])
        self.embedd13 = nn.Embedding(attributes[12], num_embedding[12])
        self.embedd14 = nn.Embedding(attributes[13], num_embedding[13])
        self.embedd15 = nn.Embedding(attributes[14], num_embedding[14])
        self.embedd16 = nn.Embedding(attributes[15], num_embedding[15])

        self.fc1 = nn.Linear(sum(num_embedding) + 1, 64)

    def forward(self, x):
        # Apply the embeddings to the features and concat. them
        # embedded = torch.cat([embedding(x[:, i+1]).reshape(-1,embedding.embedding_dim) for i, embedding in enumerate(self.embeddings)], dim=1)
        
        x0  = x[:,0].reshape(-1, 1)
        x1  = self.embedd1(x[:,1].int())
        x2  = self.embedd2(x[:,2].int())
        x3  = self.embedd3(x[:,3].int())
        x4  = self.embedd4(x[:,4].int())
        x5  = self.embedd5(x[:,5].int())
        x6  = self.embedd6(x[:,6].int())
        x7  = self.embedd7(x[:,7].int())
        x8  = self.embedd8(x[:,8].int())
        x9  = self.embedd9(x[:,9].int())
        x10 = self.embedd10(x[:,10].int())
        x11 = self.embedd11(x[:,11].int())
        x12 = self.embedd12(x[:,12].int())
        x13 = self.embedd13(x[:,13].int())
        x14 = self.embedd14(x[:,14].int())
        x15 = self.embedd15(x[:,15].int())
        x16 = self.embedd16(x[:,16].int())

        # Add age attribute to the embeddings
        # embedded = torch.cat((x[:, 0].reshape(-1,1), embedded), dim=1)
        embedded = torch.cat((x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16), dim=1).float()

        # out = self.fc1(embedded)

        return embedded   
        # return out   


class FolkEmbeddingXY(nn.Module):
    def __init__(self):
        super(FolkEmbeddingXY, self).__init__()

        #'Emp' :: [SCHL, MAR, RELP, DIS, ESP, CIT, MIG, MIL, ANC, Nativity, DEAR, DEYE, DREM, SEX, RAC1P]
        attributes    = [25, 6, 18, 3, 9, 6, 4, 5, 5, 3, 3, 3, 3, 3, 10, 2]
        num_embedding = [10, 3,  9, 3, 5, 3, 2, 3, 3, 2, 2, 2, 2, 2, 5, 1]
        
        # self.embeddings = [nn.Embedding(num_classes, emb_dim, device='cuda') for (num_classes, emb_dim) in zip(attributes, num_embedding)]
        self.embedd1 = nn.Embedding(attributes[0], num_embedding[0])
        self.embedd2 = nn.Embedding(attributes[1], num_embedding[1])
        self.embedd3 = nn.Embedding(attributes[2], num_embedding[2])
        self.embedd4 = nn.Embedding(attributes[3], num_embedding[3])
        self.embedd5 = nn.Embedding(attributes[4], num_embedding[4])
        self.embedd6 = nn.Embedding(attributes[5], num_embedding[5])
        self.embedd7 = nn.Embedding(attributes[6], num_embedding[6])
        self.embedd8 = nn.Embedding(attributes[7], num_embedding[7])
        self.embedd9 = nn.Embedding(attributes[8], num_embedding[8])
        self.embedd10 = nn.Embedding(attributes[9], num_embedding[9])
        self.embedd11 = nn.Embedding(attributes[10], num_embedding[10])
        self.embedd12 = nn.Embedding(attributes[11], num_embedding[11])
        self.embedd13 = nn.Embedding(attributes[12], num_embedding[12])
        self.embedd14 = nn.Embedding(attributes[13], num_embedding[13])
        self.embedd15 = nn.Embedding(attributes[14], num_embedding[14])
        self.embedd16 = nn.Embedding(attributes[15], num_embedding[15])

        # self.fc1 = nn.Linear(sum(num_embedding) + 1, 64)

    def forward(self, x):
        # Apply the embeddings to the features and concat. them
        # embedded = torch.cat([embedding(x[:, i+1]).reshape(-1,embedding.embedding_dim) for i, embedding in enumerate(self.embeddings)], dim=1)
        
        # x0  = x[:,0].reshape(-1, 1)
        x1  = self.embedd1(x[:,1].int())
        x2  = self.embedd2(x[:,2].int())
        x3  = self.embedd3(x[:,3].int())
        x4  = self.embedd4(x[:,4].int())
        x5  = self.embedd5(x[:,5].int())
        x6  = self.embedd6(x[:,6].int())
        x7  = self.embedd7(x[:,7].int())
        x8  = self.embedd8(x[:,8].int())
        x9  = self.embedd9(x[:,9].int())
        x10 = self.embedd10(x[:,10].int())
        x11 = self.embedd11(x[:,11].int())
        x12 = self.embedd12(x[:,12].int())
        x13 = self.embedd13(x[:,13].int())
        x14 = self.embedd14(x[:,14].int())
        x15 = self.embedd15(x[:,15].int())
        x16 = self.embedd16(x[:,16].int())

        # Add age attribute to the embeddings
        # embedded = torch.cat((x[:, 0].reshape(-1,1), embedded), dim=1)
        embedded = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16), dim=1).float()

        # out = self.fc1(embedded)

        return embedded   
        # return out   

class FolkEmbeddingYS(nn.Module):
    def __init__(self):
        super(FolkEmbeddingYS, self).__init__()

        #'Emp' :: [SCHL, MAR, RELP, DIS, ESP, CIT, MIG, MIL, ANC, Nativity, DEAR, DEYE, DREM, SEX, RAC1P]
        attributes    = [25, 6, 18, 3, 9, 6, 4, 5, 5, 3, 3, 3, 3, 3, 10, 2]
        # num_embedding = [10, 3,  9, 3, 5, 3, 2, 3, 3, 2, 2, 2, 2, 2, 5, 1]
        num_embedding = [10, 3,  9, 3, 5, 3, 2, 3, 3, 2, 2, 2, 2, 2, 5, 1]
        
        # self.embeddings = [nn.Embedding(num_classes, emb_dim, device='cuda') for (num_classes, emb_dim) in zip(attributes, num_embedding)]
        # self.embedd1 = nn.Embedding(attributes[0], num_embedding[0])
        # self.embedd2 = nn.Embedding(attributes[1], num_embedding[1])
        # self.embedd3 = nn.Embedding(attributes[2], num_embedding[2])
        # self.embedd4 = nn.Embedding(attributes[3], num_embedding[3])
        # self.embedd5 = nn.Embedding(attributes[4], num_embedding[4])
        # self.embedd6 = nn.Embedding(attributes[5], num_embedding[5])
        # self.embedd7 = nn.Embedding(attributes[6], num_embedding[6])
        # self.embedd8 = nn.Embedding(attributes[7], num_embedding[7])
        # self.embedd9 = nn.Embedding(attributes[8], num_embedding[8])
        # self.embedd10 = nn.Embedding(attributes[9], num_embedding[9])
        # self.embedd11 = nn.Embedding(attributes[10], num_embedding[10])
        # self.embedd12 = nn.Embedding(attributes[11], num_embedding[11])
        # self.embedd13 = nn.Embedding(attributes[12], num_embedding[12])
        # self.embedd14 = nn.Embedding(attributes[13], num_embedding[13])
        # self.embedd15 = nn.Embedding(attributes[14], num_embedding[14])
        self.embedd16 = nn.Embedding(attributes[15], num_embedding[15])

        # self.fc1 = nn.Linear(sum(num_embedding) + 1, 64)
        self.fc1 = nn.Linear(2, 64)

    def forward(self, x):
        # Apply the embeddings to the features and concat. them
        # embedded = torch.cat([embedding(x[:, i+1]).reshape(-1,embedding.embedding_dim) for i, embedding in enumerate(self.embeddings)], dim=1)
        
        x0  = x[:,0].reshape(-1, 1)
        # x1  = self.embedd1(x[:,1].int())
        # x2  = self.embedd2(x[:,2].int())
        # x3  = self.embedd3(x[:,3].int())
        # x4  = self.embedd4(x[:,4].int())
        # x5  = self.embedd5(x[:,5].int())
        # x6  = self.embedd6(x[:,6].int())
        # x7  = self.embedd7(x[:,7].int())
        # x8  = self.embedd8(x[:,8].int())
        # x9  = self.embedd9(x[:,9].int())
        # x10 = self.embedd10(x[:,10].int())
        # x11 = self.embedd11(x[:,11].int())
        # x12 = self.embedd12(x[:,12].int())
        # x13 = self.embedd13(x[:,13].int())
        # x14 = self.embedd14(x[:,14].int())
        # x15 = self.embedd15(x[:,15].int())
        x16 = self.embedd16(x[:,1].int())

        # Add age attribute to the embeddings
        # embedded = torch.cat((x[:, 0].reshape(-1,1), embedded), dim=1)
        embedded = torch.cat((x0, x16), dim=1).float()

        out = self.fc1(embedded)

        # return embedded   
        return out   