# gaussian.py

from torch import nn

__all__ = ['EncLinear', 'EncGaussian', 'TgtGaussian', 'AdvGaussian', 'RepGaussianHGR', 'AdvGaussianHGR']


class EncLinear(nn.Module):

    def __init__(self, nout, r):
        super().__init__()

        # ndim: in features
        # r: out features

        self.linear = nn.Linear(r, nout, bias=True)

    def forward(self, x, y=None):
        z = self.linear(x)
        return z

class EncGaussian(nn.Module):
    
    def __init__(self, ndim, r, hdl):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(ndim, 2*hdl),
            nn.PReLU(),
            nn.Linear(2*hdl, 2*hdl),
            nn.PReLU(),
            nn.Linear(2*hdl, hdl),
            nn.PReLU(),
            nn.Linear(hdl, r, bias=False)
        )

    def forward(self, x, y=None):
        z = self.encoder(x)
        ###### Normalization #####
        z = z / (torch.norm(z, dim=1, keepdim=True) + 1e-16)
        return z


class TgtGaussian(nn.Module):
    
    def __init__(self, nout, r, hdl):
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(r, 2*hdl),
            nn.PReLU(),
            nn.Linear(2*hdl, hdl),
            nn.PReLU(),
            nn.Linear(hdl, nout),
        )

    def forward(self, z, y=None):
        out = self.target(z)
        return out

class AdvGaussian(nn.Module):

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

class RepGaussianHGR(nn.Module):

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


class AdvGaussianHGR(nn.Module):

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