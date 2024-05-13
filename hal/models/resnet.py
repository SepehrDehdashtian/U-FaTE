# resnet.py

import timm
import torch
import torch.nn as nn
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor 

__all__ = ['ResNet18', 'ResNet18_mod', 'ResNet18_mod2', 'ResNet18_mod_XYS', 'ResNet18_mod_XYS_2', 'ResNet18_mod_XYS_3', 'ResNet34', 'ResNet50']

# num_classes=0 is to the features without the last layer
# The features will be of dimension (batch_size, 2048)

class ResNet18(nn.Module):
    def __init__(self, dim=256, pretrained=False, normalize_output=False):
        super().__init__()
        self.model = timm.create_model('resnet18', pretrained=pretrained,
                                        num_classes=0)
        self.fc    = nn.Linear(512, dim)
        self.normalize_output = normalize_output


    def forward(self, x, y=None):
        x   = self.model(x)
        out = self.fc(x)

        if self.normalize_output:
            out = out / (torch.norm(out, dim=1, keepdim=True) + 1e-16)
        return out

class ResNet18_mod(nn.Module):
    def __init__(self, dim=256, pretrained=False, normalize_output=False):
        super().__init__()
        model = timm.create_model('resnet18', pretrained=pretrained,
                                        num_classes=0)
        features = {'layer2.1.act2': 'out'}
        self.model = create_feature_extractor(model, return_nodes=features)
        self.avgpool = nn.AvgPool2d(3, stride=3)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc    = nn.Linear(2048, dim)
        self.normalize_output = normalize_output

    def forward(self, x, y=None):
        x   = self.model(x)
        x   = self.flatten(self.avgpool(x['out']))
        out = self.fc(x)

        if self.normalize_output:
            out = out / (torch.norm(out, dim=1, keepdim=True) + 1e-16)

        return out
    

class ResNet18_mod2(nn.Module):
    def __init__(self, dim=256, pretrained=False, normalize_output=False):
        super().__init__()
        model = timm.create_model('resnet18', pretrained=pretrained,
                                        num_classes=0)
        features = {'layer2.1.act2': 'out'}
        self.model = create_feature_extractor(model, return_nodes=features)
        self.avgpool = nn.AvgPool2d(3, stride=3)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc    = nn.Linear(10368, dim)
        self.normalize_output = normalize_output

    def forward(self, x, y=None):
        x   = self.model(x)
        x   = self.flatten(self.avgpool(x['out']))
        out = self.fc(x)

        if self.normalize_output:
            out = out / (torch.norm(out, dim=1, keepdim=True) + 1e-16)

        return out
    
class ResNet18_mod_XYS(nn.Module):
    def __init__(self, dim=256, pretrained=False, normalize_output=False):
        super().__init__()
        model = timm.create_model('resnet18', pretrained=pretrained,
                                        num_classes=0)
        features = {'layer2.1.act2': 'out'}
        self.model = create_feature_extractor(model, return_nodes=features)
        self.avgpool = nn.AvgPool2d(3, stride=3)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        # self.fc    = nn.Linear(2048, dim)
        self.fc    = nn.Linear(2050, dim)
        self.normalize_output = normalize_output

    def forward(self, x, y=None):
        img =  x[:, :3]
        y   =  x[:, 3].mean(axis=1).mean(axis=1)
        s   =  x[:, 4].mean(axis=1).mean(axis=1)

        z   = self.model(img)
        z   = self.flatten(self.avgpool(z['out']))
        # import pdb; pdb.set_trace()
        z   = torch.cat((z, y.unsqueeze(-1), s.unsqueeze(-1)), dim=1)
        out = self.fc(z)

        if self.normalize_output:
            out = out / (torch.norm(out, dim=1, keepdim=True) + 1e-16)

        return out

    
class ResNet18_mod_XYS_2(nn.Module):
    def __init__(self, dim=256, pretrained=False, normalize_output=False):
        super().__init__()
        model = timm.create_model('resnet18', pretrained=pretrained,
                                        num_classes=0)
        features = {'layer2.1.act2': 'out'}
        self.model = create_feature_extractor(model, return_nodes=features)
        self.avgpool = nn.AvgPool2d(3, stride=3)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc    = nn.Linear(2048, dim)
        self.normalize_output = normalize_output

    def forward(self, x, y=None):
        img =  x[:, :3]
        y   =  x[:, 3].mean(axis=1).mean(axis=1)
        s   =  x[:, 4].mean(axis=1).mean(axis=1)

        z   = self.model(img)
        z   = self.flatten(self.avgpool(z['out']))
        # import pdb; pdb.set_trace()
        # z   = torch.cat((z, y.unsqueeze(-1), s.unsqueeze(-1)), dim=1)
        out = self.fc(z)

        if self.normalize_output:
            out   = out / (torch.norm(out, dim=1, keepdim=True) + 1e-16)
            out   = torch.cat((out, y.unsqueeze(-1), s.unsqueeze(-1)), dim=1)
        return out
    
class ResNet18_mod_XYS_3(nn.Module):
    def __init__(self, dim=256, pretrained=False, normalize_output=False):
        super().__init__()
        model = timm.create_model('resnet18', pretrained=pretrained,
                                        num_classes=0)
        features = {'layer2.1.act2': 'out'}
        self.model = create_feature_extractor(model, return_nodes=features)
        self.avgpool = nn.AvgPool2d(3, stride=3)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc    = nn.Linear(10368, dim)
        self.normalize_output = normalize_output

    def forward(self, x, y=None):
        img =  x[:, :3]
        y   =  x[:, 3].mean(axis=1).mean(axis=1)
        s   =  x[:, 4].mean(axis=1).mean(axis=1)

        z   = self.model(img)
        z   = self.flatten(self.avgpool(z['out']))
        # import pdb; pdb.set_trace()
        # z   = torch.cat((z, y.unsqueeze(-1), s.unsqueeze(-1)), dim=1)
        out = self.fc(z)

        if self.normalize_output:
            out   = out / (torch.norm(out, dim=1, keepdim=True) + 1e-16)
            out   = torch.cat((out, y.unsqueeze(-1), s.unsqueeze(-1)), dim=1)
        return out

class ResNet34(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.model = timm.create_model('resnet34', pretrained=pretrained,
                                        num_classes=0)

    def forward(self, x, y=None):
        return self.model(x)

class ResNet50(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.model = timm.create_model('resnet50', pretrained=pretrained,
                                        num_classes=0)

    def forward(self, x, y=None):
        return self.model(x)
