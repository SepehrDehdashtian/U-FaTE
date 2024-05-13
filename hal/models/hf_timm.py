# resnet.py

import timm
import torch
import torch.nn as nn
import os

__all__ = ['TIMM']

class TIMM(nn.Module):
    def __init__(self, device, model_name, pretrained=True, *args, **kwargs):
        super().__init__()
        self.device = device

        model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model = model.eval()

        data_config = timm.data.resolve_model_data_config(model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

    @torch.no_grad()
    def forward(self, image):
        
        self.model.eval()

        with torch.no_grad(), torch.cuda.amp.autocast():
            # features = self.model(self.transforms(image))
            features = self.model(image)
            return features
