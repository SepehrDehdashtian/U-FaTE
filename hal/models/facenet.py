"""
Written by Gautam Sreekumar.

This file contains both the nn.Module and the pl.LightningModule
for FaceNet model. FaceNet is what we call ArcFace, CosFace, SphereFace,
etc. Most of them share a common structure. Only the weights will
change. So FaceNet can be used to extract features or train a classifier
on existing features or do both end-to-end. FaceNetClassifier is a
PL-wrapper around FaceNet. Sine PL needs a loss function, it needs to
have a classifier. The main purpose for additional wrapped module is
compatibility with pl.LightningDatamodules.

NOTE: The term "FaceNet" is not to be confused with the FaceNet paper.
FaceNet paper is "FaceNet: A Unified Embedding for Face Recognition and
Clustering" (https://arxiv.org/abs/1503.03832, CVPR 2015).
Just like its authors, I am not creative enough to come up with a fancy
name for a neural network that gets features from faces.
"""

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

__all__ = ['FaceNet', 'FaceNetClassifier']

from .backbones import get_model

# PyTorch Module

class FaceNet(nn.Module):
    def __init__(self, name=None, backbone_weight=None, num_classes=None):
        super().__init__()

        if name is not None:
            self.backbone = get_model(name)
            self.init_backbone_weights()
            if backbone_weight is not None:
                self.backbone.load_state_dict(torch.load(backbone_weight))
        else:
            self.backbone = None

        if num_classes is not None:
            self.classifier = nn.Sequential(nn.Linear(512, 256),
                                            nn.BatchNorm1d(256),
                                            nn.PReLU(),
                                            nn.Linear(256, num_classes))
            
            # self.classifier = nn.Linear(512, num_classes)
            self.init_classifier_weights()
        else:
            self.classifier = None
    
    def init_classifier_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def init_backbone_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.backbone is not None:
            x = self.backbone(x)

        if self.classifier is not None:
            x = self.classifier(x)
        
        return x

# PyTorch Lightning Module

class FaceNetClassifier(pl.LightningModule):
    def __init__(self, num_classes, net=None,
                backbone_weight=None,
                lr=1.0, gamma=0.97):
        # 0.97^100 = 0.04755
        # 0.97^200 = 0.00226
        super().__init__()
        self.save_hyperparameters(ignore="model")
        self.model = FaceNet(net, backbone_weight, num_classes)
        self.pretrained = (backbone_weight is not None)

        self.lr = lr
        self.gamma = gamma

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, tgt, sens = batch
        elif len(batch) == 4:
            x, tgt, sens, img_loc = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, tgt.long())
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def test_acc(self, y, logits):
        logits = torch.argmax(logits, 1)
        return torch.sum((logits == y).type(torch.float))
    
    def on_validation_epoch_start(self) -> None:
        self.val_acc = 0.
        self.val_tot = 0.

    def validation_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, tgt, sens = batch
        elif len(batch) == 4:
            x, tgt, sens, img_loc = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, tgt.long())
        self.val_acc += self.test_acc(tgt, logits)
        self.val_tot += torch.numel(tgt)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_acc/self.val_tot, prog_bar=True, on_step=False, on_epoch=True)
    
    def validation_epoch_end(self, outputs):
        # self.val_acc /= self.val_tot
        # self.log("val_acc", self.val_acc, prog_bar=True)
        pass

    def configure_optimizers(self):
        params_1 = [_ for _ in self.model.backbone.parameters()]
        params_2 = [_ for _ in self.model.classifier.parameters()]
        if self.pretrained:
            param_group_1 = {"params": params_1,
                             "lr": self.lr*0.1}
        else:
            param_group_1 = {"params": params_1,
                             "lr": self.lr}
        param_group_2 = {"params": params_2,
                         "lr": self.lr}
        optimizer = torch.optim.AdamW([param_group_1, param_group_2],
                                      lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.gamma)

        out_dict = dict()
        out_dict["optimizer"] = optimizer
        out_dict["lr_scheduler"] = {"scheduler": scheduler,
                                    "interval": "epoch",
                                    "frequency": 1}

        return out_dict
