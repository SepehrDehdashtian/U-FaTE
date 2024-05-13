# demographic_auroc.py

import torch
import hal.metrics as metrics
from hal.utils.misc import DetachableDict

__all__ = ['demographic_auroc']

def demographic_auroc(preds, target, control):

    target = torch.Tensor(target)
    control = torch.Tensor(control)
    preds = torch.Tensor(preds)

    output = DetachableDict()
    for cntl in range(control.shape[1]):
        c_values = len(control[:, cntl].unique())
        for i in range(c_values):
            target = target[control[:, cntl] == i]
            preds = preds[control[:, cntl] == i]
            output['control_' + str(cntl) + '_c_' + str(i)] = metrics.functional.auroc(preds, target)

    return output