"""
UtilityFromMSE is a wrapper around MSE function. MSE function gives the
error and we want to convert it to a quantity of utility.

In regression tasks, the worst prediction by a model is the expected
value of the target. Utility of a model predicting y_hat is given by,

U = 1 - MSE(y_hat, y)/MSE(E[Y], y)

For now, this file is kept under fairness. Later on, this would be moved
to another location.
"""

import torch
import torch.nn.functional as F
import torchmetrics as tm
from typing import Any, Callable, Optional

__all__ = ['UtilityFromMSE']

class UtilityFromMSE(tm.Metric):
    def __init__(self,
                 mean,
                 compute_on_step: bool = False,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None,
                 ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        if isinstance(mean, list):
            mean = torch.Tensor(mean).reshape(1, -1)
        self.mean = mean
        self.add_state("xx", default=[], dist_reduce_fx=None) # estimate
        self.add_state("yy", default=[], dist_reduce_fx=None) # GT

    def update(self, xx, yy):
        self.xx.append(xx)
        self.yy.append(yy)

    def compute(self):
        xx = torch.cat(self.xx, 0)
        yy = torch.cat(self.yy, 0)
        mean_vector = torch.tile(self.mean, (yy.size(0), 1))
        mean_vector = mean_vector.to(xx.device)

        utility = 1. - F.mse_loss(xx, yy)/F.mse_loss(mean_vector, yy)

        return torch.maximum(torch.zeros_like(utility), utility)
