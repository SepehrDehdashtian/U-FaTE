# non_parametric.py

import torch
import torch.nn as nn
import torchmetrics.metric as tm
from typing import Any, Callable, Optional

import hal.kernels as kernels
import hal.metrics.dependence.utils as utils
import hal.utils.misc as misc
import math
# import utils
# from rff import get_RFF

__all__ = ['NonParametricDependence']

class detachabledict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def detach(self):
        for k, v in self.items():
            self[k] = v.detach()

class NonParametricDependence(tm.Metric):
    def __init__(self,
        rff: bool,
        score_list: dict,
        kernel_z: str,
        kernel_s: str,
        kernel_z_opts: dict,
        kernel_s_opts: dict,
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
        rff = bool(rff)
        self.add_state("zz", default=[], dist_reduce_fx=None)
        self.add_state("ss", default=[], dist_reduce_fx=None)

        self.kernel_z = getattr(kernels, kernel_z)(**kernel_z_opts)
        self.kernel_s = getattr(kernels, kernel_s)(**kernel_s_opts)
        self.rff = rff
        self.fn_dict = dict()

        for fn, args in score_list.items():
            self.fn_dict[fn] = getattr(utils, fn)(**args)

    def update(self, z, s):
        self.zz.append(z)
        self.ss.append(s)

    def compute(self):
        z = torch.cat(self.zz, dim=0).type(torch.float)
        s = torch.cat(self.ss, dim=0).type(torch.float)

        if z.dim() == 1:
            z = z.unsqueeze(1)
        if s.dim() == 1:
            s = s.unsqueeze(1)
        
        ######################### Z-normalization #########################
        # z_mean = torch.mean(z, dim=0)
        # z_std = torch.std(z, dim=0)
        # z_m = (z - z_mean) / (z_std + 1e-16)
        #
        # s_mean = torch.mean(s, dim=0)
        # s_std = torch.std(s, dim=0)
        # s_m = (s - s_mean) / (s_std + 1e-16)

        ######################### Norm-normalization #######################

        # z_m = z / (z.norm() / math.sqrt(z.shape[0]))
        # s_m = s / (s.norm() / math.sqrt(s.shape[0]))

        ######################### No Normalization ##########################
        s_m = s
        z_m = z

        #####################################################################3
        score_dict = detachabledict()

        if self.rff:
            rng_state = torch.get_rng_state()
            torch.manual_seed(100)
            phi_z = self.kernel_z(z_m)
            phi_z = misc.mean_center(phi_z, dim=0)

            phi_s = self.kernel_s(s_m)
            phi_s = misc.mean_center(phi_s, dim=0)
            torch.set_rng_state(rng_state)

            for name, fn in self.fn_dict.items():
                score_dict[name] = fn(phi_z, phi_s, self.rff)

        else:
            k_z = self.kernel_z(z_m)
            k_z = misc.mean_center(k_z, dim=0)
            k_z = misc.mean_center(k_z.t(), dim=0)

            k_s = self.kernel_s(s_m)
            k_s = misc.mean_center(k_s, dim=0)
            k_s = misc.mean_center(k_s.t(), dim=0)

            for name, fn in self.fn_dict.items():
                score_dict[name] = fn(k_z, k_s, self.rff)

        return score_dict
