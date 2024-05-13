# rdc.py

import torchmetrics.metric as tm
from typing import Any, Callable, Optional

__all__ = ['DepRDC']

class DepRDC(tm.Metric):
    def __init__(self,
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
    
    def update(self, x, y):
        pass
    
    def compute(self):
        pass