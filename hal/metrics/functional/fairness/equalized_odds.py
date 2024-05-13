# equalized_odd.py

import torch
from hal.utils.misc import DetachableDict

_all__ = ['equalized_odds']

def equalized_odds(preds, target, control):
    """
    Compute the equalized odds metric.
    """    
    assert preds.shape[1] == 2
    pred = preds.max(1)[1]
    n_attributes_c = control.shape[1]

    output = DetachableDict()
    for cntl in range(n_attributes_c):    
        n_classes_c = len(control[:, cntl].unique())

        total = torch.zeros(n_classes_c)
        count = torch.zeros(n_classes_c)

        for c_temp in range(n_classes_c):
            total[c_temp] += sum((control[:, cntl] == c_temp) & (target == 1))
            count[c_temp] += sum((pred == 1) & (control[:, cntl] == c_temp) & (target == 1))

        prob = count / torch.clamp(total, 1.0)

        # calculate the difference between all pairs of probabilities
        diff = []
        for c1 in range(n_classes_c):
            for c2 in range(c1 + 1, n_classes_c):
                diff.append(abs(prob[c1] - prob[c2]))

        output['EO_c_' + str(cntl)] = torch.Tensor(diff).max()

    return output