# demographic_parity.py

import torch
from hal.utils.misc import DetachableDict

_all_ = ['demographic_parity']

def demographic_parity(preds, control):

    n_attributes_c = control.shape[1]
    n_classes_y = preds.shape[1]

    output = DetachableDict()
    for cntl in range(n_attributes_c):
        n_classes_c = len(control[:, cntl].unique())

        total = torch.zeros((n_classes_y, n_classes_c))
        count = torch.zeros((n_classes_y, n_classes_c))

        pred = preds.max(1)[1]

        for y_temp in range(n_classes_y):
            for c_temp in range(n_classes_c):
                total[y_temp, c_temp] = sum(control[:, cntl] == c_temp)
                count[y_temp, c_temp] = sum((pred == y_temp) & (control[:, cntl] == c_temp))

        prob = count / torch.clamp(total, 1.0)
        
        # calculate the difference between all pairs of probabilities
        diff = []
        for y_temp in range(n_classes_y):
            for c1 in range(n_classes_c):
                for c2 in range(c1 + 1, n_classes_c):
                    diff.append(abs(prob[y_temp, c1] - prob[y_temp, c2]))

        output["DP_c_" + str(cntl)] = torch.Tensor(diff).max()

    return output