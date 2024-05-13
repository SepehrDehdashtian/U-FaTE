# equal_opportunity.py

import torch
from hal.utils.misc import DetachableDict

__all__  = ['equal_opportunity']

def equal_opportunity(preds, target, control):
    
    n_classes_y = preds.shape[1]
    n_attributes_c = control.shape[1]

    output = DetachableDict()
    for cntl in range(n_attributes_c):
        n_classes_c = len(control[:, cntl].unique())

        total = torch.zeros((n_classes_y, n_classes_c, n_classes_y))
        count = torch.zeros((n_classes_y, n_classes_c, n_classes_y))
        
        pred = preds.max(1)[1]

        for y_temp in range(n_classes_y):
            for c_temp in range(n_classes_c):
                for y_temp_hat in range(n_classes_y):
                    total[y_temp_hat, y_temp, c_temp] += sum((control[:,cntl] == c_temp) & (target == y_temp))
                    count[y_temp_hat, y_temp, c_temp] += sum((pred == y_temp_hat) & (control[:,cntl] == c_temp) & (target == y_temp))

        prob = count / torch.clamp(total, 1.0)
        
        # calculate the difference between all pairs of probabilities
        diff = []
        for y_temp_hat in range(n_classes_y):
            for y_temp in range(n_classes_y):
                for c1 in range(n_classes_c):
                    for c2 in range(c1 + 1, n_classes_c):
                        diff.append(abs(prob[y_temp_hat, y_temp, c1] - prob[y_temp_hat, y_temp, c2]))
            
        output['EOO_c_' + str(cntl)] = torch.Tensor(diff).max()
    
    return output