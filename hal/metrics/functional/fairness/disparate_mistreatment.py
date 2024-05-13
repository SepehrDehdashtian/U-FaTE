# disparate_mistreatment.py

import torch
from hal.utils.misc import DetachableDict

__all__  = ['disparate_mistreatment']

def disparate_mistreatment(preds, target, control):

    n_attributes_c = control.shape[1]
    num_00 = torch.zeros(n_attributes_c)
    num_01 = torch.zeros(n_attributes_c)
    num_10 = torch.zeros(n_attributes_c)
    num_11 = torch.zeros(n_attributes_c)

    den_00 = torch.zeros(n_attributes_c)
    den_01 = torch.zeros(n_attributes_c)
    den_10 = torch.zeros(n_attributes_c)
    den_11 = torch.zeros(n_attributes_c)
    
    pred = preds.max(1)[1]

    output = DetachableDict()
    for cntl in range(n_attributes_c):        
        num_00[cntl] += sum((pred == 1) & (control[:, cntl] == 0) & (target == 0))
        num_01[cntl] += sum((pred == 1) & (control[:, cntl] == 1) & (target == 0))
        num_10[cntl] += sum((pred == 0) & (control[:, cntl] == 0) & (target == 1))
        num_11[cntl] += sum((pred == 0) & (control[:, cntl] == 1) & (target == 1))

        den_00[cntl] += sum((control[:, cntl] == 0) & (target == 0))
        den_01[cntl] += sum((control[:, cntl] == 1) & (target == 0))
        den_10[cntl] += sum((control[:, cntl] == 0) & (target == 1))
        den_11[cntl] += sum((control[:, cntl] == 1) & (target == 1))

        dm_fpr = num_00[cntl] / den_00[cntl] - num_01[cntl] / den_01[cntl]
        dm_fnr = num_10[cntl] / den_10[cntl] - num_11[cntl] / den_11[cntl]
        output['FPR_c_' + str(cntl)] = dm_fpr
        output['FNR_c_' + str(cntl)] = dm_fnr
    
    return output