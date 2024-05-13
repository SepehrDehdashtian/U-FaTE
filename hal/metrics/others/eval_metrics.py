
import torch
import torchmetrics as tm
from sklearn.metrics import roc_auc_score

from typing import Any, Callable, Optional

__all__ = ['ROC_AUC_Score', 'RecallScore', 'PrecisionScore', 'F1Score', 
            'TruePositive', 'FalsePositive', 
            'TrueNegative', 'FalseNegative',
            'Confusion_Matrix', 'ROC_AUC_Score_sklearn']


class ROC_AUC_Score_sklearn(tm.Metric):
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

        self.add_state("y_pred", default=[], dist_reduce_fx=None) # estimate
        self.add_state("y", default=[], dist_reduce_fx=None) # GT

    def update(self, y_hat, y):
        self.y.append(y)
        self.y_pred.append(y_hat)
        
    def compute(self):
        y = torch.cat(self.y, 0).int()
        y_pred = torch.cat(self.y_pred, 0)
        y_pred = torch.argmax(y_pred, dim=1)

        roc = roc_auc_score(y.cpu().numpy(), y_pred.detach().cpu().numpy())

        return roc

        
class ROC_AUC_Score(tm.Metric):
    def __init__(self,
                 num_classes,
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

        self.add_state("y_pred", default=[], dist_reduce_fx=None) # estimate
        self.add_state("y", default=[], dist_reduce_fx=None) # GT

        self.ROC_AUC = tm.AUROC(num_classes=num_classes, pos_label=1, average='macro')

    def update(self, y_hat, y):
        y_pred = torch.sigmoid(y_hat)
        
        self.y.append(y)
        self.y_pred.append(y_pred)
        
    def compute(self):
        y = torch.cat(self.y, 0).int()
        y_pred = torch.cat(self.y_pred, 0)

        roc = self.ROC_AUC(y_pred, y)
        
        self._reset()

        return roc

    def _reset(self):
        self.ROC_AUC = tm.AUROC(num_classes=2, pos_label=1, average='macro')


class RecallScore(tm.Metric):
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

        self.FN = torch.Tensor([0]) # False-Negative
        self.TP = torch.Tensor([0]) # True-Positive

    def update(self, y_hat, y):
        y_hat_categorical = y_hat.argmax(1)
        
        confusion_vector = y_hat_categorical / y
        # Element-wise division of the 2 tensors returns a new tensor which holds a
        # unique value for each case:
        #   1     where prediction and truth are 1 (True Positive)
        #   inf   where prediction is 1 and truth is 0 (False Positive)
        #   nan   where prediction and truth are 0 (True Negative)
        #   0     where prediction is 0 and truth is 1 (False Negative)
        
        TP = torch.sum(confusion_vector == 1).item()
        FN = torch.sum(confusion_vector == 0).item()
        
        self.FN += FN
        self.TP += TP

    def compute(self):
        recall = self.TP / (self.TP + self.FN)
        
        self._reset()

        return recall

    def _reset(self):
        self.FN = torch.Tensor([0]) # False-Negative
        self.TP = torch.Tensor([0]) # True-Positive


class PrecisionScore(tm.Metric):
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

        self.FP = torch.Tensor([0]) # False-Positive
        self.TP = torch.Tensor([0]) # True-Positive

    def update(self, y_hat, y):
        y_hat_categorical = y_hat.argmax(1)
        
        confusion_vector = y_hat_categorical / y
        # Element-wise division of the 2 tensors returns a new tensor which holds a
        # unique value for each case:
        #   1     where prediction and truth are 1 (True Positive)
        #   inf   where prediction is 1 and truth is 0 (False Positive)
        #   nan   where prediction and truth are 0 (True Negative)
        #   0     where prediction is 0 and truth is 1 (False Negative)
        
        TP = torch.sum(confusion_vector == 1).item()
        FP = torch.sum(confusion_vector == float('inf')).item()
        
        self.FP += FP
        self.TP += TP

    def compute(self):
        precision = self.TP / (self.TP + self.FP)
        self._reset()
        return precision

    def _reset(self):
        self.FP = torch.Tensor([0]) # False-Positive
        self.TP = torch.Tensor([0]) # True-Positive

class F1Score(tm.Metric):
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

        self.FP = torch.Tensor([0]) # False-Positive
        self.TP = torch.Tensor([0]) # True-Positive
        self.FN = torch.Tensor([0]) # False-Negative

    def update(self, y_hat, y):
        y_hat_categorical = y_hat.argmax(1)
        
        confusion_vector = y_hat_categorical / y
        # Element-wise division of the 2 tensors returns a new tensor which holds a
        # unique value for each case:
        #   1     where prediction and truth are 1 (True Positive)
        #   inf   where prediction is 1 and truth is 0 (False Positive)
        #   nan   where prediction and truth are 0 (True Negative)
        #   0     where prediction is 0 and truth is 1 (False Negative)
        
        TP = torch.sum(confusion_vector == 1).item()
        FP = torch.sum(confusion_vector == float('inf')).item()
        FN = torch.sum(confusion_vector == 0).item()

        self.FP += FP
        self.TP += TP
        self.FN += FN

    def compute(self):
        precision = self.TP / (self.TP + self.FP)
        recall    = self.TP / (self.TP + self.FN)

        f1_score = 2 * precision * recall / (precision + recall)

        self._reset()
        return f1_score

    def _reset(self):
        self.FP = torch.Tensor([0]) # False-Positive
        self.TP = torch.Tensor([0]) # True-Positive
        self.FN = torch.Tensor([0]) # False-Negative

class TruePositive(tm.Metric):
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

        self.TP = torch.Tensor([0]) # True-Positive

    def update(self, y_hat, y):
        y_hat_categorical = y_hat.argmax(1)
        
        confusion_vector = y_hat_categorical / y
        # Element-wise division of the 2 tensors returns a new tensor which holds a
        # unique value for each case:
        #   1     where prediction and truth are 1 (True Positive)
        #   inf   where prediction is 1 and truth is 0 (False Positive)
        #   nan   where prediction and truth are 0 (True Negative)
        #   0     where prediction is 0 and truth is 1 (False Negative)
        
        TP = torch.sum(confusion_vector == 1).item()
        self.TP += TP

    def compute(self):
        TP = self.TP
        self._reset()
        return TP
    
    def _reset(self):
        self.TP = torch.Tensor([0]) # True-Positive        



class FalsePositive(tm.Metric):
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

        self.FP = torch.Tensor([0]) # False-Positive

    def update(self, y_hat, y):
        y_hat_categorical = y_hat.argmax(1)
        
        confusion_vector = y_hat_categorical / y
        # Element-wise division of the 2 tensors returns a new tensor which holds a
        # unique value for each case:
        #   1     where prediction and truth are 1 (True Positive)
        #   inf   where prediction is 1 and truth is 0 (False Positive)
        #   nan   where prediction and truth are 0 (True Negative)
        #   0     where prediction is 0 and truth is 1 (False Negative)
        
        FP = torch.sum(confusion_vector == float('inf')).item()
        self.FP += FP

    def compute(self):
        FP = self.FP
        self._reset()
        return FP
    
    def _reset(self):
        self.FP = torch.Tensor([0]) # False-Positive   



class TrueNegative(tm.Metric):
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

        self.TN = torch.Tensor([0]) # True-Negative

    def update(self, y_hat, y):
        y_hat_categorical = y_hat.argmax(1)
        
        confusion_vector = y_hat_categorical / y
        # Element-wise division of the 2 tensors returns a new tensor which holds a
        # unique value for each case:
        #   1     where prediction and truth are 1 (True Positive)
        #   inf   where prediction is 1 and truth is 0 (False Positive)
        #   nan   where prediction and truth are 0 (True Negative)
        #   0     where prediction is 0 and truth is 1 (False Negative)
        
        TN = torch.sum(torch.isnan(confusion_vector)).item()

        self.TN += TN

    def compute(self):
        TN = self.TN
        self._reset()
        return TN
    
    def _reset(self):
        self.TN = torch.Tensor([0]) # True-Negative   



class FalseNegative(tm.Metric):
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

        self.FN = torch.Tensor([0]) # False-Negative

    def update(self, y_hat, y):
        y_hat_categorical = y_hat.argmax(1)
        
        confusion_vector = y_hat_categorical / y
        # Element-wise division of the 2 tensors returns a new tensor which holds a
        # unique value for each case:
        #   1     where prediction and truth are 1 (True Positive)
        #   inf   where prediction is 1 and truth is 0 (False Positive)
        #   nan   where prediction and truth are 0 (True Negative)
        #   0     where prediction is 0 and truth is 1 (False Negative)
        
        FN = torch.sum(confusion_vector == 0).item()

        self.FN += FN

    def compute(self):
        FN = self.FN
        self._reset()
        return FN
    
    def _reset(self):
        self.FN = torch.Tensor([0]) # False-Negative   

class Confusion_Matrix(tm.Metric):
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

        self.FP = torch.Tensor([0]) # False-Positive
        self.TP = torch.Tensor([0]) # True-Positive
        self.FN = torch.Tensor([0]) # False-Negative
        self.TN = torch.Tensor([0]) # True-Negative

    def update(self, y_hat, y):
        y_hat_categorical = y_hat.argmax(1)
        
        confusion_vector = y_hat_categorical / y
        # Element-wise division of the 2 tensors returns a new tensor which holds a
        # unique value for each case:
        #   1     where prediction and truth are 1 (True Positive)
        #   inf   where prediction is 1 and truth is 0 (False Positive)
        #   nan   where prediction and truth are 0 (True Negative)
        #   0     where prediction is 0 and truth is 1 (False Negative)
        
        FP = torch.sum(confusion_vector == float('inf')).item()
        TP = torch.sum(confusion_vector == 1).item()
        FN = torch.sum(confusion_vector == 0).item()
        TN = torch.sum(torch.isnan(confusion_vector)).item()

        self.FP += FP
        self.TP += TP
        self.FN += FN
        self.TN += TN

    def compute(self):
        TP, FP, TN, FN = self.TP, self.FP, self.TN, self.FN
        self._reset()
        return TP, FP, TN, FN

    def _reset(self):
        self.FP = torch.Tensor([0]) # False-Positive
        self.TP = torch.Tensor([0]) # True-Positive
        self.FN = torch.Tensor([0]) # False-Negative
        self.TN = torch.Tensor([0]) # True-Negative





