'''Contains the MetricsManager class to manage metrics on multi class segmentation tasks
and the LossManager class to manage loss during training'''
import torch

class MetricsManager():
    '''Class to manage metrics on multi class segmentation tasks
    Use metrics attribute to access the metrics computed on the batches
    Use update to add a new batch to the metrics'''

    # CONSTRUCTOR

    def __init__(self, num_classes=10, device='cpu'):
        self.metrics = {'accuracy': [], 'precision': [], 'recall': [], 'dice_score': []}
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.num_classes = num_classes
        self.device = 'cpu'

    # REQUESTS

    def _compute_confusion_matrix(self, y_pred, y_true):
        '''Compute the confusion matrix for the batch
        Return a tuple : (tp, fp, fn, tn)'''
        tp_k = []
        fp_k = []
        fn_k = []
        tn_k = []
        for k in range(1, self.num_classes):
            tp = ((y_pred == k) & (y_true == k)).sum().item()
            fn = ((y_true == k)).sum().item() - tp
            fp = ((y_pred == k)).sum().item() - tp
            tn = len(y_pred.flatten()) - tp - fn - fp
            tp_k.append(tp)
            fp_k.append(fp)
            fn_k.append(fn)
            tn_k.append(tn)
        return torch.mean(torch.tensor(tp_k, dtype=torch.float).to(self.device)), torch.mean(torch.tensor(fp_k, dtype=torch.float).to(self.device)), torch.mean(torch.tensor(fn_k, dtype=torch.float).to(self.device)), torch.mean(torch.tensor(tn_k, dtype=torch.float).to(self.device))
    
    def _compute_macro_metrics(self, tp, fp, fn, tn):
        '''Compute the macro metrics
        Return a tuple : (accuracy, precision, recall, dice_score)'''
        # Avoid division by zero
        epsilon = 1e-10
        accuracy = (tp + tn) / ((tp + fp + fn + tn) + epsilon)
        precision = tp / ((tp + fp) + epsilon)
        recall = tp / ((tp + fn) + epsilon)
        dice_score = 2 * tp / ((2 * tp + fp + fn) + epsilon)
        return accuracy, precision, recall, dice_score
    
    def get_overall_metrics(self):
        '''## Compute the overall metrics
        Return a tuple : (accuracy, precision, recall, dice_score)'''
        accuracy, precision, recall, dice_score = self._compute_macro_metrics(self.tp, self.fp, self.fn, self.tn)
        return accuracy, precision, recall, dice_score
    
    def get_metrics_var(self):
        '''Return the variance of the metrics
        Return a tuple : (accuracy, precision, recall, dice_score)'''
        accuracy = torch.var(torch.tensor(self.metrics['accuracy']))
        precision = torch.var(torch.tensor(self.metrics['precision']))
        recall = torch.var(torch.tensor(self.metrics['recall']))
        dice_score = torch.var(torch.tensor(self.metrics['dice_score']))
        return accuracy, precision, recall, dice_score

    # COMMANDS
    
    def update(self, y_pred, y_true):
        '''Update the metrics with the new batch
        y_pred and y_true must be torch tensors with the same shape'''
        tp, fp, fn, tn = self._compute_confusion_matrix(y_pred, y_true)
        accuracy, precision, recall, dice_score = self._compute_macro_metrics(tp, fp, fn, tn)
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn
        self.metrics['accuracy'].append(accuracy)
        self.metrics['precision'].append(precision)
        self.metrics['recall'].append(recall)
        self.metrics['dice_score'].append(dice_score)

class LossManager():
    '''Class to manage loss during training
    Use epoch_end function at the end of each epoch and use
    losses attribute to access the losses of each epochs'''
    def __init__(self) -> None:
        self.losses = []
        self.current_epoch_losses = []

    def epoch_end(self):
        self.losses.append(torch.mean(torch.tensor(self.current_epoch_losses)))
        self.current_epoch_losses = []

    def add(self, loss):
        '''Add the loss of the current batch to the current epoch losses'''
        self.current_epoch_losses.append(loss)

