import torch
import torch.nn.functional as F

from .base import BaseLoss


class CrossEntropyLoss(BaseLoss):
    def __init__(self, scale=2**4, label_smooth=True, eps=0.1, loss_term_weight=1.0, log_accuracy=False):
        super(CrossEntropyLoss, self).__init__(loss_term_weight)
        self.scale = scale
        self.label_smooth = label_smooth
        self.eps = eps
        self.log_accuracy = log_accuracy

    def forward(self, logits, labels):
        """
            logits: [n, c]
            labels: [n]
        """
        n, c = logits.size()
        log_preds = F.log_softmax(logits, dim=1)  # [n, c]
        one_hot_labels = self.label2one_hot(
            labels, c)
        # print('shape in softmax.py is {}'.format(one_hot_labels.shape))
        # print('shape in softmax.py after unsqueeze is {}'.format(one_hot_labels.shape))
        # print(labels,one_hot_labels)
        loss = self.compute_loss(log_preds, one_hot_labels)
        # print('loss is {}'.format(loss))
        self.info.update({'loss': loss.detach().clone()})
        if self.log_accuracy:
            pred = logits.argmax(dim=-1)  # [n, p]

            ## change
            # print('in accu, pred shape is {} ,labels shape is {}'.format(pred.shape,labels.shape))
            # print('pred is {}'.format(pred))
            # print('labels is {}'.format(labels))
            accu = (pred == labels).float().mean()
            # print('acc is {}'.format(accu))
            

            self.info.update({'accuracy': accu})
        return loss, self.info

    def compute_loss(self, predis, labels):
        # print('in compute_loss, pred shape is {} ,labels shape is {}'.format(predis.shape,labels.shape))
        softmax_loss = -(labels * predis).sum(1)  # [n, p]
        losses = softmax_loss.mean(0)   # [p]

        if self.label_smooth:
            smooth_loss = - predis.mean(dim=1)  # [n, p]
            smooth_loss = smooth_loss.mean(0)  # [p]
            losses = smooth_loss * self.eps + losses * (1. - self.eps)
        return losses

    def label2one_hot(self, label, class_num):
        label = label.unsqueeze(-1)
        batch_size = label.size(0)
        device = label.device
        # print(batch_size,class_num,label)
        return torch.zeros(batch_size, class_num).to(device).scatter(1, label, 1)
