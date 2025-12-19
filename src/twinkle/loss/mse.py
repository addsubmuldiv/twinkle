from src.twinkle.loss.base import Loss
import torch


class MSELoss(Loss):

    def __call__(self, preds, labels, **kwargs):
        return torch.nn.MSELoss()(preds, labels)