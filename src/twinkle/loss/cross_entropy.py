from src.twinkle.loss.base import Loss
import torch


class CrossEntropyLoss(Loss):

    def __call__(self, logits, labels, **kwargs):
        return torch.nn.CrossEntropyLoss()(logits, labels)