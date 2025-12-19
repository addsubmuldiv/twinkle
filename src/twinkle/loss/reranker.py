from src.twinkle.loss.base import Loss
import torch


class RerankerLoss(Loss):

    def __call__(self, logits, labels, **kwargs):
        logits = logits.squeeze(1)
        labels = labels.to(logits.dtype)
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)
        return loss