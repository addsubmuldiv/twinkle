from src.twinkle.loss.base import Loss
import torch


class CosineSimilarityLoss(Loss):

    def __call__(self, sentence1, sentence2, labels, **kwargs):
        cos_score_transformation = torch.nn.Identity()
        loss_fct = torch.MSELoss()
        output = cos_score_transformation(torch.cosine_similarity(sentence1, sentence2))
        return loss_fct(output, labels.to(output.dtype).view(-1))