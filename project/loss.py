import torch
from torch import nn
from torch.nn import functional as F
import utils
import numpy as np


class LossBinary:
    """
    Loss defined as \alpha BCE - (1 - \alpha) SoftJaccard
    """

    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1).float()
            jaccard_output = torch.sigmoid(outputs)

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))
        return loss


class LossMulti:
    def __init__(self, jaccard_weight=0, class_weights=None, num_classes=1):
        if class_weights is not None:
            nll_weight = utils.cuda(
                torch.from_numpy(class_weights.astype(np.float32)))
        else:
            nll_weight = None
        self.nll_loss = nn.NLLLoss(weight=nll_weight)
        self.jaccard_weight = jaccard_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            for cls in range(self.num_classes):
                jaccard_target = (targets == cls).float()
                jaccard_output = outputs[:, cls].exp()
                intersection = (jaccard_output * jaccard_target).sum()

                union = jaccard_output.sum() + jaccard_target.sum()
                loss -= torch.log((intersection + eps) / (union - intersection + eps)) * self.jaccard_weight
        return loss


class LossAll:
    def __init__(self, alpha, beta, gamma, jaccard_weight=0, class_weights=None, num_classes=1):
        if class_weights is not None:
            self.nll_weight = utils.cuda(
                torch.from_numpy(class_weights.astype(np.float32)))
        else:
            self.nll_weight = None
        self.jaccard_weight = jaccard_weight
        self.num_classes = num_classes
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
    def __call__(self, output_binary, target_binary,output_parts,target_parts,output_instruments,target_instruments):
        loss1=LossBinary(jaccard_weight=self.jaccard_weight)
        loss_binary=loss1(output_binary,target_binary)
        loss2=LossMulti(self.jaccard_weight,self.nll_weight,4)
        loss_parts=loss2(output_parts,target_parts)
        loss3=LossMulti(self.jaccard_weight,self.nll_weight,8)
        loss_instruments=loss3(output_instruments,target_instruments)
        loss = self.alpha*loss_binary+self.beta+loss_parts+self.gamma*loss_instruments
        return loss

