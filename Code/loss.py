import torch
import torch.nn.functional as F
from torch.autograd import Variable


class DiceLoss(torch.nn.Module):
    def __init__(self, class_weights=None):
        super(DiceLoss, self).__init__()
        self.class_weights = class_weights

    def forward(self, output, target):
        softmax = torch.nn.Softmax2d()
        smooth = 1.
        loss = 0.
        output = softmax(output)
        target = target.float()
        for c in range(8):
            iflat = output[:, c].contiguous().view(-1)
            tflat = target[:, c].contiguous().view(-1)
            intersection = (iflat * tflat).sum()

            if self.class_weights is not None:
                w = self.class_weights[c]
                loss += w * (1 - ((2. * intersection + smooth) /
                                  (iflat.sum() + tflat.sum() + smooth)))
            else:
                loss += (1 - ((2. * intersection + smooth) /
                              (iflat.sum() + tflat.sum() + smooth)))
        return loss.mean() / 9


class FocalLoss(torch.nn.Module):

    def __init__(self, gamma=1.2, weight=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average

    def forward(self, output, target):
        logpt = -F.cross_entropy(output, target, weight=self.weight)
        pt = torch.exp(logpt)
        # compute the loss
        loss = -((1 - pt) ** self.gamma) * logpt
        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

