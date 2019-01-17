import torch
import torch.nn.functional as F


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
        for c in range(9):
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
        return loss/target.size(0)

class SoftDiceLoss(torch.nn.Module):
    '''
    Soft Dice Loss
    '''

    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1.
        logits = F.softmax(logits)
        iflat = logits.view(-1)
        tflat = targets.view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
    
class LogDiceLoss(torch.nn.Module):
    '''
    Soft Dice Loss
    '''

    def __init__(self, weight=None, size_average=True):
        super(LogDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1.
        logits = F.sigmoid(logits)
        iflat = logits.view(-1)
        tflat = targets.view(-1)
        intersection = (iflat * tflat).sum()
        return -torch.log((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
    
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):

        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = F.logsigmoid(-input * (target * 2 - 1))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.mean()
    
    
class f2loss(torch.nn.Module):
    def __init__(self, beta =2):
        super().__init__()
        self.beta = beta
    
    def forward(self, y_pred, y_true):
        
        assert y_pred.shape == y_true.shape
        y_pred = torch.nn.functional.sigmoid(y_pred)
        beta_sq = self.beta ** 2
        bce = torch.nn.functional.binary_cross_entropy(y_pred, y_true)
        tp_loss = torch.sum(y_true * (1- bce))
        fp_loss = torch.sum((1-y_true) * bce)
        
        return -torch.mean((1+beta_sq) * tp_loss/((beta_sq * torch.sum(y_true)) + tp_loss + fp_loss))