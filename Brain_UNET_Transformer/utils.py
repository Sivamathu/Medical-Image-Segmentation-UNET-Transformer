import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        smooth = 1e-6
        intersection = (preds * targets).sum()
        dice = (2. * intersection + smooth) / (
            preds.sum() + targets.sum() + smooth
        )
        return 1 - dice


def dice_score(preds, targets):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()   # binarize
    smooth = 1e-6
    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (
        preds.sum() + targets.sum() + smooth
    )
    return dice
