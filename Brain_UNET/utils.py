import torch
import torch.nn as nn


# ---------------- DICE LOSS ----------------
class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)

        smooth = 1e-6
        intersection = (preds * targets).sum()
        dice = (2 * intersection + smooth) / (
            preds.sum() + targets.sum() + smooth
        )
        return 1 - dice


# ---------------- DICE SCORE ----------------
def dice_score(preds, targets):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

    smooth = 1e-6
    intersection = (preds * targets).sum()
    dice = (2 * intersection + smooth) / (
        preds.sum() + targets.sum() + smooth
    )
    return dice


# ---------------- IOU SCORE ----------------
def iou_score(preds, targets):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

    smooth = 1e-6
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou
