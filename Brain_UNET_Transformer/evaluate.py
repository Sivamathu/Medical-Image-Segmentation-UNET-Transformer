import torch
from torch.utils.data import DataLoader
from dataset import BrainTumorDataset
from model_transformer import UNetTransformer
from utils import dice_score
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = BrainTumorDataset("dataset/images", "dataset/masks")
loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

model = UNetTransformer().to(device)
model.load_state_dict(torch.load("unet_brain_tumor.pth", map_location=device))
model.eval()

dice_scores = []
iou_scores = []

with torch.no_grad():
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)
        preds = torch.sigmoid(preds)
        preds = (preds > 0.5).float()

        intersection = (preds * masks).sum()
        union = preds.sum() + masks.sum() - intersection

        dice = (2 * intersection) / (preds.sum() + masks.sum() + 1e-6)
        iou = intersection / (union + 1e-6)

        dice_scores.append(dice.item())
        iou_scores.append(iou.item())

print("Average Dice Score:", np.mean(dice_scores))
print("Average IoU Score :", np.mean(iou_scores))
