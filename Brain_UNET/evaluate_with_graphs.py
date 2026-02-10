import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import BrainTumorDataset
from model import UNet
from utils import dice_score, iou_score

# ---------------- SETUP ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_DIR = "results/graphs"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------- DATA ----------------
dataset = BrainTumorDataset(
    "dataset/images",
    "dataset/masks"
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

# ---------------- MODEL ----------------
model = UNet().to(device)
model.load_state_dict(torch.load("unet_brain_tumor.pth", map_location=device))
model.eval()

dice_scores = []
iou_scores = []

# ---------------- EVALUATION ----------------
with torch.no_grad():
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)

        dice = dice_score(preds, masks)
        iou = iou_score(preds, masks)

        dice_scores.append(dice.item())
        iou_scores.append(iou.item())

dice_scores = np.array(dice_scores)
iou_scores = np.array(iou_scores)

print("Average Dice Score:", dice_scores.mean())
print("Average IoU Score :", iou_scores.mean())

# ---------------- GRAPHS ----------------

# 1️⃣ Dice Histogram
plt.figure()
plt.hist(dice_scores, bins=20)
plt.title("Dice Score Distribution")
plt.xlabel("Dice Score")
plt.ylabel("Frequency")
plt.savefig(f"{SAVE_DIR}/dice_histogram.png")
plt.close()

# 2️⃣ IoU Histogram
plt.figure()
plt.hist(iou_scores, bins=20)
plt.title("IoU Score Distribution")
plt.xlabel("IoU Score")
plt.ylabel("Frequency")
plt.savefig(f"{SAVE_DIR}/iou_histogram.png")
plt.close()

# 3️⃣ Dice vs IoU Scatter
plt.figure()
plt.scatter(dice_scores, iou_scores)
plt.title("Dice vs IoU Scatter Plot")
plt.xlabel("Dice Score")
plt.ylabel("IoU Score")
plt.savefig(f"{SAVE_DIR}/dice_vs_iou_scatter.png")
plt.close()

# 4️⃣ Box Plot Comparison
plt.figure()
plt.boxplot([dice_scores, iou_scores], labels=["Dice", "IoU"])
plt.title("Dice vs IoU Box Plot")
plt.ylabel("Score")
plt.savefig(f"{SAVE_DIR}/boxplot_dice_iou.png")
plt.close()

# 5️⃣ Average Score Bar Chart
plt.figure()
plt.bar(["Dice", "IoU"], [dice_scores.mean(), iou_scores.mean()])
plt.title("Average Segmentation Scores")
plt.ylabel("Score")
plt.savefig(f"{SAVE_DIR}/average_scores_bar.png")
plt.close()

print(f"\nGraphs saved successfully in: {SAVE_DIR}")
