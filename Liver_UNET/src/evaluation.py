# ============================================================
# Liver Tumor Segmentation - Evaluation Script (FINAL)
# ============================================================

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib

# IMPORTANT: Force matplotlib to save files on Windows
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================================================
# 1. CONFIGURATION
# ============================================================

IMG_SIZE = 128
BATCH_SIZE = 8
MAX_EVAL_SAMPLES = 500   # safe for RAM (increase only if you have >16GB RAM)

# ============================================================
# 2. PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results", "evaluation")

os.makedirs(RESULTS_DIR, exist_ok=True)

images_path = os.path.join(DATASET_PATH, "images")
masks_path  = os.path.join(DATASET_PATH, "masks")

# ---------- MODEL PATH AUTO-DETECT ----------
POSSIBLE_MODEL_PATHS = [
    os.path.join(BASE_DIR, "..", "unet_liver_tumor.h5"),
    os.path.join(BASE_DIR, "unet_liver_tumor.h5"),
    os.path.join(BASE_DIR, "..", "models", "unet_liver_tumor.h5"),
]

MODEL_PATH = None
for p in POSSIBLE_MODEL_PATHS:
    if os.path.exists(p):
        MODEL_PATH = p
        break

if MODEL_PATH is None:
    raise FileNotFoundError("❌ unet_liver_tumor.h5 not found")

print("✅ Model found at:", MODEL_PATH)

# ============================================================
# 3. LOAD MODEL
# ============================================================

print("\n⏳ Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded")

# ============================================================
# 4. METRIC FUNCTIONS
# ============================================================

def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-7)

def iou_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / (union + 1e-7)

# ============================================================
# 5. LOAD IMAGE LIST (NO IMAGE DATA YET)
# ============================================================

image_names = sorted(os.listdir(images_path))

if len(image_names) == 0:
    raise RuntimeError("❌ No images found in dataset/images")

# Use only last N samples (acts like validation set)
image_names = image_names[-MAX_EVAL_SAMPLES:]

print(f"🧪 Evaluating on {len(image_names)} samples")

# ============================================================
# 6. EVALUATION LOOP (MEMORY SAFE)
# ============================================================

dice_scores = []
iou_scores = []

print("\n⏳ Running evaluation...")

for i in tqdm(range(0, len(image_names), BATCH_SIZE), desc="Evaluating"):
    batch_names = image_names[i:i + BATCH_SIZE]

    X_batch = []
    y_batch = []

    for name in batch_names:
        img = cv2.imread(os.path.join(images_path, name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(masks_path, name), cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE)) / 255.0

        X_batch.append(img)
        y_batch.append(mask)

    X_batch = np.array(X_batch)[..., np.newaxis]
    y_batch = np.array(y_batch)[..., np.newaxis]

    preds = model.predict(X_batch, verbose=0)
    preds_bin = (preds > 0.5).astype(np.uint8)

    for j in range(len(batch_names)):
        dice_scores.append(dice_coefficient(y_batch[j], preds_bin[j]))
        iou_scores.append(iou_score(y_batch[j], preds_bin[j]))

# ============================================================
# 7. FINAL METRICS
# ============================================================

mean_dice = np.mean(dice_scores)
mean_iou  = np.mean(iou_scores)

print("\n📊 FINAL METRICS")
print(f"Mean Dice : {mean_dice:.4f}")
print(f"Mean IoU  : {mean_iou:.4f}")

# ============================================================
# 8. SAVE METRICS TO FILE
# ============================================================

metrics_file = os.path.join(RESULTS_DIR, "metrics.txt")
with open(metrics_file, "w") as f:
    f.write(f"Mean Dice: {mean_dice:.4f}\n")
    f.write(f"Mean IoU : {mean_iou:.4f}\n")

print("✅ Metrics saved:", metrics_file)

# ============================================================
# 9. SAFE SAVE FUNCTION (WINDOWS FIX)
# ============================================================

def safe_save(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if os.path.exists(path):
        print("✅ Saved:", path)
    else:
        print("❌ Failed to save:", path)

# ============================================================
# 10. SAVE GRAPHS (DIFFERENT COLORS)
# ============================================================

print("\n⏳ Saving graphs...")

# ---- Dice & IoU Histogram ----
fig1 = plt.figure(figsize=(8, 5))
plt.hist(dice_scores, bins=20, alpha=0.7, color="green", label="Dice")
plt.hist(iou_scores, bins=20, alpha=0.7, color="orange", label="IoU")
plt.title("Dice & IoU Score Distribution")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)

safe_save(fig1, os.path.join(RESULTS_DIR, "dice_iou_distribution.png"))

# ---- Overall Metrics Bar Chart ----
fig2 = plt.figure(figsize=(6, 5))
plt.bar(
    ["Dice", "IoU"],
    [mean_dice, mean_iou],
    color=["green", "red"]
)
plt.ylim(0, 1)
plt.title("Overall Evaluation Metrics")
plt.grid(axis="y")

safe_save(fig2, os.path.join(RESULTS_DIR, "overall_metrics.png"))

# ============================================================
# 11. FINAL CONFIRMATION
# ============================================================

print("\n📁 Files inside results folder:")
print(os.listdir(RESULTS_DIR))

print("\n✅ EVALUATION COMPLETED SUCCESSFULLY")
print("📂 Results location:", RESULTS_DIR)
