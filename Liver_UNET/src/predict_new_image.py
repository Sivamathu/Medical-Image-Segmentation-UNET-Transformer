# ============================================================
# Liver Tumor Prediction on New Image (TensorFlow)
# ============================================================

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ---------------- LOAD MODEL ----------------
MODEL_PATH = "unet_liver_tumor.h5"
model = tf.keras.models.load_model(MODEL_PATH)
print("[INFO] Model loaded successfully")

# ---------------- GIVE NEW IMAGE PATH HERE ----------------
img_path = "images.jpg"   # <-- change this
# ----------------------------------------------------------

SAVE_DIR = "results/predictions"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------- READ & PREPROCESS IMAGE ----------------
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError("❌ Image not found. Check the image path.")

image_resized = cv2.resize(image, (128, 128))
image_norm = image_resized.astype(np.float32) / 255.0

# Model expects: (batch, height, width, channels)
input_tensor = np.expand_dims(image_norm, axis=(0, -1))

# ---------------- PREDICT ----------------
pred = model.predict(input_tensor)[0, :, :, 0]

binary_mask = (pred > 0.5).astype(np.uint8) * 255

# ---------------- FIND BOUNDING BOX ----------------
contours, _ = cv2.findContours(
    binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

image_color = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2BGR)

if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cv2.rectangle(image_color, (x, y), (x + w, y + h), (0, 0, 255), 2)
    print("[INFO] Tumor detected")
else:
    print("[INFO] No tumor detected")

# ---------------- SAVE IMAGE ----------------
save_path = os.path.join(SAVE_DIR, "tumor_prediction_with_bbox.png")
cv2.imwrite(save_path, image_color)

print(f"✅ Prediction image saved at: {save_path}")

# ---------------- DISPLAY RESULTS ----------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Input CT Image")
plt.imshow(image_resized, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Predicted Mask")
plt.imshow(binary_mask, cmap="Reds")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Tumor Localization")
plt.imshow(image_color)
plt.axis("off")

plt.tight_layout()
plt.show()
