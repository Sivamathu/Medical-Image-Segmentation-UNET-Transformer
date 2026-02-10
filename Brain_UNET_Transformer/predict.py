import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model_transformer import UNetTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = UNetTransformer().to(device)
model.load_state_dict(torch.load("unet_brain_tumor.pth", map_location=device))
model.eval()

# Load one test image
img_path = "dataset/images/" + sorted(os.listdir("dataset/images"))[0]
mask_path = "dataset/masks/" + sorted(os.listdir("dataset/masks"))[0]

image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

image_resized = cv2.resize(image, (256, 256))
mask_resized = cv2.resize(mask, (256, 256))

image_norm = image_resized.astype(np.float32) / 255.0
input_tensor = torch.tensor(image_norm).unsqueeze(0).unsqueeze(0).to(device)

# Prediction
with torch.no_grad():
    pred = model(input_tensor)
    pred = torch.sigmoid(pred)
    pred = pred.squeeze().cpu().numpy()
    pred = (pred > 0.5).astype(np.uint8)

# Visualization
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Original MRI")
plt.imshow(image_resized, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Ground Truth Mask")
plt.imshow(mask_resized, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Predicted Mask")
plt.imshow(pred, cmap="gray")
plt.axis("off")

plt.show()
