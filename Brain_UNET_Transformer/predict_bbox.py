import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model_transformer import UNetTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load trained model
model = UNetTransformer().to(device)

state_dict = torch.load("unet_brain_tumor.pth", map_location=device)
model.load_state_dict(state_dict, strict=False)

model.eval()

# ----------- GIVE NEW IMAGE PATH HERE -----------
img_path = "new_images-1.jpg"
# -----------------------------------------------

# Read image
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
image_resized = cv2.resize(image, (256, 256))
image_norm = image_resized.astype(np.float32) / 255.0

input_tensor = torch.tensor(image_norm).unsqueeze(0).unsqueeze(0).to(device)

# Predict mask
with torch.no_grad():
    pred = model(input_tensor)
    pred = torch.sigmoid(pred)
    pred = pred.squeeze().cpu().numpy()

binary_mask = (pred > 0.5).astype(np.uint8) * 255

# Find contours
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

image_color = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2BGR)

if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cv2.rectangle(image_color, (x, y), (x + w, y + h), (0, 0, 255), 2)

plt.figure(figsize=(6, 6))
plt.title("Tumor Localization (Bounding Box)")
plt.imshow(image_color)
plt.axis("off")
plt.show()
