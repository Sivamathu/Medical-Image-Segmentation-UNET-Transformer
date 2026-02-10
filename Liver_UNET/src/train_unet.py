# ============================================================
# Liver Tumor Segmentation using U-Net
# ============================================================

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from scipy.spatial.distance import directed_hausdorff

# ============================================================
# 1. Dataset Path (Portable for VS Code)
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset")

images_path = os.path.join(DATASET_PATH, "images")
masks_path  = os.path.join(DATASET_PATH, "masks")

# ============================================================
# 2. Load Dataset
# ============================================================

images = []
masks = []

print("[INFO] Loading dataset...")

image_names = sorted(os.listdir(images_path))[:500]  # limit for demo

for img_name in tqdm(image_names):
    img = cv2.imread(os.path.join(images_path, img_name), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(os.path.join(masks_path, img_name), cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (128, 128))
    mask = cv2.resize(mask, (128, 128))

    images.append(img)
    masks.append(mask)

images = np.array(images, dtype=np.float32) / 255.0
masks  = np.array(masks, dtype=np.float32) / 255.0

images = np.expand_dims(images, axis=-1)
masks  = np.expand_dims(masks, axis=-1)

print("Images shape:", images.shape)
print("Masks shape :", masks.shape)

# ============================================================
# 3. Train / Validation Split
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    images, masks, test_size=0.2, random_state=42
)

print("Train:", X_train.shape, y_train.shape)
print("Val  :", X_val.shape, y_val.shape)

# ============================================================
# 4. Data Augmentation
# ============================================================

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# ============================================================
# 5. U-Net Model
# ============================================================

def unet_model(input_size=(128, 128, 1)):
    inputs = layers.Input(input_size)

    # Encoder
    c1 = layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
    c1 = layers.Dropout(0.1)(c1)
    c1 = layers.Conv2D(16, 3, activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(32, 3, activation="relu", padding="same")(p1)
    c2 = layers.Dropout(0.1)(c2)
    c2 = layers.Conv2D(32, 3, activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(64, 3, activation="relu", padding="same")(p2)
    c3 = layers.Dropout(0.2)(c3)
    c3 = layers.Conv2D(64, 3, activation="relu", padding="same")(c3)
    p3 = layers.MaxPooling2D()(c3)

    # Bottleneck
    c4 = layers.Conv2D(128, 3, activation="relu", padding="same")(p3)
    c4 = layers.Dropout(0.3)(c4)
    c4 = layers.Conv2D(128, 3, activation="relu", padding="same")(c4)

    # Decoder
    u5 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv2D(64, 3, activation="relu", padding="same")(u5)
    c5 = layers.Conv2D(64, 3, activation="relu", padding="same")(c5)

    u6 = layers.Conv2DTranspose(32, 2, strides=2, padding="same")(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(32, 3, activation="relu", padding="same")(u6)
    c6 = layers.Conv2D(32, 3, activation="relu", padding="same")(c6)

    u7 = layers.Conv2DTranspose(16, 2, strides=2, padding="same")(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(16, 3, activation="relu", padding="same")(u7)
    c7 = layers.Conv2D(16, 3, activation="relu", padding="same")(c7)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c7)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

model = unet_model()
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================================================
# 6. Train Model with Early Stopping
# ============================================================

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=5,
        restore_best_weights=True
    )
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=16,
    epochs=40,
    callbacks=callbacks
)

# ============================================================
# 7. Evaluate Model
# ============================================================

loss, acc = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {acc * 100:.2f}%")

# ============================================================
# 8. Metrics
# ============================================================

def dice_coefficient(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1e-7)

def hausdorff_distance(y_true, y_pred):
    coords_true = np.argwhere(y_true)
    coords_pred = np.argwhere(y_pred)
    if len(coords_true) == 0 or len(coords_pred) == 0:
        return np.nan
    return max(
        directed_hausdorff(coords_true, coords_pred)[0],
        directed_hausdorff(coords_pred, coords_true)[0]
    )

# ============================================================
# 9. Visualization
# ============================================================

def plot_predictions(model, X, y, num=3):
    preds = model.predict(X[:num])

    for i in range(num):
        img = X[i].squeeze()
        gt = (y[i].squeeze() > 0.5).astype(np.uint8)
        pr = (preds[i].squeeze() > 0.5).astype(np.uint8)

        dice = dice_coefficient(gt, pr)
        haus = hausdorff_distance(gt, pr)

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.title("Input CT")
        plt.imshow(img, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Ground Truth (Tumor)")
        plt.imshow(img, cmap="gray")
        plt.imshow(gt, cmap="Greens", alpha=0.4)
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Prediction")
        plt.imshow(img, cmap="gray")
        plt.imshow(pr, cmap="Reds", alpha=0.4)
        plt.axis("off")

        plt.suptitle(
            f"Liver Tumor Segmentation | Dice: {dice:.3f} | Hausdorff: {haus:.2f}px",
            fontsize=14,
            fontweight="bold"
        )

        plt.show()

# Run visualization
plot_predictions(model, X_val, y_val, num=3)

# ============================================================
# 10. Save Model
# ============================================================

model.save("unet_liver_tumor.h5")
print("[INFO] Model saved as unet_liver_tumor.h5")
