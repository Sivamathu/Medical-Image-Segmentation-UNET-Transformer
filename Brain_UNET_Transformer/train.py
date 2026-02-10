import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import BrainTumorDataset, get_transforms
from model_transformer import UNetTransformer
from utils import DiceLoss, dice_score


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Load dataset
    dataset = BrainTumorDataset(
        "dataset/images",
        "dataset/masks",
        transform=get_transforms()
    )

    # Train / Validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Model
    model = UNetTransformer().to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    dice_loss = DiceLoss()
    bce_loss = nn.BCEWithLogitsLoss()

    epochs = 30

    for epoch in range(epochs):
        # ---------------- TRAIN ----------------
        model.train()
        train_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)
            loss = dice_loss(preds, masks) + bce_loss(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(train_loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # ---------------- VALIDATION ----------------
        model.eval()
        val_dice = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                preds = model(images)
                val_dice += dice_score(preds, masks).item()

        avg_val_dice = val_dice / len(val_loader)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Dice: {avg_val_dice:.4f}"
        )

    # Save model
    torch.save(model.state_dict(), "unet_brain_tumor.pth")
    print("Model saved as unet_brain_tumor.pth")


if __name__ == "__main__":
    main()
