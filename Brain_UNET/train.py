import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BrainTumorDataset, get_transforms
from model import UNet
from utils import DiceLoss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    dataset = BrainTumorDataset(
        "dataset/images",
        "dataset/masks",
        transform=get_transforms()
    )

    train_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,   # IMPORTANT (see below)
        pin_memory=True
    )

    model = UNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    dice_loss = DiceLoss()
    bce_loss = nn.BCEWithLogitsLoss()

    epochs = 30

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        loop = tqdm(train_loader)
        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)
            loss = dice_loss(preds, masks) + bce_loss(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} Average Loss: {epoch_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "unet_brain_tumor.pth")
    print("Model saved as unet_brain_tumor.pth")


if __name__ == "__main__":
    main()
