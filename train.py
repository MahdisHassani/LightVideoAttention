import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from src.models.model import LiteUNet
from src.data.dataset import VideoDataset


# Dice Loss
class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)

        smooth = 1e-6
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()

        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice


# Metrics
def compute_metrics(preds, targets):
    preds = torch.sigmoid(preds)

    preds_bin = (preds > 0.3).float()

    smooth = 1e-6

    intersection = (preds_bin * targets).sum()
    union = preds_bin.sum() + targets.sum()
    dice = (2 * intersection + smooth) / (union + smooth)

    union_iou = preds_bin.sum() + targets.sum() - intersection
    iou = (intersection + smooth) / (union_iou + smooth)

    mae = torch.abs(preds - targets).mean()

    return dice.item(), iou.item(), mae.item()


# Train Function
def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = VideoDataset("data/processed/train")
    val_dataset   = VideoDataset("data/processed/val")

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=4, shuffle=False)

    model = LiteUNet().to(device)

    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")

    for epoch in range(20):

        model.train()
        train_loss = 0

        for imgs, flows in tqdm(train_loader):
            imgs = imgs.to(device)
            flows = flows.to(device)

            preds = model(imgs)
            
            loss = (0.5 * bce_loss(preds, flows) + 0.5 * dice_loss(preds, flows))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)


        model.eval()
        val_loss = 0

        total_dice = 0
        total_iou = 0
        total_mae = 0

        with torch.no_grad():
            for imgs, flows in val_loader:
                imgs = imgs.to(device)
                flows = flows.to(device)

                preds = model(imgs)
                
                loss = (0.5 * bce_loss(preds, flows) + 0.5 * dice_loss(preds, flows))

                val_loss += loss.item()

                dice, iou, mae = compute_metrics(preds, flows)

                total_dice += dice
                total_iou += iou
                total_mae += mae

        val_loss /= len(val_loader)
        avg_dice = total_dice / len(val_loader)
        avg_iou  = total_iou / len(val_loader)
        avg_mae  = total_mae / len(val_loader)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Dice: {avg_dice:.4f} | IoU: {avg_iou:.4f} | MAE: {avg_mae:.4f}")


        if val_loss < best_val_loss:
            best_val_loss = val_loss

            os.makedirs("outputs/models", exist_ok=True)
            torch.save(model.state_dict(), "outputs/models/best_model.pth")

            print("✅ Best model saved!")

    print("Training Finished!")


if __name__ == "__main__":
    train()