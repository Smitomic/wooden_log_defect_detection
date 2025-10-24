import os
import argparse
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

from src.model import DilatedSegCNN
from src.datamodule import make_dataloaders
from src.postprocess.crf import apply_dense_crf
from src.preprocess import compute_log_bboxes

# region IoU utilities

def compute_iou(preds, targets, num_classes=7, ignore_index=None, return_per_class=False):
    preds = preds.view(-1)
    targets = targets.view(-1)
    ious = []

    for cls in range(num_classes):
        if ignore_index is not None and cls == ignore_index:
            ious.append(float('nan'))
            continue

        pred_mask = preds == cls
        target_mask = targets == cls

        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()

        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((intersection / union).item())

    mean_iou = np.nanmean(ious)
    if return_per_class:
        return mean_iou, ious
    else:
        return mean_iou

# endregion

# region Training loop

def train_model(
    root_dir,
    out_dir,
    epochs=25,
    batch_size=4,
    lr=1e-3,
    size=256,
    num_classes=7,
    early_stop_patience=5,
    use_crf=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    bbox_dict = compute_log_bboxes(root_dir)
    train_loader, val_loader = make_dataloaders(
        root_dir=root_dir, batch_size=batch_size, bbox_dict=bbox_dict, size=(size, size), split_mode="slice"
    )

    # Model
    model = DilatedSegCNN(in_channels=1, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Logging
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    best_iou = 0
    patience_counter = 0

    train_losses, val_losses, val_ious = [], [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]"):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss, val_iou = 0.0, 0.0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [val]"):
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                preds_softmax = torch.softmax(outputs, dim=1).cpu().numpy()
                imgs_np = imgs.cpu().numpy()

                refined_preds = []
                for b in range(imgs_np.shape[0]):
                    # Convert tensor to uint8 image
                    img_gray = (imgs_np[b, 0] * 255).astype(np.uint8)
                    if use_crf:
                        refined = apply_dense_crf(img_gray, preds_softmax[b], n_classes=num_classes)
                    else:
                        refined = np.argmax(preds_softmax[b], axis=0)
                    refined_preds.append(refined)

                refined_preds = torch.tensor(np.stack(refined_preds))
                val_iou += compute_iou(refined_preds, masks.cpu(), num_classes=num_classes)

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_ious.append(val_iou)

        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")

        # Early stopping and checkpointing
        if val_iou > best_iou:
            best_iou = val_iou
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(out_dir, "checkpoints", "best.pt"))
            print(f"Saved new best model (IoU={best_iou:.4f}).")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}. Best IoU={best_iou:.4f}.")
                break

    # Save final model
    torch.save(model.state_dict(), os.path.join(out_dir, "checkpoints", "last.pt"))
    print("Training complete.")
    print(f"Best Validation IoU: {best_iou:.4f}")

    return model, (train_losses, val_losses, val_ious)

# endregion

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    args = parser.parse_args()

    train_model(
        root_dir=args.images,
        out_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        size=args.size,
        num_classes=args.num_classes,
        early_stop_patience=args.early_stop_patience,
    )
