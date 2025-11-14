import os
import argparse
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

from src.model import DilatedSegCNN
from src.datamodule import make_dataloaders
from src.postprocess.crf import apply_dense_crf


# region IoU utilities
def compute_iou(preds, targets, num_classes=7, ignore_index=None, return_per_class=False):
    preds = preds.view(-1)
    targets = targets.view(-1)
    ious = []

    for cls in range(num_classes):
        if ignore_index is not None and cls == ignore_index:
            ious.append(float("nan"))
            continue

        pred_mask = preds == cls
        target_mask = targets == cls

        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()

        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append((intersection / union).item())

    mean_iou = np.nanmean(ious)
    if return_per_class:
        return mean_iou, ious
    else:
        return mean_iou


class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return

        if current_score < self.best_score - self.delta:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
# endregion

# region Evaluation on val set (with optional CRF)
def evaluate(model, criterion, val_loader, device, num_classes=7, use_crf=False):
    model.eval()
    val_loss = 0.0
    val_iou = 0.0
    ious_per_class = [[] for _ in range(num_classes)]

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            if use_crf:
                probs = torch.softmax(outputs, dim=1).cpu().numpy().astype(np.float32)
                imgs_np = images.cpu().numpy()

                batch_ious = []
                for b in range(probs.shape[0]):
                    img = imgs_np[b, 0]  # raw float image
                    img = img - img.min()  # per-image normalization
                    img = img / (img.max() - img.min() + 1e-6)
                    img_gray = (img * 255).astype(np.uint8)

                    probs_b = probs[b]  # shape (C, H, W)

                    # ensure shapes match
                    if probs_b.shape[1:] != img_gray.shape:
                        raise ValueError(f"CRF shape mismatch: probs={probs_b.shape}, img={img_gray.shape}")

                    refined = apply_dense_crf(
                        img_gray,
                        probs_b,
                        n_classes=num_classes,
                    )

                    refined = torch.tensor(refined, dtype=torch.long)

                    iou, per_class = compute_iou(refined, masks[b].cpu(), num_classes, return_per_class=True)
                    batch_ious.append(iou)

                    for cls, cls_iou in enumerate(per_class):
                        if not np.isnan(cls_iou):
                            ious_per_class[cls].append(cls_iou)

                val_iou += sum(batch_ious) / len(batch_ious)
            else:
                preds = torch.argmax(outputs, dim=1)
                batch_ious = []
                for b in range(preds.shape[0]):
                    iou, per_class = compute_iou(preds[b], masks[b], num_classes, return_per_class=True)
                    batch_ious.append(iou)
                    for cls, cls_iou in enumerate(per_class):
                        if not np.isnan(cls_iou):
                            ious_per_class[cls].append(cls_iou)
                val_iou += sum(batch_ious) / len(batch_ious)

    avg_per_class = [np.nanmean(c) if c else float("nan") for c in ious_per_class]
    return val_loss / len(val_loader), val_iou / len(val_loader), avg_per_class
# endregion

# region Training loop
def train_model(
    root_dir,
    out_dir,
    epochs=200,
    batch_size=8,
    lr=1e-3,
    size=256,
    num_classes=7,
    early_stop_patience=15,
    use_crf=False,
    use_augmentation=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on", device)

    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)

    train_loader, val_loader = make_dataloaders(
        root_dir=root_dir,
        batch_size=batch_size,
        size=(size, size),
        use_augmentation=use_augmentation,
    )

    model = DilatedSegCNN(in_channels=1, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_iou = 0.0
    best_epoch = 0
    best_per_class = []
    best_state = None

    early_stopper = EarlyStopping(patience=early_stop_patience, delta=0.001)

    train_losses, val_losses, val_ious = [], [], []

    for epoch in range(epochs):
        model.train()
        running_loss, running_iou = 0.0, 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]"):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            batch_iou = compute_iou(preds, masks, num_classes=num_classes)
            running_iou += batch_iou

        avg_train_loss = running_loss / len(train_loader)
        avg_train_iou = running_iou / len(train_loader)

        val_loss, val_iou, per_class_ious = evaluate(
            model, criterion, val_loader, device,
            num_classes=num_classes, use_crf=use_crf
        )

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        val_ious.append(val_iou)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | Train mIoU: {avg_train_iou:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val mIoU: {val_iou:.4f}"
        )

        # checkpoint best
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_epoch = epoch + 1
            best_per_class = per_class_ious
            best_state = model.state_dict()
            torch.save(best_state, os.path.join(out_dir, "checkpoints", "best.pt"))
            print(f"Saved new best model (IoU={best_val_iou:.4f}).")

        # early stopping
        early_stopper(val_loss)
        if early_stopper.early_stop:
            print(f"Early stopping at epoch {epoch+1}. Best IoU={best_val_iou:.4f}.")
            break

    # save last
    torch.save(model.state_dict(), os.path.join(out_dir, "checkpoints", "last.pt"))

    print(f"\nBest Validation IoU: {best_val_iou:.4f} (epoch {best_epoch})")
    class_names = ["Background", "Obvod", "Hniloba", "Dutina", "HrcaZ", "HrcaN", "Trhlina"]
    print("Best Per-Class IoU:")
    for i, iou in enumerate(best_per_class):
        print(f"{class_names[i]:<10} | IoU: {iou:.4f}")

    return model, (train_losses, val_losses, val_ious)
# endregion

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--early_stop_patience", type=int, default=15)
    parser.add_argument("--use_crf", action="store_true")
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
        use_crf=args.use_crf,
    )
