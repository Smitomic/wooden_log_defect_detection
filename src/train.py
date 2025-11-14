import os
import argparse
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

from src.model import DilatedSegCNN
from src.datamodule import make_dataloaders
from src.postprocess.crf import apply_dense_crf
from src.postprocess.mrf import mrf_gibbs_sampling


# region IoU utilities
def compute_iou_global(preds, targets, num_classes=7):
    preds = preds.view(-1)
    targets = targets.view(-1)

    ious = []
    for cls in range(num_classes):
        pred_mask = preds == cls
        target_mask = targets == cls

        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()

        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(intersection / union)

    mean_iou = np.nanmean(ious)
    return mean_iou, ious


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

        # stop if loss increases
        if current_score < self.best_score - self.delta:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
# endregion

# region Evaluation on val set
def evaluate(model, criterion, val_loader, device, num_classes=7, use_crf=False, use_mrf=False):
    model.eval()
    val_loss = 0.0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)

            # MRF
            if use_mrf:
                preds_batch = []
                probs_np = probs.cpu().numpy()
                for b in range(len(probs_np)):
                    pmap = torch.tensor(probs_np[b], dtype=torch.float32)
                    refined = mrf_gibbs_sampling(pmap)
                    preds_batch.append(refined)
                preds = torch.tensor(np.stack(preds_batch), dtype=torch.long)

            # CRF
            elif use_crf:
                preds_batch = []
                probs_np = probs.cpu().numpy()
                imgs_np = images.cpu().numpy()

                for b in range(len(probs_np)):
                    img = imgs_np[b, 0]
                    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-6)
                    img_gray = (img_norm * 255).astype(np.uint8)

                    refined = apply_dense_crf(
                        img_gray,
                        probs_np[b],
                        n_classes=num_classes,
                    )
                    preds_batch.append(refined)

                preds = torch.tensor(np.stack(preds_batch), dtype=torch.long)

            # RAW
            else:
                preds = torch.argmax(outputs, dim=1).cpu()

            all_preds.append(preds.reshape(-1))
            all_targets.append(masks.cpu().reshape(-1))

    # global evaluation
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    mean_iou, per_class_ious = compute_iou_global(all_preds, all_targets, num_classes)

    return val_loss / len(val_loader), mean_iou, per_class_ious
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
    use_mrf=False,
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

    early_stopper = EarlyStopping(patience=early_stop_patience, delta=0.0001)

    train_losses, val_losses, val_ious = [], [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        train_preds, train_targets = [], []

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]"):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu()
            train_preds.append(preds.reshape(-1))
            train_targets.append(masks.cpu().reshape(-1))

        train_preds = torch.cat(train_preds)
        train_targets = torch.cat(train_targets)

        train_iou, _ = compute_iou_global(train_preds, train_targets, num_classes)

        val_loss, val_iou, per_class_ious = evaluate(
            model, criterion, val_loader, device,
            num_classes=num_classes, use_crf=use_crf, use_mrf=use_mrf
        )

        train_losses.append(running_loss / len(train_loader))
        val_losses.append(val_loss)
        val_ious.append(val_iou)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {running_loss/len(train_loader):.4f} | Train mIoU: {train_iou:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val mIoU: {val_iou:.4f}"
        )

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_epoch = epoch + 1
            best_per_class = per_class_ious
            best_state = model.state_dict()
            torch.save(best_state, os.path.join(out_dir, "checkpoints", "best.pt"))
            print(f"Saved new best model (IoU={best_val_iou:.4f}).")

        early_stopper(val_loss)
        if early_stopper.early_stop:
            print(f"Early stopping at epoch {epoch+1}. Best IoU={best_val_iou:.4f}.")
            break

    #torch.save(model.state_dict(), os.path.join(out_dir, "checkpoints", "last.pt"))

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
    parser.add_argument("--use_mrf", action="store_true")
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
        use_mrf=args.use_mrf,
    )
