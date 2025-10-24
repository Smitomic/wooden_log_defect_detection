import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.model import DilatedSegCNN
from src.datamodule import make_dataloaders
from src.preprocess import compute_log_bboxes

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

def evaluate_model(weights_path, root_dir, num_classes=7, batch_size=2, size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    bbox_dict = compute_log_bboxes(root_dir)
    _, val_loader = make_dataloaders(
        root_dir=root_dir, batch_size=batch_size, bbox_dict=bbox_dict, size=(size, size)
    )

    # Model
    model = DilatedSegCNN(in_channels=1, num_classes=num_classes)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    print("Evaluating:", weights_path)
    total_iou, per_class_totals, n_batches = 0.0, np.zeros(num_classes), 0

    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc="Evaluating"):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
            mean_iou, per_class = compute_iou(preds.cpu(), masks.cpu(), num_classes, return_per_class=True)
            total_iou += mean_iou
            per_class_totals += np.nan_to_num(per_class)
            n_batches += 1

    mean_iou = total_iou / n_batches
    per_class_ious = per_class_totals / n_batches

    class_names = ["Background", "Obvod", "Hniloba", "Dutina", "HrcaZ", "HrcaN", "Trhlina"]
    print("\nFinal Per-Class IoUs:")
    for i, name in enumerate(class_names[:num_classes]):
        print(f"{name:<10} | IoU: {per_class_ious[i]:.4f}")
    print(f"\nMean IoU: {mean_iou:.4f}")

    return mean_iou, per_class_ious


def visualize_slices(model, loader, device="cpu", n_slices=5, start_index=0, class_colors=None):
    # Visualizes multiple consecutive slices with their predictions.
    model.eval()
    imgs, masks = [], []
    with torch.no_grad():
        for batch_i, (img, mask) in enumerate(loader):
            imgs.append(img)
            masks.append(mask)
            if len(imgs) * img.shape[0] >= n_slices:
                break

    imgs = torch.cat(imgs, dim=0)
    masks = torch.cat(masks, dim=0)
    imgs, masks = imgs[start_index:start_index + n_slices], masks[start_index:start_index + n_slices]

    imgs = imgs.to(device)
    preds = model(imgs)
    preds = torch.argmax(F.softmax(preds, dim=1), dim=1).cpu()

    imgs = imgs.cpu()

    n = len(imgs)
    plt.figure(figsize=(12, n * 3))

    for i in range(n):
        plt.subplot(n, 3, 3 * i + 1)
        plt.imshow(imgs[i, 0], cmap='gray')
        plt.title(f"Slice {start_index + i}")
        plt.axis("off")

        plt.subplot(n, 3, 3 * i + 2)
        plt.imshow(masks[i], cmap='jet')
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(n, 3, 3 * i + 3)
        plt.imshow(preds[i], cmap='jet')
        plt.title("Prediction")
        plt.axis("off")

    plt.tight_layout()
    plt.show()