from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .config import CLASS_NAMES, NUM_CLASSES
from .mrf import mrf_gibbs_sampling_2d


class EarlyStopping:
    """
    Stop training when the validation loss stops improving.

    Parameters:
    patience : number of epochs to wait after last improvement
    delta    : minimum change in loss to qualify as improvement
    """

    def __init__(self, patience: int = 10, delta: float = 1e-4):
        self.patience   = patience
        self.delta      = delta
        self.best_score = None
        self.counter    = 0
        self.early_stop = False

    def __call__(self, current_loss: float) -> None:
        if self.best_score is None:
            self.best_score = current_loss
            return

        if current_loss < self.best_score - self.delta:
            self.best_score = current_loss
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# Evaluation loop
def evaluate(
    model,
    loader:      torch.utils.data.DataLoader,
    criterion,
    device:      torch.device,
    use_mrf:     bool  = False,
    mrf_beta:    float = 0.8,
    num_classes: int   = NUM_CLASSES,
) -> tuple[float, float, dict]:
    """
    Run one evaluation pass over *loader*.

    Returns:
    val_loss   : mean loss over all batches
    mean_iou   : macro-averaged IoU across all classes (NaN-safe)
    per_class  : dict mapping class_name → {iou, dice, recall, precision, vs}
    """
    model.eval()
    total_loss = 0.0

    tp_sum = np.zeros(num_classes, dtype=np.int64)
    fp_sum = np.zeros(num_classes, dtype=np.int64)
    fn_sum = np.zeros(num_classes, dtype=np.int64)

    with torch.no_grad():
        for imgs_b, masks_b in loader:
            imgs_b  = imgs_b.to(device)
            masks_b = masks_b.to(device)

            logits = model(imgs_b)
            loss   = criterion(logits, masks_b)
            total_loss += loss.item()

            probs = F.softmax(logits, dim=1)

            if use_mrf:
                preds_list = [
                    mrf_gibbs_sampling_2d(probs[b], beta=mrf_beta)
                    for b in range(len(probs))
                ]
                preds = torch.stack(preds_list).cpu().numpy()
            else:
                preds = torch.argmax(probs, dim=1).cpu().numpy()

            masks_np = masks_b.cpu().numpy()

            for pred, gt in zip(preds, masks_np):
                for c in range(num_classes):
                    p = pred == c; g = gt == c
                    tp_sum[c] += int(( p &  g).sum())
                    fp_sum[c] += int(( p & ~g).sum())
                    fn_sum[c] += int((~p &  g).sum())

    # Global IoU (matches training accumulation)
    per_class: dict[str, dict] = {}
    ious = []
    for c, name in enumerate(CLASS_NAMES[:num_classes]):
        tp, fp, fn    = int(tp_sum[c]), int(fp_sum[c]), int(fn_sum[c])
        iou           = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
        dice          = 2*tp / (2*tp+fp+fn) if (2*tp+fp+fn) > 0 else float("nan")
        recall        = tp / (tp + fn)      if (tp + fn) > 0 else float("nan")
        precision     = tp / (tp + fp)      if (tp + fp) > 0 else float("nan")
        vs            = 1 - abs((tp+fp) - (tp+fn)) / ((tp+fp) + (tp+fn) + 1e-9)
        per_class[name] = {
            "iou": iou, "dice": dice,
            "recall": recall, "precision": precision, "vs": vs,
        }
        ious.append(iou)

    mean_iou = float(np.nanmean(ious))
    return total_loss / max(len(loader), 1), mean_iou, per_class