from __future__ import annotations

import numpy as np
from scipy.ndimage import label, find_objects
from scipy.spatial import cKDTree
from scipy.spatial.distance import directed_hausdorff
from skimage import measure

from .config import CLASS_NAMES, DEFECT_CLASSES, MAX_BOUNDARY_VOXELS


# region Training-compatible global metric accumulation
def compute_metrics_global(
        preds: np.ndarray,
        targets: np.ndarray,
        num_classes: int = 5,
) -> dict[str, dict]:
    """
    Compute IoU, Dice, Recall, Precision and VS from flattened arrays.

    This replicates the TP/FP/FN accumulation used in the training notebooks
    so that evaluation numbers are directly comparable to training curves.

    Parameters:
    preds, targets : 1D int64 arrays (can be any length)

    Returns:
    dict mapping class_name → {iou, dice, recall, precision, vs}
    """
    results = {}
    for c, name in enumerate(CLASS_NAMES[:num_classes]):
        pred_c = preds == c
        gt_c = targets == c
        tp = int((pred_c & gt_c).sum())
        fp = int((pred_c & ~gt_c).sum())
        fn = int((~pred_c & gt_c).sum())
        pred_total = tp + fp
        gt_total = tp + fn
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
        recall = tp / gt_total if gt_total > 0 else float("nan")
        precision = tp / pred_total if pred_total > 0 else float("nan")
        vs = 1 - abs(pred_total - gt_total) / (pred_total + gt_total + 1e-9)
        results[name] = {
            "iou": iou, "dice": dice,
            "recall": recall, "precision": precision, "vs": vs,
        }
    return results
# endregion

# region Boundary metrics
def compute_boundary_metrics_2d(
        pred_bin: np.ndarray,
        gt_bin: np.ndarray,
) -> tuple[float, float]:
    """
    Hausdorff Distance and ASSD for 2D binary masks.

    Returns (nan, nan) if either mask is empty.
    """
    pred_pts = np.column_stack(np.where(pred_bin))
    gt_pts = np.column_stack(np.where(gt_bin))
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("nan"), float("nan")
    hd = max(directed_hausdorff(pred_pts, gt_pts)[0],
             directed_hausdorff(gt_pts, pred_pts)[0])
    assd = (cKDTree(gt_pts).query(pred_pts)[0].mean() +
            cKDTree(pred_pts).query(gt_pts)[0].mean()) / 2
    return hd, assd


def compute_boundary_metrics_3d(
        pred_bin: np.ndarray,
        gt_bin: np.ndarray,
) -> tuple[float, float]:
    # 3D HD and ASSD on voxel (z, y, x) coordinates.
    pred_pts = np.column_stack(np.where(pred_bin))
    gt_pts = np.column_stack(np.where(gt_bin))
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("nan"), float("nan")
    hd = max(directed_hausdorff(pred_pts, gt_pts)[0],
             directed_hausdorff(gt_pts, pred_pts)[0])
    assd = (cKDTree(gt_pts).query(pred_pts)[0].mean() +
            cKDTree(pred_pts).query(gt_pts)[0].mean()) / 2
    return hd, assd
# endregion

# region Per-image 2D metric set (used in evaluation notebook)
def compute_image_metrics(
        pred_np: np.ndarray,
        gt_np: np.ndarray,
        class_idx: int,
) -> dict[str, float]:
    """
    Full metric set for one class on one 2D image.

    Returns:
    dict with keys: iou, dice, recall, precision, vs, hd, assd
    """
    pred_bin = (pred_np == class_idx)
    gt_bin = (gt_np == class_idx)

    tp = int((pred_bin & gt_bin).sum())
    fp = int((pred_bin & ~gt_bin).sum())
    fn = int((~pred_bin & gt_bin).sum())

    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    vs = 1 - abs(pred_bin.sum() - gt_bin.sum()) / (pred_bin.sum() + gt_bin.sum() + 1e-9)

    hd, assd = (compute_boundary_metrics_2d(pred_bin, gt_bin)
                if gt_bin.sum() > 0 else (float("nan"), float("nan")))

    return {"iou": iou, "dice": dice, "recall": recall,
            "precision": precision, "vs": vs, "hd": hd, "assd": assd}
# endregion

# region 3D volume metric set (used in evaluation_3D notebook)
def safe_compactness(region_mask: np.ndarray) -> float | None:
    # Surface-area-based compactness proxy via marching cubes. Returns None on failure.
    try:
        verts, _, _, _ = measure.marching_cubes(region_mask.astype(np.float32), level=0.5)
        surface = float(verts.shape[0])
        volume = float(np.count_nonzero(region_mask))
        return (surface ** 1.5) / (volume + 1e-6)
    except Exception:
        return None


def compute_volume_metrics_3d(
        pred_vol: np.ndarray,
        gt_vol: np.ndarray,
        class_idx: int,
) -> dict[str, float]:
    """
    Full 3D metric suite: overlap + boundary + volume structure.

    IMPORTANT - annotation mismatch caveat:
    GT was annotated slice-by-slice in 2D without enforcing 3D consistency.
    Overlap metrics measure agreement with 2D annotation decisions.
    Volume metrics measure anatomical plausibility of the 3D structure.
    These two sets measure fundamentally different things - report both.

    Keys returned:
    Overlap : iou, dice, recall, precision, vs
    Boundary: hd_3d, assd_3d  (only for defect classes, size-guarded)
    Volume  : volume_vox, components, continuity, compactness
    """
    pred_bin = (pred_vol == class_idx)
    gt_bin = (gt_vol == class_idx)

    tp = int((pred_bin & gt_bin).sum())
    fp = int((pred_bin & ~gt_bin).sum())
    fn = int((~pred_bin & gt_bin).sum())

    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    vs = 1 - abs(pred_bin.sum() - gt_bin.sum()) / (pred_bin.sum() + gt_bin.sum() + 1e-9)

    # Boundary metrics — defect classes only, size-guarded
    gt_count = int(gt_bin.sum())
    pred_count = int(pred_bin.sum())
    if (class_idx in DEFECT_CLASSES
            and gt_count > 0 and gt_count < MAX_BOUNDARY_VOXELS
            and pred_count > 0 and pred_count < MAX_BOUNDARY_VOXELS):
        hd, assd = compute_boundary_metrics_3d(pred_bin, gt_bin)
    else:
        hd, assd = float("nan"), float("nan")

    # Volume structure metrics
    n_vox = int(pred_bin.sum())
    if n_vox > 0:
        labeled, n_comp = label(pred_bin)
        slices = find_objects(labeled)
        comp_sizes = [int((labeled[s] == (i + 1)).sum()) for i, s in enumerate(slices)]
        compacts = [c for c in
                    [safe_compactness((labeled[s] == (i + 1)))
                     for i, s in enumerate(slices)]
                    if c is not None]
        continuity = max(comp_sizes) / (n_vox + 1e-9) if comp_sizes else float("nan")
        compactness = float(np.mean(compacts)) if compacts else float("nan")
    else:
        n_comp = 0
        continuity = compactness = float("nan")

    return {
        "iou": iou, "dice": dice,
        "recall": recall, "precision": precision,
        "vs": vs, "hd_3d": hd,
        "assd_3d": assd, "volume_vox": n_vox,
        "components": n_comp, "continuity": continuity,
        "compactness": compactness,
    }
# endregion