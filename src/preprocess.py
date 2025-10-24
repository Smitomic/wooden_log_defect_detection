import os
import cv2
import numpy as np
from glob import glob
from pathlib import Path
from tqdm import tqdm
import torch

# region Bounding box & cropping utilities

def crop_to_foreground(img, mask, margin=10):
    # Crop tightly around mask > 0 region.
    coords = np.column_stack(np.where(mask > 0))
    if coords.size == 0:
        return img, mask
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    y_min = max(y_min - margin, 0)
    x_min = max(x_min - margin, 0)
    y_max = min(y_max + margin, img.shape[0])
    x_max = min(x_max + margin, img.shape[1])
    return img[y_min:y_max, x_min:x_max], mask[y_min:y_max, x_min:x_max]


def compute_global_bbox(mask_paths, margin=10):
    # Compute one bounding box covering all masks in a log (in order to preserve 3D alignment of the wooden log).
    y_mins, x_mins, y_maxs, x_maxs = [], [], [], []
    for mpath in tqdm(mask_paths, desc="Computing global bbox"):
        mask = cv2.imread(str(mpath), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        coords = np.column_stack(np.where(mask > 0))
        if coords.size == 0:
            continue
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        y_mins.append(y_min)
        x_mins.append(x_min)
        y_maxs.append(y_max)
        x_maxs.append(x_max)
    if not y_mins:
        return None
    y_min = max(min(y_mins) - margin, 0)
    x_min = max(min(x_mins) - margin, 0)
    y_max = max(y_maxs) + margin
    x_max = max(x_maxs) + margin
    return y_min, y_max, x_min, x_max


def crop_with_bbox(img, mask, bbox):
    # Apply fixed bbox cropping (used for log-consistent crops).
    y_min, y_max, x_min, x_max = bbox
    return img[y_min:y_max, x_min:x_max], mask[y_min:y_max, x_min:x_max]

# endregion

# region Preprocessing pairs

def preprocess_pair(img_path, mask_path, bbox=None, target_size=(256, 256)):
    """
    Preprocess image + mask:
      1. Optional fixed crop (bbox)
      2. Resize to target_size
      3. Normalize + convert to torch tensors
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

    if bbox is not None:
        img, mask = crop_with_bbox(img, mask, bbox)
    else:
        img, mask = crop_to_foreground(img, mask)

    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # (1, H, W)

    mask[mask == 9] = 3  # Map defect class 9 → 3

    img_tensor = torch.tensor(img, dtype=torch.float32)
    mask_tensor = torch.tensor(mask, dtype=torch.long)
    return img_tensor, mask_tensor


def compute_log_bboxes(root_dir, margin=10):
    # Compute bounding boxes for each log folder.
    bbox_dict = {}
    root_dir = os.path.abspath(root_dir)
    log_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

    for log in log_folders:
        log_path = os.path.join(root_dir, log)

        # Collect masks from both possible locations
        new_format_masks = glob(os.path.join(log_path, "**", "PixelLabelData", "*.png"), recursive=True)
        old_format_masks = glob(os.path.join(log_path, "**", "LabelingProject", "GroundTruthProject", "PixelLabelData", "*.png"), recursive=True)

        mask_paths = new_format_masks + old_format_masks
        mask_paths = list(set(mask_paths))  # remove duplicates

        if not mask_paths:
            print(f"No masks found for log {log}")
            continue

        bbox = compute_global_bbox(mask_paths, margin=margin)
        if bbox:
            bbox_dict[log] = bbox

    print(f"Computed bounding boxes for {len(bbox_dict)} logs.")
    return bbox_dict

# endregion