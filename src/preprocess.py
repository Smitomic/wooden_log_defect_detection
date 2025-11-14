import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import torch


# region Per-slice cropping
def crop_to_foreground(img, mask, margin: int = 10):
    # Crops tightly around the non-background region of the mask
    coords = np.column_stack(np.where(mask > 0))
    if coords.size == 0:
        # no foreground, return original
        return img, mask

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    y_min = max(y_min - margin, 0)
    x_min = max(x_min - margin, 0)
    y_max = min(y_max + margin, img.shape[0])
    x_max = min(x_max + margin, img.shape[1])

    cropped_img = img[y_min:y_max, x_min:x_max]
    cropped_mask = mask[y_min:y_max, x_min:x_max]

    return cropped_img, cropped_mask


def preprocess_pair(img_path, mask_path, bbox=None, target_size=(256, 256)):
    """
    Per-sample preprocessing:
    - If bbox is None: per-slice crop_to_foreground (old behaviour).
    - If bbox is not None: use fixed bbox (for future log-consistent training).

    Steps:
      1. Load grayscale image and mask.
      2. Crop around wooden log (per-slice).
      3. Resize to target_size.
      4. Normalize image to [0,1].
      5. Map class 9 -> 3.
      6. Convert to torch tensors.
    """
    # 1. load
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    # 2. crop
    if bbox is not None:
        y_min, y_max, x_min, x_max = bbox
        img = img[y_min:y_max, x_min:x_max]
        mask = mask[y_min:y_max, x_min:x_max]
    else:
        img, mask = crop_to_foreground(img, mask)

    # 3. resize
    assert isinstance(target_size, tuple), f"target_size must be a tuple, got {type(target_size)}"
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

    # 4. normalize image
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # (1, H, W)

    # 5. label remapping
    # Map weird label 9 to 3 (Dutina)
    mask[mask == 9] = 3

    # 6. to tensors
    img_tensor = torch.tensor(img, dtype=torch.float32)
    mask_tensor = torch.tensor(mask, dtype=torch.long)

    return img_tensor, mask_tensor
# endregion

# region log-global bbox
def compute_global_bbox(mask_paths, margin=10):
    """
    Compute ONE bounding box covering all masks in a log.
    For future: compute log-global bounding boxes, probably usable for 3D model training e.g. ViT
    """
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


def compute_log_bboxes(root_dir, margin=10):
    # For future: compute log-global bounding boxes, probably usable for 3D model training e.g. ViT
    bbox_dict = {}
    root_dir = os.path.abspath(root_dir)
    log_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

    for log in log_folders:
        log_path = os.path.join(root_dir, log)
        new_format_masks = glob(os.path.join(log_path, "**", "PixelLabelData", "*.png"), recursive=True)
        old_format_masks = glob(
            os.path.join(log_path, "**", "LabelingProject", "GroundTruthProject", "PixelLabelData", "*.png"),
            recursive=True
        )
        mask_paths = list(set(new_format_masks + old_format_masks))
        if not mask_paths:
            print(f"No masks found for log {log}")
            continue

        bbox = compute_global_bbox(mask_paths, margin=margin)
        if bbox:
            bbox_dict[log] = bbox

    print(f"Computed bounding boxes for {len(bbox_dict)} logs.")
    return bbox_dict
# endregion