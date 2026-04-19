"""
Preprocessing: crop, remap, z-score normalise, cache to .npy.

Two cache strategies:
  build_cache        - per-slice crop_to_foreground (used in training and 2D eval)
  build_cache_global - uniform per-log bounding box (required for 3D eval and
                       3D MRF so all slices share the same spatial reference frame)
"""

import os
import shutil
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

from .config import CLASS_REMAP, PATCH_SIZE
from .data import get_log_id


# region Crop / remap
def crop_to_foreground(
        img: np.ndarray,
        mask: np.ndarray,
        margin: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    # Tight crop around the non-background region of *mask*.
    coords = np.column_stack(np.where(mask > 0))
    if coords.size == 0:
        return img, mask
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    y_min = max(y_min - margin, 0);
    x_min = max(x_min - margin, 0)
    y_max = min(y_max + margin, img.shape[0])
    x_max = min(x_max + margin, img.shape[1])
    return img[y_min:y_max, x_min:x_max], mask[y_min:y_max, x_min:x_max]


def remap_mask(mask: np.ndarray, class_remap: dict | None = None) -> np.ndarray:
    """
    Re-map raw GT pixel values to contiguous model class indices.

    Default mapping (from config.CLASS_REMAP):
        raw 1 -> 0  Background
        raw 2 -> 1  Bark
        raw 3 -> 2  Wood
        raw 4 -> 3  Knot
        raw 5 -> 4  Crack
    """
    if class_remap is None:
        class_remap = CLASS_REMAP
    out = np.zeros_like(mask, dtype=np.int64)
    for raw, idx in class_remap.items():
        out[mask == raw] = idx
    return out
# endregion

# region Single-slice load + preprocess
def load_and_preprocess(
        img_path: str,
        mask_path: str,
        patch_size: int = PATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one (image, mask) pair, apply per-slice crop, resize, and
    **z-score normalise** the image.

    Z-score normalisation is mandatory: the trained models were trained with
    ``(img - mean) / std`` - NOT ``img / 255``.  Using /255 here will
    produce silently wrong predictions.

    Returns:
    img  : float32 ndarray, shape (1, H, W), z-score normalised
    mask : int64  ndarray, shape (H, W),    class indices 0–4
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if img is None: raise FileNotFoundError(img_path)
    if mask is None: raise FileNotFoundError(mask_path)

    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    img, mask = crop_to_foreground(img, mask)
    img = cv2.resize(img, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)

    # Z-score normalise (per-image, NOT per-dataset)
    img = img.astype(np.float32)
    img = (img - img.mean()) / (img.std() + 1e-6)

    return (
        np.expand_dims(img, 0).astype(np.float32),
        remap_mask(mask).astype(np.int64),
    )
# endregion

# region Per-slice cache  (training + 2D evaluation)
def build_cache(
        pairs: list[tuple[str, str]],
        cache_dir: str,
        force: bool = False,
        patch_size: int = PATCH_SIZE,
) -> list[tuple[str, str]]:
    """
    Pre-process all pairs and save as .npy files.
    Returns a list of (img_npy_path, mask_npy_path) tuples.

    Uses per-slice ``crop_to_foreground`` - appropriate for 2D evaluation
    and training.  Not to be used for 3D volume assembly (need to use
    ``build_cache_global`` instead).
    """
    if force and os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    cached = []
    for img_path, mask_path in tqdm(pairs, desc=f"Caching → {os.path.basename(cache_dir)}"):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        img_file = os.path.join(cache_dir, stem + "_img.npy")
        mask_file = os.path.join(cache_dir, stem + "_mask.npy")
        if not os.path.exists(img_file):
            img, mask = load_and_preprocess(img_path, mask_path, patch_size)
            np.save(img_file, img)
            np.save(mask_file, mask)
        cached.append((img_file, mask_file))

    # Quick sanity check
    test_img = np.load(cached[0][0])
    test_mask = np.load(cached[0][1])
    print(f"  Cache OK — img: {test_img.shape} {test_img.dtype} | "
          f"mask: {test_mask.shape} unique: {np.unique(test_mask).tolist()}")
    return cached


# region Global-bbox cache  (3D evaluation - required for volume assembly)
def _compute_global_bbox(
        mask_paths: list[str],
        margin: int = 10,
) -> tuple[int, int, int, int]:
    """
    Per-slice ``crop_to_foreground`` shifts the crop window as the log bends,
    producing laterally misaligned slices when stacked into a 3D volume.
    This function computes ONE bbox so every slice gets the same crop window,
    preserving spatial coherence across the depth axis.

    Returns  (y0, x0, y1, x1)
    """
    y_min_g = x_min_g = float("inf")
    y_max_g = x_max_g = -float("inf")
    h_ref = w_ref = None

    for mp in mask_paths:
        mask = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        if h_ref is None:
            h_ref, w_ref = mask.shape[:2]
        coords = np.column_stack(np.where(mask > 0))
        if coords.size == 0:
            continue
        y_min_g = min(y_min_g, coords[:, 0].min())
        x_min_g = min(x_min_g, coords[:, 1].min())
        y_max_g = max(y_max_g, coords[:, 0].max())
        x_max_g = max(x_max_g, coords[:, 1].max())

    if y_min_g == float("inf"):  # all-empty log - fall back to full frame
        return 0, 0, h_ref or 512, w_ref or 512

    return (
        max(int(y_min_g) - margin, 0),
        max(int(x_min_g) - margin, 0),
        min(int(y_max_g) + margin, h_ref),
        min(int(x_max_g) + margin, w_ref),
    )


def _load_slice_global_bbox(
        img_path: str,
        mask_path: str,
        bbox: tuple[int, int, int, int],
        patch_size: int = PATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    # Load one slice using a pre-computed fixed bbox (no per-slice shift).
    y0, x0, y1, x1 = bbox
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    img = img[y0:y1, x0:x1]
    mask = mask[y0:y1, x0:x1]
    img = cv2.resize(img, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)
    img = img.astype(np.float32)
    img = (img - img.mean()) / (img.std() + 1e-6)
    return np.expand_dims(img, 0).astype(np.float32), remap_mask(mask).astype(np.int64)


def build_cache_global(
        pairs: list[tuple[str, str]],
        cache_dir: str,
        images_root: str,
        force: bool = False,
        patch_size: int = PATCH_SIZE,
) -> list[tuple[str, str]]:
    """
    Cache test slices using a **per-log global bounding box**.

    All slices of a log share the same spatial reference frame, which is
    required for:
      - correct 3D connectivity metrics (components, continuity)
      - 3D MRF smoothing (voxels must be spatially aligned across z)
      - 3D Plotly visualisations (cracks must not jump laterally)

    Replaces ``build_cache`` for any code that stacks slices into a volume.
    """
    if force and os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    # Group by log to compute one bbox per log
    log_to_pairs: dict[str, list] = defaultdict(list)
    for img_path, mask_path in pairs:
        log_id = get_log_id(img_path, images_root)
        log_to_pairs[log_id].append((img_path, mask_path))

    log_bboxes: dict[str, tuple] = {}
    for log_id, log_pairs in log_to_pairs.items():
        mask_paths_log = [mp for _, mp in log_pairs]
        log_bboxes[log_id] = _compute_global_bbox(mask_paths_log)
        y0, x0, y1, x1 = log_bboxes[log_id]
        print(f"  {log_id}: global bbox  y:[{y0},{y1}]  x:[{x0},{x1}]")

    cached = []
    for img_path, mask_path in tqdm(pairs, desc="Caching (global bbox)"):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        img_file = os.path.join(cache_dir, stem + "_img.npy")
        mask_file = os.path.join(cache_dir, stem + "_mask.npy")
        if not os.path.exists(img_file):
            log_id = get_log_id(img_path, images_root)
            bbox = log_bboxes[log_id]
            img_arr, mask_arr = _load_slice_global_bbox(img_path, mask_path, bbox, patch_size)
            np.save(img_file, img_arr)
            np.save(mask_file, mask_arr)
        cached.append((img_file, mask_file))

    print(f"  Cache ready: {len(cached)} pairs (global bbox)")
    return cached
# endregion