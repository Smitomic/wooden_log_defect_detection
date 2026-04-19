import os
import re
from collections import defaultdict
from glob import glob

from .config import IMAGES_ROOT, GT_ROOT


# region Pair matching
def mask_stem_to_image_stem(mask_filename: str, prefix_override: dict | None = None) -> str:
    """
    Convert a mask filename to the corresponding image stem.

    Mask filenames look like  ``GT_1_<image_stem>.png``.
    Splitting on ``_`` with maxsplit=2 extracts the stem.
    """
    if prefix_override is None:
        prefix_override = {}
    name = os.path.splitext(mask_filename)[0]
    parts = name.split("_", 2)
    stem = parts[2] if len(parts) == 3 else name
    for wrong, correct in prefix_override.items():
        if stem.startswith(wrong):
            stem = correct + stem[len(wrong):]
            break
    return stem


def build_pairs(
        images_root: str = IMAGES_ROOT,
        gt_root: str = GT_ROOT,
        prefix_override: dict | None = None,
) -> list[tuple[str, str]]:
    """
    Find all (image_path, mask_path) pairs and sort them by the numeric
    page index embedded in the image filename.

    Sorting is critical for 3D volume assembly - glob returns files in
    directory order (alphabetically per subdirectory) which is NOT
    page-number order when logs have non-zero-padded filenames.
    """
    image_paths = sorted(glob(os.path.join(images_root, "**", "*.tif"), recursive=True))
    mask_paths = sorted(glob(
        os.path.join(gt_root, "**", "GroundTruthProject", "PixelLabelData", "*.png"),
        recursive=True,
    ))
    image_lookup = {os.path.splitext(os.path.basename(p))[0]: p for p in image_paths}

    pairs = []
    for mp in mask_paths:
        stem = mask_stem_to_image_stem(os.path.basename(mp), prefix_override)
        if stem in image_lookup:
            pairs.append((image_lookup[stem], mp))

    # Sort by numeric page index in the image filename.
    # e.g. "7372_1_out_page_0609.tif" -> key [7372, 1, 609]
    def _numeric_key(pair: tuple[str, str]) -> list[int]:
        return [int(x) for x in re.findall(r"\d+", os.path.basename(pair[0]))]

    pairs.sort(key=_numeric_key)
    return pairs
# endregion

# region Log ID helpers
def get_log_id(img_path: str, images_root: str = IMAGES_ROOT) -> str | None:
    """
    Return the log directory name that contains *img_path*.

    Walks the path components and returns the first one that matches a
    directory name inside *images_root*.
    """
    parts = img_path.replace("\\", "/").split("/")
    log_names = set(os.listdir(images_root))
    return next((p for p in parts if p in log_names), None)


def group_by_log(
        pairs: list[tuple[str, str]],
        images_root: str = IMAGES_ROOT,
) -> dict[str, list[tuple[str, str]]]:
    """Group (image, mask) pairs by log directory name."""
    log_pairs: dict[str, list] = defaultdict(list)
    for pair in pairs:
        log_id = get_log_id(pair[0], images_root)
        if log_id:
            log_pairs[log_id].append(pair)
    return dict(log_pairs)
# endregion