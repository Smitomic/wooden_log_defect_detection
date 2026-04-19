import json
import os

import numpy as np
from scipy.ndimage import label, find_objects
from skimage import measure

# 5-class scheme (new models)
CLASS_LABELS = {          # exported for app.py metrics panel
    1: "Bark",
    2: "Wood",
    3: "Knot",
    4: "Crack",
}

# 7-class scheme (old models)
_OLD_CLASS_LABELS = {
    1: "Obvod",
    2: "Hniloba",
    3: "Dutina",
    4: "Hrca (Zdravá)",
    5: "Hrca (Nezdravá)",
    6: "Trhlina",
}

# Static fallback expectations per scheme
_NEW_EXPECTED_COMPONENTS = {3: 5,   4: 10}
_NEW_EXPECTED_VOLUME     = {1: (500, 5000), 2: (5000, 50000), 3: (10, 500), 4: (1, 200)}

_OLD_EXPECTED_COMPONENTS = {4: 8, 5: 8, 2: 1, 3: 1, 6: 2}
_OLD_EXPECTED_VOLUME     = {4: (5, 20), 5: (5, 20), 2: (10, 60), 3: (5, 40), 6: (1, 20)}


# region Helpers
def _safe_compactness(region_mask: np.ndarray) -> float | None:
    try:
        verts, _, _, _ = measure.marching_cubes(region_mask.astype(np.float32), level=0.5)
        surface = float(verts.shape[0])
        volume  = float(np.count_nonzero(region_mask))
        return (surface ** 1.5) / (volume + 1e-6)
    except Exception:
        return None


def _load_expected_values(path: str | None = None, class_scheme: str = "new") -> dict | None:
    if path is None:
        filename = "expected_values_new.json" if class_scheme == "new" else "expected_values_old.json"
        path = os.path.join(os.path.dirname(__file__), filename)
        # Fall back to legacy expected_values.json for old scheme if new file absent
        if not os.path.exists(path) and class_scheme == "old":
            path = os.path.join(os.path.dirname(__file__), "expected_values.json")
    try:
        with open(path) as f:
            data = json.load(f)
        return {str(k): v for k, v in data.items()}
    except Exception:
        return None
# endregion

# region Main metric function
def compute_volume_metrics(
    volume_3d:    np.ndarray,
    voxel_size:   tuple = (1, 1, 1),
    class_scheme: str = "new",
) -> tuple[dict, dict]:
    """
    Compute 3D structural metrics for each non-background class.

    Anomaly detection has been removed - the metrics are presented as-is
    for the user to interpret. Flagging model predictions as anomalous based
    on statistics derived from those same predictions is circular.

    Parameters:
    volume_3d    : (D, H, W) int array with labels
    voxel_size   : physical voxel dimensions in mm
    class_scheme : "new" (5-class) or "old" (7-class)

    Returns:
    metrics   : dict[cls_int → {volume_vox, volume_cm3, components, compactness, continuity}]
    anomalies : dict[cls_int → list]  - always empty dicts, kept for API compatibility
    """
    labels_map = CLASS_LABELS if class_scheme == "new" else _OLD_CLASS_LABELS

    metrics:   dict[int, dict]      = {}
    anomalies: dict[int, list]      = {}

    vx, vy, vz   = voxel_size
    voxel_volume = vx * vy * vz

    for cls, name in labels_map.items():
        mask  = volume_3d == cls
        n_vox = int(np.count_nonzero(mask))

        if n_vox == 0:
            metrics[cls]   = {"volume_vox": 0, "volume_cm3": 0.0,
                               "components": 0, "compactness": None, "continuity": None}
            anomalies[cls] = []
            continue

        vol_cm3 = round((n_vox * voxel_volume) / 1000.0, 2)

        labeled, n_comp = label(mask)
        slices          = find_objects(labeled)
        comp_sizes      = [int((labeled[s] == (i + 1)).sum()) for i, s in enumerate(slices)]
        compacts        = [c for c in [_safe_compactness((labeled[s] == (i + 1)))
                                       for i, s in enumerate(slices)] if c is not None]

        compactness = float(np.mean(compacts)) if compacts else None
        continuity  = round(max(comp_sizes) / (n_vox + 1e-9), 4) if comp_sizes else None

        metrics[cls]   = {
            "volume_vox":  n_vox,
            "volume_cm3":  vol_cm3,
            "components":  n_comp,
            "compactness": compactness,
            "continuity":  continuity,
        }
        anomalies[cls] = []

    return metrics, anomalies
# endregion