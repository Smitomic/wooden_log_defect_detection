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


def _load_expected_values(path: str | None = None) -> dict | None:
    if path is None:
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
    ev_path:      str | None = None,
    class_scheme: str = "new",
) -> tuple[dict, dict]:
    """
    Compute 3D quality metrics and anomaly flags for each non-background class.

    volume_3d    : (D, H, W) int array with labels
    voxel_size   : physical voxel dimensions in mm
    ev_path      : optional path to expected_values.json
    class_scheme : "new" (5-class) or "old" (7-class)
    """
    labels_map = CLASS_LABELS if class_scheme == "new" else _OLD_CLASS_LABELS
    exp_comp   = _NEW_EXPECTED_COMPONENTS if class_scheme == "new" else _OLD_EXPECTED_COMPONENTS
    exp_vol    = _NEW_EXPECTED_VOLUME     if class_scheme == "new" else _OLD_EXPECTED_VOLUME
    metrics:   dict[int, dict]       = {}
    anomalies: dict[int, list[str]]  = {}

    vx, vy, vz   = voxel_size
    voxel_volume = vx * vy * vz

    expected = _load_expected_values(ev_path)

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

        metrics[cls] = {
            "volume_vox":  n_vox,
            "volume_cm3":  vol_cm3,
            "components":  n_comp,
            "compactness": compactness,
            "continuity":  continuity,
        }

        # Anomaly detection
        cls_anoms: list[str] = []
        use_auto = expected and str(cls) in expected

        if use_auto:
            ev = expected[str(cls)]
            if ev.get("volume_std", 0) > 0:
                if abs(vol_cm3 - ev["volume_mean"]) > 2 * ev["volume_std"]:
                    cls_anoms.append(
                        f"Volume unusual (expected ~{ev['volume_mean']:.1f}±{ev['volume_std']:.1f})"
                    )
            if ev.get("components_std", 0) > 0:
                if abs(n_comp - ev["components_mean"]) > 2 * ev["components_std"]:
                    cls_anoms.append(
                        f"Component count unusual (expected ~{ev['components_mean']:.1f})"
                    )
            if continuity is not None and ev.get("continuity_std", 0) > 0:
                if abs(continuity - ev["continuity_mean"]) > 2 * ev["continuity_std"]:
                    cls_anoms.append("Abnormal continuity")
            if compactness is not None and ev.get("compactness_std", 0) > 0:
                if abs(compactness - ev["compactness_mean"]) > 2 * ev["compactness_std"]:
                    cls_anoms.append("Abnormal compactness")
        else:
            if cls in exp_comp:
                if abs(n_comp - exp_comp[cls]) >= 5:
                    cls_anoms.append(
                        f"Components unusual (expected ~{exp_comp[cls]}, got {n_comp})"
                    )
            if cls in exp_vol:
                lo, hi = exp_vol[cls]
                if vol_cm3 < lo or vol_cm3 > hi:
                    cls_anoms.append(f"Volume {vol_cm3} cm³ outside typical range {lo}–{hi} cm³")
            if continuity is not None and continuity < 0.3 and n_comp > 3:
                cls_anoms.append("Fragmented (low continuity, many components)")

        anomalies[cls] = cls_anoms

    return metrics, anomalies
# endregion