import numpy as np
from skimage import measure
from scipy.ndimage import label, find_objects

CLASS_LABELS = {
    1: "Obvod",
    2: "Hniloba",
    3: "Dutina",
    4: "Hrca (Zdravá)",
    5: "Hrca (Nezdravá)",
    6: "Trhlina",
}


def _safe_compactness(region_mask: np.ndarray) -> float | None:
    try:
        verts, faces, normals, values = measure.marching_cubes(
            region_mask.astype(np.float32), level=0.5
        )
    except Exception:
        return None

    surface_area = float(verts.shape[0])
    volume = float(np.count_nonzero(region_mask))
    if volume <= 0:
        return None

    return (surface_area ** 1.5) / (volume + 1e-6)


def compute_volume_metrics(volume_3d: np.ndarray, voxel_size=(1, 1, 1)):
    """
    Compute 3D metrics per defect class:

      - volume_cm3
      - number of connected components
      - compactness (mean across components)
      - continuity (largest component size / total voxels)

    Also returns anomaly flags — tailored per defect type.
    """
    metrics: dict[int, dict] = {}
    anomalies: dict[int, list[str]] = {}

    vx, vy, vz = voxel_size
    voxel_volume = vx * vy * vz  # mm³

    # TODO: Realistic expected volume ranges (cm³)
    EXPECTED_VOLUME_RANGE = {
        4: (4, 25),   # HrcaZ
        5: (4, 30),   # HrcaN
        2: (10, 60),  # Hniloba
        3: (5, 40),   # Dutina
        6: (1, 20),   # Trhlina
    }

    # Where fragmentation / merging checks make sense
    FRAGMENTATION_ALLOWED = {4, 5}   # knots can naturally fragment
    MERGING_ALLOWED = {4, 5}         # knots can merge if branches overlap

    # Connected components expectations (only for some defects)
    EXPECTED_COMPONENTS = {
        2: 1,  # Hniloba
        3: 1,  # Dutina
        6: 2,  # Cracks
    }

    for cls, name in CLASS_LABELS.items():
        cls_mask = volume_3d == cls
        n_vox = int(np.count_nonzero(cls_mask))

        if n_vox == 0:
            metrics[cls] = {
                "volume_cm3": 0.0,
                "components": 0,
                "compactness": None,
                "continuity": None,
            }
            anomalies[cls] = []
            continue

        # Basic volume
        vol_cm3 = (n_vox * voxel_volume) / 1000.0

        # Connected components
        labeled, n_comp = label(cls_mask)
        slices = find_objects(labeled)

        compacts = []
        comp_sizes = []
        for idx, slc in enumerate(slices, start=1):
            region = labeled[slc] == idx
            comp_sizes.append(int(np.count_nonzero(region)))
            c = _safe_compactness(region)
            if c is not None:
                compacts.append(c)

        compactness = float(np.mean(compacts)) if compacts else None
        continuity = max(comp_sizes) / (n_vox + 1e-6) if comp_sizes else None

        metrics[cls] = {
            "volume_cm3": round(vol_cm3, 2),
            "components": int(n_comp),
            "compactness": compactness,
            "continuity": round(continuity, 3) if continuity is not None else None,
        }

        # Anomaly detection logic
        cls_anoms: list[str] = []

        # 1. Unexpected component count
        if cls in EXPECTED_COMPONENTS:
            expected = EXPECTED_COMPONENTS[cls]
            if abs(n_comp - expected) >= 2:
                cls_anoms.append(
                    f"Unusual number of components (expected ~{expected}, got {n_comp})"
                )

        # 2. Volume anomalies
        if cls in EXPECTED_VOLUME_RANGE:
            low, high = EXPECTED_VOLUME_RANGE[cls]
            if vol_cm3 < low or vol_cm3 > high:
                cls_anoms.append(
                    f"Volume {round(vol_cm3,2)} cm³ outside expected {low}-{high} cm³"
                )

        # 3. Merging anomalies
        if continuity is not None and cls not in MERGING_ALLOWED and cls != 1:
            if continuity > 0.9 and n_comp > 1:
                cls_anoms.append(
                    "Over-merged region (high continuity with several components)"
                )

        # 4. Fragmentation anomalies
        if continuity is not None and cls not in FRAGMENTATION_ALLOWED:
            if continuity < 0.3 and n_comp > 3:
                cls_anoms.append(
                    "Fragmented region (low continuity + many pieces)"
                )

        anomalies[cls] = cls_anoms

    return metrics, anomalies
