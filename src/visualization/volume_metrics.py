from skimage import measure
from scipy.ndimage import label, find_objects
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

CLASS_LABELS = {
    1: "Obvod",
    2: "Hniloba",
    3: "Dutina",
    4: "Hrca (Zdravá)",
    5: "Hrca (Nezdravá)",
    6: "Trhlina",
}


def _safe_compactness(region_mask: np.ndarray) -> float | None:
    # Compute a rough compactness measure for a binary region.
    try:
        verts, faces, normals, values = measure.marching_cubes(
            region_mask.astype(np.float32), level=0.5
        )
    except Exception:
        return None

    surface_area = float(verts.shape[0])  # proxy (we don't need exact)
    volume = float(np.count_nonzero(region_mask))
    if volume <= 0:
        return None

    # Heuristic compactness (lower is "rounder", higher is more irregular)
    return (surface_area ** 1.5) / (volume + 1e-6)

def _load_expected_values(path=None):
    # default = expected_values.json next to this file
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "expected_values.json")

    try:
        with open(path, "r") as f:
            data = json.load(f)
        # ensure keys are str
        return {str(k): v for k, v in data.items()}
    except:
        return None


def compute_volume_metrics(volume_3d: np.ndarray, voxel_size=(1, 1, 1)):
    """
    Compute 3D metrics per defect class:

      - volume_cm3
      - number of connected components
      - compactness (mean across components)
      - continuity (largest component size / total voxels)

    Also returns anomaly flags using autocalibrated expectations
    if expected_values.json exists, otherwise falls back to
    static coarse expectations.
    """
    metrics: dict[int, dict] = {}
    anomalies: dict[int, list[str]] = {}

    vx, vy, vz = voxel_size
    voxel_volume = vx * vy * vz

    # load autocalibrated expected values (if exist)
    expected_values = _load_expected_values()

    # static fallback expectations
    EXPECTED_COMPONENTS = {
        4: 8,  # HrcaZ
        5: 8,  # HrcaN
        2: 1,  # Hniloba
        3: 1,  # Dutina
        6: 2,  # Trhlina
    }

    EXPECTED_VOLUME_RANGE = {
        4: (5, 20),
        5: (5, 20),
        2: (10, 60),
        3: (5, 40),
        6: (1, 20),
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

        # autocalibrated expectations
        if expected_values and str(cls) in expected_values:
            if cls != 1:
                continue

            ev = expected_values[str(cls)]

            # volume
            if ev["volume_std"] > 0:
                if abs(vol_cm3 - ev["volume_mean"]) > 2 * ev["volume_std"]:
                    cls_anoms.append(
                        f"Volume unusual (expected ~{ev['volume_mean']:.1f}±{ev['volume_std']:.1f})"
                    )

            # components
            if ev["components_std"] > 0:
                if abs(n_comp - ev["components_mean"]) > 2 * ev["components_std"]:
                    cls_anoms.append(
                        f"Component count unusual (expected ~{ev['components_mean']:.1f})"
                    )

            # continuity
            if continuity is not None and ev["continuity_std"] > 0:
                if abs(continuity - ev["continuity_mean"]) > 2 * ev["continuity_std"]:
                    cls_anoms.append("Abnormal continuity")

            # compactness
            if compactness is not None and ev["compactness_std"] > 0:
                if abs(compactness - ev["compactness_mean"]) > 2 * ev["compactness_std"]:
                    cls_anoms.append("Abnormal compactness")

        else:
            # static fallback expectation (if no json available)_
            if cls in EXPECTED_COMPONENTS:
                expected = EXPECTED_COMPONENTS[cls]
                if abs(n_comp - expected) >= 5:
                    cls_anoms.append(
                        f"Component count unusual (expected ~{expected}, got {n_comp})"
                    )

            if cls in EXPECTED_VOLUME_RANGE:
                low, high = EXPECTED_VOLUME_RANGE[cls]
                if vol_cm3 < low or vol_cm3 > high:
                    cls_anoms.append(
                        f"Volume {round(vol_cm3,2)} cm³ is outside typical range {low}-{high} cm³"
                    )

            if continuity is not None:
                # high continuity + many parts = over-merged except Obvod
                if continuity > 0.9 and n_comp > 1 and cls != 1:
                    cls_anoms.append(
                        "Over-merged region (high continuity with multiple components)"
                    )
                # low continuity + many parts = fragmentation problem
                if continuity < 0.3 and n_comp > 3:
                    cls_anoms.append(
                        "Fragmented region (low continuity with many components)"
                    )

        anomalies[cls] = cls_anoms

    return metrics, anomalies

def autocalibrate_expected_values(
    tiff_folder,
    model_path,
    save_path="expected_values.json",
    use_mrf=True
):
    # lazy import necessary here to avoid circular import
    # lazy import necessary here to avoid circular import
    from src.pipelines.segmentation_pipeline import SegmentationPipeline

    tiffs = sorted([
        os.path.join(tiff_folder, f)
        for f in os.listdir(tiff_folder)
        if f.lower().endswith((".tif", ".tiff"))
    ])

    pipe = SegmentationPipeline(model_type="cnn", use_mrf=use_mrf)

    rows = []

    for tiff_path in tqdm(tiffs):
        _, _, metrics, _ = pipe.run(
            tiff_path=tiff_path,
            model_path=model_path,
            visualize=False,
            return_metrics=True
        )

        for cls, m in metrics.items():
            rows.append({
                "class": cls,
                "volume": m["volume_cm3"],
                "components": m["components"],
                "continuity": m["continuity"],
                "compactness": m["compactness"],
            })

    df = pd.DataFrame(rows)

    # Build expectations dynamically
    expected = {}

    for cls in CLASS_LABELS.keys():
        sub = df[df["class"] == cls]

        if len(sub) == 0:
            continue

        expected[cls] = {
            "volume_mean": float(sub["volume"].mean()),
            "volume_std": float(sub["volume"].std()),

            "components_mean": float(sub["components"].mean()),
            "components_std": float(sub["components"].std()),

            "continuity_mean": float(sub["continuity"].mean()),
            "continuity_std": float(sub["continuity"].std()),

            "compactness_mean": float(sub["compactness"].mean()),
            "compactness_std": float(sub["compactness"].std()),
        }

    with open(save_path, "w") as f:
        json.dump(expected, f, indent=2)

    print(f"Saved autocalibrated expectations to {save_path}")
    return expected

if __name__ == "__main__":
    print(_load_expected_values())