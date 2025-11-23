import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
from glob import glob

from src.pipelines.segmentation_pipeline import SegmentationPipeline


# CONFIG
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_DIR = os.path.join(BASE, "data", "volume_metric_test")
MODEL_PATH = os.path.join(BASE, "logs", "old_model_mrf", "best.pt")
OUTPUT_JSON = os.path.join(BASE, "src", "visualization", "expected_values.json")    # Auto-calibration file
USE_MRF = True


def accumulate_dict(target, source):
    # Append metric values for dataset-wide averaging.
    for cls, m in source.items():
        if cls not in target:
            target[cls] = {"vol": [], "comp": [], "cont": [], "compactness": []}
        target[cls]["vol"].append(m["volume_cm3"])
        target[cls]["comp"].append(m["components"])
        target[cls]["cont"].append(m["continuity"] or 0.0)
        target[cls]["compactness"].append(m["compactness"] or 0.0)


def compute_stats(values):
    # Mean and std helper with protection against empty lists.
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr))


def main():
    tiff_files = sorted(glob(os.path.join(DATA_DIR, "*.tiff")))
    if not tiff_files:
        print(f"No TIFF files found in {DATA_DIR}")
        return

    print(f"Found {len(tiff_files)} logs for calibration.")
    print("Running segmentation + metrics...")

    pipe = SegmentationPipeline(
        model_type="cnn",
        use_mrf=USE_MRF,
        num_classes=7,
    )

    collected = {}

    for i, tiff_path in enumerate(tiff_files, 1):
        print(f"\n[{i}/{len(tiff_files)}] Processing {os.path.basename(tiff_path)}")

        refined_vol, _, metrics, anomalies = pipe.run(
            tiff_path=tiff_path,
            model_path=MODEL_PATH,
            visualize=False,
            return_metrics=True
        )

        accumulate_dict(collected, metrics)

    print("\nComputing global statistics…")

    expected = {}

    for cls, vals in collected.items():
        vol_m, vol_s = compute_stats(vals["vol"])
        comp_m, comp_s = compute_stats(vals["comp"])
        cont_m, cont_s = compute_stats(vals["cont"])
        cpt_m, cpt_s = compute_stats(vals["compactness"])

        expected[cls] = {
            "volume_mean": vol_m,
            "volume_std": vol_s,
            "components_mean": comp_m,
            "components_std": comp_s,
            "continuity_mean": cont_m,
            "continuity_std": cont_s,
            "compactness_mean": cpt_m,
            "compactness_std": cpt_s,
        }

    # Save
    with open(OUTPUT_JSON, "w") as f:
        json.dump(expected, f, indent=4)

    print(f"\nDONE! Saved autocalibrated expectations → {OUTPUT_JSON}")

    # Print summary for inspection
    print("\nSUMMARY")
    for cls, stats in expected.items():
        print(f"\nClass {cls}:")
        for k, v in stats.items():
            print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()
