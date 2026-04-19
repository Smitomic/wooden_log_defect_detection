import os

import numpy as np
import torch
from scipy.ndimage import binary_closing

from src.postprocess.segment_log import segment_tiff_volume
from src.postprocess.mrf import mrf_gibbs_sampling
from src.postprocess.mrf3d import mrf_gibbs_sampling_3d
from src.visualization.mesh_viewer import show_volume
from src.visualization.volume_metrics import compute_volume_metrics

# Registry of supported models - mirrors wood_utils.config.MODELS
# Paths are relative to the project root (two levels above this file).
_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

MODEL_REGISTRY = {
    "DilatedCNN+MRF log-split": {
        "path":             os.path.join(_BASE, "logs", "20_trees", "checkpoints", "best.pt"),
        "type":             "dilated",
        "num_classes":      5,
        "normalize":        "zscore",
        "class_scheme":     "new",
        "crack_class":      4,
        "apply_2d_mrf":     True,
        "mrf_beta":         0.8,
        "mrf3d_beta":       0.3,
        "mrf3d_iterations": 3,
        "skip_classes":     [4],
    },
    "DilatedCNN+MRF random-split": {
        "path":             os.path.join(_BASE, "logs", "20_trees_random-split", "checkpoints", "best.pt"),
        "type":             "dilated",
        "num_classes":      5,
        "normalize":        "zscore",
        "class_scheme":     "new",
        "crack_class":      4,
        "apply_2d_mrf":     True,
        "mrf_beta":         0.8,
        "mrf3d_beta":       0.3,
        "mrf3d_iterations": 3,
        "skip_classes":     [4],
    },
    "UNet++ log-split": {
        "path":             os.path.join(_BASE, "logs", "20_trees_UNET++", "checkpoints", "best.pt"),
        "type":             "unetpp",
        "num_classes":      5,
        "normalize":        "zscore",
        "class_scheme":     "new",
        "crack_class":      4,
        "apply_2d_mrf":     False,
        "mrf_beta":         0.0,
        "mrf3d_beta":       0.3,
        "mrf3d_iterations": 3,
        "skip_classes":     [4],
    },
    "UNet++ random-split": {
        "path":             os.path.join(_BASE, "logs", "20_trees_UNET++_random-split", "checkpoints", "best.pt"),
        "type":             "unetpp",
        "num_classes":      5,
        "normalize":        "zscore",
        "class_scheme":     "new",
        "crack_class":      4,
        "apply_2d_mrf":     False,
        "mrf_beta":         0.0,
        "mrf3d_beta":       0.3,
        "mrf3d_iterations": 3,
        "skip_classes":     [4],
    },
    # Old 7-class models (/255 normalisation)
    "Old model (no MRF)": {
        "path":             os.path.join(_BASE, "logs", "old_model", "checkpoints", "best.pt"),
        "type":             "dilated",
        "num_classes":      7,
        "normalize":        "div255",
        "class_scheme":     "old",
        "crack_class":      6,
        "apply_2d_mrf":     False,
        "mrf_beta":         0.0,
        "mrf3d_beta":       0.8,
        "mrf3d_iterations": 3,
        "skip_classes":     [],
    },
    "Old model (with MRF)": {
        "path":             os.path.join(_BASE, "logs", "old_model_mrf", "checkpoints", "best.pt"),
        "type":             "dilated",
        "num_classes":      7,
        "normalize":        "div255",
        "class_scheme":     "old",
        "crack_class":      6,
        "apply_2d_mrf":     True,
        "mrf_beta":         0.8,
        "mrf3d_beta":       0.8,
        "mrf3d_iterations": 3,
        "skip_classes":     [],
    },
}


# region Z-axis morphological closing

def apply_z_closing(pred_vol: np.ndarray, z_extent: int = 3, crack_class: int = 4) -> np.ndarray:
    """
    Bridge isolated crack fragments across consecutive slices by applying
    binary_closing with a (z_extent, 1, 1) structuring element - depth-only,
    so crack width is never expanded laterally.

    Only NEW voxels are written, existing non-crack labels are never overridden.

    pred_vol    : (D, H, W) int64 label volume
    z_extent    : length of the structuring element along the depth axis.
                  Must be odd. Bridges up to (z_extent - 1) missing slices.
                  Recommended values: 3, 5, 7.
    crack_class : class index to close. 4 for new 5-class models (Crack),
                  6 for old 7-class models (Trhlina).
    """
    struct     = np.zeros((z_extent, 1, 1), dtype=bool)
    struct[:, 0, 0] = True

    crack_mask = (pred_vol == crack_class)
    closed     = binary_closing(crack_mask, structure=struct)

    out = pred_vol.copy()
    out[closed & ~crack_mask] = crack_class
    return out
# endregion

class SegmentationPipeline:
    """
    Full pipeline: TIFF → 2D segmentation → optional 3D MRF → mesh + metrics.

    model_name : key in MODEL_REGISTRY
    apply_3d_mrf : override to force or suppress 3D MRF regardless of model defaults
    device     : torch device string; defaults to cuda if available
    target_size: spatial size each slice is resized to before inference
    num_classes: must match the trained model (5 for all new models)
    """

    NUM_CLASSES = 5

    def __init__(
        self,
        model_name:   str         = "UNet++ log-split",
        apply_3d_mrf: bool        = True,
        apply_z_close: bool       = False,
        z_extent:     int         = 3,
        device:       str | None  = None,
        target_size:  tuple       = (256, 256),
        num_classes:  int         = 5,
    ):
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model {model_name!r}. "
                f"Choose from: {list(MODEL_REGISTRY.keys())}"
            )
        self.model_name    = model_name
        self.model_cfg     = MODEL_REGISTRY[model_name]
        self.apply_3d_mrf  = apply_3d_mrf
        self.apply_z_close = apply_z_close
        self.z_extent      = z_extent
        self.device        = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.target_size   = target_size
        self.num_classes   = num_classes

    def run(
        self,
        tiff_path:        str,
        model_path:       str | None = None,
        visualize:        bool = True,
        save_path:        str | None = None,
        progress_callback = None,
        return_metrics:   bool = False,
    ):
        """
        Run the full pipeline on a TIFF volume.

        Parameters
        ----------
        model_path : optional override for checkpoint path
        visualize  : whether to build the Plotly 3D mesh
        return_metrics : whether to compute volume metrics + anomaly flags

        Returns
        -------
        If return_metrics=False : (refined_volume, fig)
        If return_metrics=True  : (refined_volume, fig, metrics, anomalies)
        """
        cfg        = self.model_cfg
        checkpoint = model_path or cfg["path"]
        model_type = cfg["type"]
        num_classes  = cfg["num_classes"]
        normalize    = cfg["normalize"]
        class_scheme = cfg["class_scheme"]

        if not os.path.exists(checkpoint):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint}\n"
                f"Have you trained the '{self.model_name}' model?"
            )

        print(f"Pipeline: {self.model_name}")
        print(f"  checkpoint   : {checkpoint}")
        print(f"  model type   : {model_type}  ({num_classes} classes, {normalize})")
        print(f"  device       : {self.device}")

        # 1. Per-slice 2D inference
        segmented_volume, prob_volume, gray_volume = segment_tiff_volume(
            tiff_path         = tiff_path,
            model_path        = checkpoint,
            model_type        = model_type,
            num_classes       = num_classes,
            normalize         = normalize,
            device            = self.device,
            target_size       = self.target_size,
            return_probs      = True,
            progress_callback = progress_callback,
        )

        # 2. Optional 2D MRF (DilatedCNN models only)
        if cfg["apply_2d_mrf"]:
            print("  Applying 2D MRF per slice…")
            refined_2d = []
            for i in range(segmented_volume.shape[0]):
                prob_map = torch.tensor(prob_volume[:, i])
                refined_2d.append(mrf_gibbs_sampling(prob_map, beta=cfg["mrf_beta"]).numpy())
            segmented_volume = np.stack(refined_2d)

        # 3. Optional 3D MRF
        if self.apply_3d_mrf:
            print(f"  Applying 3D MRF  "
                  f"(beta={cfg['mrf3d_beta']}, "
                  f"iterations={cfg['mrf3d_iterations']}, "
                  f"skip_classes={cfg['skip_classes']})…")
            prob_tensor  = torch.tensor(prob_volume, dtype=torch.float32, device=self.device)
            refined_labels = mrf_gibbs_sampling_3d(
                prob_tensor,
                iterations   = cfg["mrf3d_iterations"],
                beta         = cfg["mrf3d_beta"],
                skip_classes = cfg["skip_classes"],
            )
            refined_volume = refined_labels.cpu().numpy()
        else:
            refined_volume = segmented_volume

        # 4. Optional z-axis morphological closing
        if self.apply_z_close:
            crack_class = cfg.get("crack_class", 4)
            print(f"  Applying z-axis closing  (z_extent={self.z_extent}, crack_class={crack_class})…")
            refined_volume = apply_z_closing(
                refined_volume,
                z_extent=self.z_extent,
                crack_class=crack_class,
            )

        # 5. Optional save
        if save_path:
            import tifffile
            tifffile.imwrite(save_path, refined_volume.astype(np.uint8))
            print(f"  Saved refined volume → {save_path}")

        # 6. Optional 3D visualisation
        fig = None
        if visualize:
            print("  Building 3D mesh…")
            fig = show_volume(
                refined_volume,
                title=f"3D — {self.model_name}",
                class_scheme=class_scheme,
            )

        # 7. Optional volume metrics
        metrics = anomalies = None
        if return_metrics:
            metrics, anomalies = compute_volume_metrics(
                refined_volume,
                class_scheme=class_scheme,
            )

        if return_metrics:
            return refined_volume, fig, metrics, anomalies
        return refined_volume, fig