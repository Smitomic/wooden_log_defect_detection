import os
from glob import glob

import numpy as np
import torch

from src.postprocess.crf import apply_dense_crf
from src.postprocess.mrf import mrf_gibbs_sampling
from src.postprocess.segment_log import segment_tiff_volume
from src.postprocess.mrf3d import mrf_gibbs_sampling_3d
from src.visualization.mesh_viewer import show_volume
from src.visualization.volume_metrics import compute_volume_metrics
from src.model import DilatedSegCNN


class SegmentationPipeline:
    def __init__(
        self,
        model_type: str = "cnn",
        use_mrf: bool = True,
        num_classes: int = 7,
        target_size=(256, 256),
        device=None,
    ):
        self.model_type = model_type.lower()
        self.use_mrf = use_mrf
        self.num_classes = num_classes
        self.target_size = target_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self, model_path):
        if self.model_type == "cnn":
            model = DilatedSegCNN(in_channels=1, num_classes=self.num_classes)
        elif self.model_type == "vit":
            # TODO: change to ViT when ready
            raise ValueError(f"Unknown model type: {self.model_type}")
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device).eval()
        return model

    def _resolve_model_path(self, model_path=None):
        if model_path and os.path.exists(model_path):
            return model_path

        # Default: pick the newest or "best.pt" checkpoint in logs/
        logs_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "logs")
        candidates = glob(os.path.join(logs_dir, "**", "*.pt"), recursive=True)
        if not candidates:
            raise FileNotFoundError(f"No .pt checkpoints found in {logs_dir}")
        # Prefer a file called 'best.pt' if present, else most recent by mtime
        for c in candidates:
            if os.path.basename(c) == "best.pt":
                return c
        return max(candidates, key=os.path.getmtime)

    def _infer_postprocess_mode(self, model_path):
        # Example model_path:
        # logs/old_model_crf/checkpoints/best.pt

        # Go UP TWO levels to reach the actual model directory
        model_dir = os.path.basename(
            os.path.dirname(
                os.path.dirname(model_path)
            )
        ).lower()

        if "crf" in model_dir:
            return "crf"
        if "mrf" in model_dir:
            return "mrf2d"
        return "none"

    def run(
        self,
        tiff_path,
        model_path=None,
        visualize: bool = True,
        save_path: str | None = None,
        progress_callback=None,
        return_metrics: bool = False,
    ):
        # Full pipeline: TIFF → 2D segmentation → optional 3D MRF → optional visualization.
        model_path = self._resolve_model_path(model_path)
        print(f"Using model checkpoint: {model_path}")
        mode = self._infer_postprocess_mode(model_path)
        print(f"Postprocessing mode detected: {mode}")
        print(f"Running pipeline on {self.device} using {self.model_type.upper()}")

        # 1. Segmentation per slice (probabilities + labels)
        segmented_volume, prob_volume, gray_volume = segment_tiff_volume(
            tiff_path=tiff_path,
            model_path=model_path,
            num_classes=self.num_classes,
            device=self.device,
            target_size=self.target_size,
            return_probs=True,
            progress_callback=progress_callback,
        )

        # Apply per-slice postprocessing
        if mode == "crf":
            # CRF ends up being too slow, while also considering the performance in defect detection being the worst
            """
            print("Applying per-slice CRF refinement...")
            refined = []
            for i in range(segmented_volume.shape[0]):
                img_gray = gray_volume[i]  # [H,W]
                probs = prob_volume[:, i]  # [C,H,W]
                refined_slice = apply_dense_crf(img_gray, probs)
                refined.append(refined_slice)
            segmented_volume = np.stack(refined)
            """


        elif mode == "mrf2d":
            print("Applying per-slice 2D MRF refinement...")
            refined = []
            for i in range(segmented_volume.shape[0]):
                prob_map = torch.tensor(prob_volume[:, i])
                refined_slice = mrf_gibbs_sampling(prob_map)
                refined.append(refined_slice)
            segmented_volume = np.stack(refined)

        # 2. Optional 3D MRF refinement
        if self.use_mrf:
            print("Applying 3-D Gibbs MRF refinement...")
            prob_tensor = torch.tensor(
                prob_volume, dtype=torch.float32, device=self.device
            )
            refined_labels = mrf_gibbs_sampling_3d(
                prob_tensor,
                iterations=3,
                beta=0.8,
            )
            refined_volume = refined_labels.cpu().numpy()
        else:
            refined_volume = segmented_volume

        # 3. Optional save
        if save_path:
            import tifffile as tiff

            tiff.imwrite(save_path, refined_volume.astype(np.uint8))
            print(f"Saved refined volume to {save_path}")

        # 4. Optional 3D visualization
        fig = None
        if visualize:
            print("Building 3D mesh viewer...")
            fig = show_volume(refined_volume)
            print("3D mesh viewer finished...")

        # 5. Optional 3D quality metrics
        metrics = None
        anomalies = None
        if return_metrics:
            metrics, anomalies = compute_volume_metrics(refined_volume)

        if return_metrics:
            return refined_volume, fig, metrics, anomalies
        else:
            return refined_volume, fig
