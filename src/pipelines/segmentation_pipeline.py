import os
from glob import glob

import numpy as np
from src.postprocess.segment_log import segment_tiff_volume
import torch
from src.postprocess.mrf3d import mrf_gibbs_sampling_3d
from src.visualization.mesh_viewer import show_volume
from src.model import DilatedSegCNN


class SegmentationPipeline:
    def __init__(self, model_type="cnn", use_crf=False, use_mrf=False,
                 num_classes=7, target_size=(256,256), device=None):
        self.model_type = model_type.lower() # "cnn"  or "vit"
        self.use_crf = use_crf
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

    def run(self, tiff_path, model_path=None, visualize=True, save_path=None, progress_callback=None):
        # Full pipeline: TIFF → segmentation → optional CRF → optional MRF → visualization.
        model_path = self._resolve_model_path(model_path)
        print(f"Using model checkpoint: {model_path}")
        print(f"Running pipeline on {self.device} using {self.model_type.upper()}")

        # 1. segmentation per slice (CRF optional) and ask for probability volume as well
        segmented_volume, prob_volume = segment_tiff_volume(
            tiff_path=tiff_path,
            model_path=model_path,
            num_classes=self.num_classes,
            device=self.device,
            use_crf=self.use_crf,
            target_size=self.target_size,
            return_probs=True,
            progress_callback=progress_callback,
        )

        # 2. MRF refinement
        if self.use_mrf:
            print("Applying 3-D Gibbs MRF refinement...")
            prob_tensor = torch.tensor(prob_volume, dtype=torch.float32, device=self.device)
            refined_labels = mrf_gibbs_sampling_3d(prob_tensor, iterations=3, beta=0.8)
            refined_volume = refined_labels.cpu().numpy()
        else:
            refined_volume = segmented_volume

        # 3. Save
        if save_path:
            import tifffile as tiff
            tiff.imwrite(save_path, refined_volume.astype(np.uint8))
            print(f"Saved refined volume to {save_path}")

        # 4. Visualization
        fig = None
        if visualize:
            print("Building 3D mesh viewer...")
            fig = show_volume(refined_volume)
            print("3D mesh viewer finished...")
            # fig.show()

        # For per-class confidence or visualization of slices by entropy purposes later
        """
        if save_path:
            import tifffile as tiff
            tiff.imwrite(save_path.replace(".tiff", "_labels.tiff"), refined_volume.astype(np.uint8))
            if prob_volume is not None:
                np.save(save_path.replace(".tiff", "_probs.npy"), prob_volume)
                
        entropy = -np.sum(prob_volume * np.log(prob_volume + 1e-6), axis=0)
        """

        return refined_volume, fig
