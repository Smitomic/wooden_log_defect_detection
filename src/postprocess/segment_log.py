import torch
import numpy as np
import cv2
from tqdm import tqdm
from src.model import DilatedSegCNN
from src.postprocess.crf import apply_dense_crf
import tifffile as tiff
import torch.nn.functional as F

def segment_tiff_volume(
    tiff_path,
    model_path,
    num_classes=7,
    device=None,
    target_size=(256, 256),
    return_probs=False,
    progress_callback=None
):
    """
    Segment all slices in a TIFF volume using a trained model.
    Optionally return per-voxel probabilities for later MRF refinement.
    Reports progress if a callback is provided.
    """

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Segmenting on {device}")

    model = DilatedSegCNN(in_channels=1, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    volume = tiff.imread(tiff_path).astype("float32")

    # Normalize TIFF shape
    # Case A: (Z, H, W, 4) or (1, H, W, 4) -> RGBA / BGRA etc.
    if volume.ndim == 4 and volume.shape[-1] in (3, 4):
        print(f"[WARNING] TIFF has {volume.shape[-1]} channels — converting to grayscale.")
        # convert RGB/RGBA -> grayscale
        volume = volume[..., :3].mean(axis=-1)  # drop alpha if present

    # Case B: (H, W, 4) -> grayscale
    elif volume.ndim == 3 and volume.shape[-1] in (3, 4):
        print(f"[WARNING] Single-slice TIFF has {volume.shape[-1]} channels — converting to grayscale.")
        volume = volume[..., :3].mean(axis=-1)
        volume = volume[None, ...]  # add Z dimension

    # Case C: (1, H, W) -> expected, do nothing

    # Case D: (H, W) -> single slice
    elif volume.ndim == 2:
        volume = volume[None, ...]

    # Case E: completely unexpected shape
    elif volume.ndim not in (3,):
        raise ValueError(f"Unsupported TIFF shape {volume.shape} in file {tiff_path}")

    # after normalization, the volume must now be: (Z, H, W)

    n_slices = volume.shape[0]

    segmented_slices = []
    prob_slices = [] if return_probs else None

    with torch.no_grad():
        for i in tqdm(range(n_slices), desc="Running inference"):
            img = volume[i] / 255.0
            img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            tensor = torch.tensor(img_resized[None, None], dtype=torch.float32, device=device)

            logits = model(tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()  # [C,H,W]

            segmented_slices.append(np.argmax(probs, axis=0))

            if return_probs:
                prob_slices.append(probs)

            # Progress callback update every 10 slices or at end
            if progress_callback is not None:
                if i % max(1, n_slices // 50) == 0 or i == n_slices - 1:
                    progress_callback(i + 1, n_slices)

    segmented_volume = np.stack(segmented_slices)  # [D,H,W]
    if return_probs:
        prob_volume = np.stack(prob_slices, axis=1)  # [C,D,H,W]
        return segmented_volume, prob_volume
    else:
        return segmented_volume

