import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tifffile as tiff
from tqdm import tqdm

NUM_CLASSES  = 5
TARGET_SIZE  = (256, 256)


# # region Model loader
def load_model(model_path: str, model_type: str = "dilated", num_classes: int = NUM_CLASSES, device: str | None = None):
    """
    model_path : path to .pt file
    model_type : "dilated" (DilatedSegCNN) or "unetpp" (UNet++ ResNet34)
    device     : torch device string; defaults to cuda if available
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_type == "dilated":
        from src.model import DilatedSegCNN
        model = DilatedSegCNN(in_channels=1, num_classes=num_classes)
    elif model_type == "unetpp":
        import segmentation_models_pytorch as smp
        import torchvision.models as tvm
        model = smp.UnetPlusPlus(
            encoder_name="resnet34", encoder_weights=None,
            in_channels=1, classes=NUM_CLASSES, activation=None,
        )
        # Load ImageNet encoder weights (adapted for 1-channel input)
        enc_w = tvm.resnet34(weights=tvm.ResNet34_Weights.IMAGENET1K_V1).state_dict()
        enc_w["conv1.weight"] = enc_w["conv1.weight"].mean(dim=1, keepdim=True)
        model_dict = model.encoder.state_dict()
        matched    = {k: v for k, v in enc_w.items()
                      if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(matched)
        model.encoder.load_state_dict(model_dict, strict=False)
    else:
        raise ValueError(f"Unknown model_type {model_type!r}. Expected 'dilated' or 'unetpp'.")

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()
# endregion

# region TIFF volume segmentation
def segment_tiff_volume(
    tiff_path:         str,
    model_path:        str,
    model_type:        str   = "dilated",
    num_classes:       int   = NUM_CLASSES,
    normalize:         str   = "zscore",
    device:            str | None = None,
    target_size:       tuple = TARGET_SIZE,
    return_probs:      bool  = False,
    progress_callback  = None,
):
    """
    normalize : "zscore"  - (img - mean) / std  [new 5-class models]
                "div255"  - img / 255.0          [old 7-class models]


    If return_probs=False : (segmented_volume, gray_volume)
    If return_probs=True  : (segmented_volume, prob_volume, gray_volume)
      segmented_volume : (D, H, W) uint8
      prob_volume      : (C, D, H, W) float32  - for downstream MRF
      gray_volume      : (D, H, W) float32     - original pixel values
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Segmenting on {device}  [model_type={model_type}, normalize={normalize}]")

    model = load_model(model_path, model_type=model_type, device=device, num_classes=num_classes)

    # Load TIFF
    volume = tiff.imread(tiff_path).astype("float32")

    if volume.ndim == 4 and volume.shape[-1] in (3, 4):
        print(f"  TIFF has {volume.shape[-1]} channels — converting to grayscale")
        volume = volume[..., :3].mean(axis=-1)
    elif volume.ndim == 3 and volume.shape[-1] in (3, 4):
        print(f"  Single-slice TIFF has {volume.shape[-1]} channels — converting")
        volume = volume[..., :3].mean(axis=-1)[None]
    elif volume.ndim == 2:
        volume = volume[None]
    elif volume.ndim != 3:
        raise ValueError(f"Unsupported TIFF shape {volume.shape}")

    n_slices = volume.shape[0]
    segmented_slices = []
    prob_slices = [] if return_probs else None

    with torch.no_grad():
        for i in tqdm(range(n_slices), desc="Inference"):
            img = volume[i]

            # Resize
            img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

            # Normalise
            img_f = img_resized.astype(np.float32)
            if normalize == "div255":
                img_f = img_f / 255.0
            else:                        # zscore (default, new 5-class models)
                img_f = (img_f - img_f.mean()) / (img_f.std() + 1e-6)

            tensor = torch.tensor(img_f[None, None], dtype=torch.float32, device=device)
            probs  = F.softmax(model(tensor), dim=1)[0].cpu().numpy()   # (C, H, W)

            segmented_slices.append(np.argmax(probs, axis=0).astype(np.uint8))
            if return_probs:
                prob_slices.append(probs)

            if progress_callback is not None:
                if i % max(1, n_slices // 50) == 0 or i == n_slices - 1:
                    progress_callback(i + 1, n_slices)

    segmented_volume = np.stack(segmented_slices)   # (D, H, W)

    if return_probs:
        prob_volume = np.stack(prob_slices, axis=1)   # (C, D, H, W)
        return segmented_volume, prob_volume, volume
    return segmented_volume, volume
# endregion