import numpy as np
import torch

def apply_dense_crf(
    image,
    mask_logits,
    n_classes=7,
    iter_steps=5,
    sxy_gaussian=1,
    compat_gaussian=2,
    sxy_bilateral=15,
    srgb_bilateral=5,
    compat_bilateral=4
):
    # Apply DenseCRF to softmax logits. Works for torch or numpy inputs.
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except ImportError:
        print("pydensecrf not installed. Run: pip install pydensecrf")
        # fallback: raw argmax
        if isinstance(mask_logits, torch.Tensor):
            return torch.argmax(mask_logits, dim=0).cpu().numpy()
        return np.argmax(mask_logits, axis=0)

    # 1. Convert image -> 3-channel uint8
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    image = image.astype(np.uint8)
    h, w = image.shape[:2]

    # 2. Convert logits -> softmax probabilities
    if isinstance(mask_logits, torch.Tensor):
        mask_logits = mask_logits.detach().cpu().numpy()

    # mask_logits: shape (C, H, W)
    probs = mask_logits.astype(np.float32)
    probs = np.exp(probs) / np.sum(np.exp(probs), axis=0, keepdims=True)

    # Ensure contiguous
    probs = np.ascontiguousarray(probs)

    # 3. Setup CRF
    d = dcrf.DenseCRF2D(w, h, n_classes)

    unary = unary_from_softmax(probs)
    d.setUnaryEnergy(unary)

    # Gaussian smoothing
    d.addPairwiseGaussian(
        sxy=(sxy_gaussian, sxy_gaussian),
        compat=compat_gaussian
    )

    # Bilateral term (image-guided)
    d.addPairwiseBilateral(
        sxy=(sxy_bilateral, sxy_bilateral),
        srgb=(srgb_bilateral,) * 3,
        rgbim=image,
        compat=compat_bilateral
    )

    # 4. Run CRF
    q = d.inference(iter_steps)
    refined = np.argmax(q, axis=0).reshape((h, w))

    return refined
