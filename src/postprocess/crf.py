import numpy as np
import torch

def apply_dense_crf(image, mask_logits, n_classes=7,
                    iter_steps=5, sxy_gaussian=3, compat_gaussian=3,
                    sxy_bilateral=50, srgb_bilateral=5, compat_bilateral=10):
    # Apply Dense CRF refinement to model predictions.
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except ImportError:
        print("pydensecrf not installed. Run: pip install pydensecrf")
        return np.argmax(mask_logits, axis=0)

    h, w = image.shape[:2]
    # Ensure image is 3-channel
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)

    # Softmax over classes
    probs = torch.softmax(torch.tensor(mask_logits), dim=0).numpy()

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d = dcrf.DenseCRF2D(w, h, n_classes)
    d.setUnaryEnergy(unary)

    # Pairwise terms
    d.addPairwiseGaussian(sxy=(sxy_gaussian, sxy_gaussian), compat=compat_gaussian)
    d.addPairwiseBilateral(sxy=(sxy_bilateral, sxy_bilateral), srgb=(srgb_bilateral,)*3,
                           rgbim=image, compat=compat_bilateral)

    q = d.inference(iter_steps)
    refined = np.argmax(q, axis=0).reshape((h, w))
    return refined
