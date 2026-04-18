import torch
import torch.nn.functional as F


def mrf_gibbs_sampling_3d(
    prob_map:     torch.Tensor,
    iterations:   int         = 3,
    beta:         float       = 0.3,
    skip_classes: list | None = None,
) -> torch.Tensor:
    """
     Parameters
    ----------
    prob_map     : (C, D, H, W) float tensor of softmax probabilities
    iterations   : number of passes (3 is usually sufficient)
    beta         : smoothing strength.  0.3 preserves crack hairlines;
                   the old value of 0.8 destroyed them.
    skip_classes : class indices to freeze (restored after MRF).
                   Default [4] = Crack.

    Returns
    -------
    labels : (D, H, W) long tensor
    """
    if skip_classes is None:
        skip_classes = [4]   # Crack - MRF degrades hairline crack predictions

    C, D, H, W = prob_map.shape
    dev         = prob_map.device

    original_labels = torch.argmax(prob_map, dim=0)
    labels          = original_labels.clone()

    # 3×3×3 26-neighbourhood kernel, one channel per class
    kernel              = torch.ones((C, 1, 3, 3, 3), device=dev)
    kernel[:, :, 1, 1, 1] = 0

    for _ in range(iterations):
        # (D,H,W) -> (1, C, D, H, W) one-hot
        onehot = F.one_hot(labels, num_classes=C).permute(3, 0, 1, 2).float().unsqueeze(0)
        nc     = F.conv3d(onehot, kernel, padding=1, groups=C).squeeze(0)
        energy = -torch.log(prob_map + 1e-6) + beta * (26 - nc)
        labels = torch.argmin(energy, dim=0)

    # Restore frozen classes
    for cls in skip_classes:
        labels[original_labels == cls] = cls

    return labels