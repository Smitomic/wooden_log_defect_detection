import torch
import torch.nn.functional as F


# region 2D MRF (per-slice)
def mrf_gibbs_sampling_2d(
        prob_map: torch.Tensor,
        iterations: int = 5,
        beta: float = 0.8,
) -> torch.Tensor:
    """
    Apply 2D Gibbs MRF to a single-slice probability map.

    Parameters:
    prob_map   : (C, H, W) float tensor of softmax probabilities
    iterations : number of Gibbs passes
    beta       : pairwise smoothing weight

    Returns:
    labels : (H, W) long tensor of refined class assignments
    """
    C, H, W = prob_map.shape
    dev = prob_map.device
    labels = torch.argmax(prob_map, dim=0)

    # 3×3 neighbourhood kernel (one per class, centre excluded)
    kernel = torch.ones((C, 1, 3, 3), device=dev)
    kernel[:, :, 1, 1] = 0

    for _ in range(iterations):
        onehot = F.one_hot(labels, num_classes=C).permute(2, 0, 1)[None].float()
        neighbor_count = F.conv2d(onehot, kernel, padding=1, groups=C).squeeze(0)
        smooth_cost = 8 - neighbor_count  # disagreement with 8-neighbours
        unary = -torch.log(prob_map + 1e-6)
        energy = unary + beta * smooth_cost
        labels = torch.argmin(energy, dim=0)

    return labels


# Keeping the old name as an alias for backward compatibility with training notebooks
mrf_gibbs_sampling = mrf_gibbs_sampling_2d
# endregion

# region 3D MRF (full volume, vectorized)
def mrf_gibbs_sampling_3d(
        prob_map: torch.Tensor,
        iterations: int = 3,
        beta: float = 0.3,
        skip_classes: list[int] | None = None,
) -> torch.Tensor:
    """
    Apply 3D Gibbs MRF to a full probability volume using a vectorized
    grouped conv3d (single pass per iteration, all classes simultaneously).

    Parameters:
    prob_map     : (C, D, H, W) float tensor of softmax probabilities
    iterations   : number of Gibbs passes (3 is usually sufficient)
    beta         : pairwise smoothing weight.  Use 0.3 to preserve thin
                   structures; 0.8 over-smooths crack hairlines.
    skip_classes : class indices whose argmax labels are frozen (restored
                   after MRF).  Default [4] (Crack) — MRF degrades hairlines.

    Returns:
    labels : (D, H, W) long tensor of refined class assignments
    """
    if skip_classes is None:
        skip_classes = [4]  # protect Crack by default

    C, D, H, W = prob_map.shape
    dev = prob_map.device

    # Save original labels for frozen-class restoration
    original_labels = torch.argmax(prob_map, dim=0)  # (D, H, W)
    labels = original_labels.clone()

    # 3×3×3 kernel (26-neighbourhood, centre excluded)
    kernel = torch.ones((C, 1, 3, 3, 3), device=dev)
    kernel[:, :, 1, 1, 1] = 0

    for _ in range(iterations):
        # one_hot: (D,H,W) → (1, C, D, H, W)
        onehot = F.one_hot(labels, num_classes=C).permute(3, 0, 1, 2).float().unsqueeze(0)
        # Grouped conv3d counts neighbours of each class at every voxel
        nc = F.conv3d(onehot, kernel, padding=1, groups=C).squeeze(0)  # (C,D,H,W)
        energy = -torch.log(prob_map + 1e-6) + beta * (26 - nc)
        labels = torch.argmin(energy, dim=0)

    # Restore frozen classes - MRF must not vote them away
    for cls in skip_classes:
        frozen_mask = (original_labels == cls)
        labels[frozen_mask] = cls

    return labels
# endregion