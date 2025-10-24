import torch
import torch.nn.functional as F

def mrf_gibbs_sampling_3d(prob_map, iterations=3, beta=0.8):
    # 3D MRF refinement using Gibbs-style updates.
    C, D, H, W = prob_map.shape
    device = prob_map.device

    # initial labels: argmax per voxel
    labels = torch.argmax(prob_map, dim=0)  # [D, H, W]

    # 3×3×3 kernel (26-neighborhood)
    kernel = torch.ones((1, 1, 3, 3, 3), device=device)
    kernel[0, 0, 1, 1, 1] = 0  # exclude center voxel

    for _ in range(iterations):
        smooth_costs = torch.zeros((C, D, H, W), device=device)

        # compute smoothness term per class
        for cls in range(C):
            cls_mask = (labels == cls).float().unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
            neigh_count = F.conv3d(cls_mask, kernel, padding=1).squeeze()  # [D,H,W]
            smooth_costs[cls] = (26 - neigh_count)  # disagreement cost

        unary = -torch.log(prob_map + 1e-6)  # data term
        energy = unary + beta * smooth_costs
        labels = torch.argmin(energy, dim=0)

    return labels
