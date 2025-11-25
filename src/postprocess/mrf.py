import torch
import torch.nn.functional as F

def mrf_gibbs_sampling(prob_map, iterations=5, beta=0.8):
    C, H, W = prob_map.shape
    device = prob_map.device

    # Initial guess for labels: Each pixel is assigned the class with the highest probability
    labels = torch.argmax(prob_map, dim=0)

    # Kernel: one 3×3 per class
    kernel = torch.ones((C, 1, 3, 3), device=device)
    kernel[:, :, 1, 1] = 0  # exclude center pixel

    for _ in range(iterations):
        # Convert labels to one-hot (1, C, H, W)
        onehot = F.one_hot(labels, num_classes=C).permute(2, 0, 1)[None].float()

        # Grouped convolution: each class processed separately
        neighbor_count = F.conv2d(
            onehot,
            kernel,
            padding=1,
            groups=C
        ).squeeze(0)

        smooth_cost = 8 - neighbor_count
        # Unary term: -log(prob)
        unary = -torch.log(prob_map + 1e-6)
        # Energy = unary + beta * pairwise disagreement
        energy = unary + beta * smooth_cost
        # Pick label with lowest energy
        labels = torch.argmin(energy, dim=0)

    return labels
