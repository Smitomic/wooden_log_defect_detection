import torch
import torch.nn.functional as F

def mrf_gibbs_sampling(prob_map, iterations=5, beta=0.8):
    C, H, W = prob_map.shape
    device = prob_map.device

    # Initial guess for labels: Each pixel is assigned the class with the highest probability
    labels = torch.argmax(prob_map, dim=0)

    # Define convolutional kernel to count neighbors (8-neighborhood)
    kernel = torch.ones((1, 1, 3, 3), device=device)
    kernel[0, 0, 1, 1] = 0  # exclude center pixel

    for _ in range(iterations):
        smooth_costs = []
        for cls in range(C):
            # Create binary map for current class
            class_map = (labels == cls).float()[None, None, ...]
            # Count how many neighbors have this class
            neighbor_count = F.conv2d(class_map, kernel, padding=1).squeeze(0).squeeze(0)
            # Beta - neighbour count
            if cls == 0:
                smooth_costs = 8 - neighbor_count.unsqueeze(0)
            else:
                smooth_costs = torch.cat([smooth_costs, (8 - neighbor_count).unsqueeze(0)], dim=0)

        # Unary term: -log(prob)
        unary = -torch.log(prob_map + 1e-6)

        # Energy = unary + beta * pairwise disagreement
        energy = unary + beta * smooth_costs

        # Pick label with lowest energy
        labels = torch.argmin(energy, dim=0)

    return labels
