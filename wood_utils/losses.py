import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NUM_CLASSES

# Class weights
# Tuned for the 20-trees dataset pixel distribution:
#  Background and Wood are abundant -> low weights
#  Knot and Crack are rare → high weights
CLASS_WEIGHTS = torch.tensor(
    [1.0, 5.0, 2.0, 50.0, 50.0],
    dtype=torch.float32,
)


# region Focal Loss
class FocalLoss(nn.Module):
    """
    Focal cross-entropy loss (Lin et al. 2017).

    Reduces the relative loss for easy, well-classified examples so the
    model focuses learning on hard, misclassified examples.  Critical for
    Knot and Crack classes which make up < 0.1 % of pixels.

    Parameters:
    gamma   : focusing parameter.  γ=0 → standard cross-entropy.
    weight  : per-class weights tensor (same as ``nn.CrossEntropyLoss``).
    """

    def __init__(
            self,
            gamma: float = 2.0,
            weight: torch.Tensor | None = None,
            num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits : (B, C, H, W)
        # targets: (B, H, W)
        log_p = F.log_softmax(logits, dim=1)
        p = torch.exp(log_p)

        ce = F.nll_loss(log_p, targets, weight=self.weight, reduction="none")  # (B,H,W)
        pt = p.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,H,W)

        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()
# endregion

# region Combined Focal + Dice Loss
class CombinedLoss(nn.Module):
    """
    Weighted sum of FocalLoss and soft Dice loss.

    Dice loss ensures the model is penalised for systematic under-prediction
    of small classes (e.g. a model that never predicts Crack can get 0
    cross-entropy contribution from Crack but high Dice penalty).

    Parameters:
    alpha   : weight on focal term  (default 0.5)
    gamma   : focal exponent
    weight  : per-class weights for focal term
    smooth  : Laplace smoothing for Dice denominator
    """

    def __init__(
            self,
            alpha: float = 0.5,
            gamma: float = 2.0,
            weight: torch.Tensor | None = None,
            smooth: float = 1.0,
            num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.focal = FocalLoss(gamma=gamma, weight=weight, num_classes=num_classes)
        self.alpha = alpha
        self.smooth = smooth
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal_loss = self.focal(logits, targets)

        # Soft Dice (differentiable, operates on softmax probabilities)
        probs = F.softmax(logits, dim=1)  # (B,C,H,W)
        onehot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()

        intersection = (probs * onehot).sum(dim=(0, 2, 3))
        denominator = (probs + onehot).sum(dim=(0, 2, 3))
        dice_per_cls = 1 - (2 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = dice_per_cls.mean()

        return self.alpha * focal_loss + (1 - self.alpha) * dice_loss
# endregion