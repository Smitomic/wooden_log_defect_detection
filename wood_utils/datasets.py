import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms.v2 as transforms
import torchvision.tv_tensors as tv_tensors

from .config import PATCH_SIZE


# Augmentation transform (used in training)
DEFAULT_AUGMENT = transforms.Compose([
    transforms.RandomRotation(degrees=(-180, 180)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
])


# region Training dataset - loads from cache, optional augmentation + repeat
class WoodTrainDataset(Dataset):
    """
    Loads from pre-computed .npy cache.

    *repeat* copies each sample so one epoch sees each slice multiple times.
    Augmentation is applied to all copies except the first (copy 0 = clean).
    """

    def __init__(
        self,
        cached_pairs: list[tuple[str, str]],
        repeat:       int   = 5,
        transform     = DEFAULT_AUGMENT,
    ):
        self.pairs     = cached_pairs
        self.repeat    = repeat
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs) * self.repeat

    def __getitem__(self, idx: int):
        base_idx   = idx % len(self.pairs)
        repeat_idx = idx // len(self.pairs)

        img_file, mask_file = self.pairs[base_idx]
        img  = torch.tensor(np.load(img_file),  dtype=torch.float32)
        mask = torch.tensor(np.load(mask_file), dtype=torch.long)

        if self.transform is not None and repeat_idx > 0:
            img  = tv_tensors.Image(img)
            mask = tv_tensors.Mask(mask)
            img, mask = self.transform(img, mask)
            img  = torch.as_tensor(img,  dtype=torch.float32)
            mask = torch.as_tensor(mask, dtype=torch.long)

        return img, mask
# endregion

# region Validation / test dataset — no augmentation
class WoodValDataset(Dataset):
    # Loads from pre-computed .npy cache - no augmentation.

    def __init__(self, cached_pairs: list[tuple[str, str]]):
        self.pairs = cached_pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_file, mask_file = self.pairs[idx]
        img  = torch.tensor(np.load(img_file),  dtype=torch.float32)
        mask = torch.tensor(np.load(mask_file), dtype=torch.long)
        return img, mask
# endregion

# region Defect-aware sampler (training)
def make_defect_aware_sampler(
    cached_pairs:       list[tuple[str, str]],
    repeat:             int   = 5,
    defect_sample_ratio: float = 0.5,
):
    """
    Build a WeightedRandomSampler that guarantees at least
    *defect_sample_ratio* of every batch comes from slices that contain
    at least one Knot or Crack pixel.

    With 19 % of images having zero defects a naive sampler will waste
    roughly 1 in 5 training steps on clean images.
    """
    from torch.utils.data import WeightedRandomSampler

    DEFECT_CLASSES = {3, 4}

    flags = []
    for img_file, mask_file in cached_pairs:
        mask = np.load(mask_file)
        has_defect = any(int((mask == c).sum()) > 0 for c in DEFECT_CLASSES)
        flags.append(has_defect)

    n_total   = len(flags) * repeat
    n_defect  = sum(flags) * repeat
    n_clean   = n_total - n_defect

    w_defect  = (defect_sample_ratio / n_defect)  if n_defect  > 0 else 1.0
    w_clean   = ((1 - defect_sample_ratio) / n_clean) if n_clean > 0 else 1.0

    weights = []
    for _ in range(repeat):
        for has_defect in flags:
            weights.append(w_defect if has_defect else w_clean)

    print(f"  Defect-aware sampler: {n_defect} defect / {n_clean} clean"
          f"  (target ratio {defect_sample_ratio:.0%})")
    return WeightedRandomSampler(weights, num_samples=n_total, replacement=True)
# endregion

# region Evaluation DataLoader (batched, in-memory)
def make_eval_loader(
    cached_pairs: list[tuple[str, str]],
    batch_size:   int = 32,
) -> DataLoader:
    # Load all cached pairs into RAM and return a DataLoader.
    imgs  = np.stack([np.load(p[0]) for p in cached_pairs])
    masks = np.stack([np.load(p[1]) for p in cached_pairs])
    ds    = TensorDataset(
        torch.tensor(imgs,  dtype=torch.float32),
        torch.tensor(masks, dtype=torch.long),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
# endregion