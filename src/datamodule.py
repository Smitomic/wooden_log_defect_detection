import os
from glob import glob
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import torchvision.transforms.v2 as transforms
import torchvision.tv_tensors as tv_tensors

from src.preprocess import crop_to_foreground


class WoodDefectDataset(Dataset):
    """
    Using preprocess-like logic for a single pass (no augmentation / repeat).
    Not used for training with augmentation, see WoodAugmentedDataset instead.
    """
    def __init__(self, image_paths: List[str], mask_paths: List[str], size=(256, 256)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.size = size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        img, mask = crop_to_foreground(img, mask)
        img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)

        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        mask[mask == 9] = 3  # map 9 → 3 (Dutina)

        img_tensor = torch.tensor(img, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.long)

        return img_tensor, mask_tensor


class WoodAugmentedDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        size=(256, 256),
        transform=None,
        repeat: int = 5,
    ):
        self.original_image_paths = list(image_paths)
        self.original_mask_paths = list(mask_paths)
        self.size = size
        self.transform = transform
        self.repeat = repeat

        # replicate paths
        self.image_paths = self.original_image_paths * repeat
        self.mask_paths = self.original_mask_paths * repeat

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        base_idx = idx % len(self.original_image_paths)
        repeat_idx = idx // len(self.original_image_paths)

        img_path = self.original_image_paths[base_idx]
        mask_path = self.original_mask_paths[base_idx]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        img, mask = crop_to_foreground(img, mask)
        img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)

        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        mask[mask == 9] = 3

        img_tensor = tv_tensors.Image(torch.tensor(img, dtype=torch.float32))
        mask_tensor = tv_tensors.Mask(torch.tensor(mask, dtype=torch.long))

        if self.transform is not None and repeat_idx > 0:
            img_tensor, mask_tensor = self.transform(img_tensor, mask_tensor)

        return img_tensor, mask_tensor


def _collect_image_mask_pairs(root_dir: str) -> Tuple[list, list]:
    """
    Reproduce exactly the 'filtered_image_paths' and 'filtered_mask_paths'
    logic used in the good notebook (PixelLabelData only).
    """
    root_dir = os.path.abspath(root_dir)
    image_paths = sorted(glob(os.path.join(root_dir, "**", "*.jpg"), recursive=True))
    mask_paths = sorted(glob(os.path.join(root_dir, "**", "PixelLabelData", "*.png"), recursive=True))

    mask_filenames = {
        os.path.basename(mask).replace("Label_1_", "").replace(".png", ".jpg"): mask
        for mask in mask_paths
    }

    filtered_image_paths = []
    filtered_mask_paths = []

    for img_path in image_paths:
        img_filename = os.path.basename(img_path)
        if img_filename in mask_filenames:
            filtered_image_paths.append(img_path)
            filtered_mask_paths.append(mask_filenames[img_filename])

    print(f"Filtered Total Images: {len(filtered_image_paths)}")
    print(f"Filtered Total Masks: {len(filtered_mask_paths)}")
    return filtered_image_paths, filtered_mask_paths


def make_dataloaders(
    root_dir: str,
    batch_size: int = 8,
    size=(256, 256),
    use_augmentation: bool = True,
    repeat: int = 5,
):
    """
    1. PixelLabelData-based pairing
    2. train/test split by slice (test_size=0.2, random_state=42)
    3. optional augmentation with repeat>1.
    """
    img_paths, mask_paths = _collect_image_mask_pairs(root_dir)

    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        img_paths,
        mask_paths,
        test_size=0.2,
        random_state=42,
    )

    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomRotation(degrees=(-180, 180)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])
        val_transform = None

        train_dataset = WoodAugmentedDataset(
            train_imgs, train_masks, size=size, transform=train_transform, repeat=repeat
        )
        val_dataset = WoodAugmentedDataset(
            val_imgs, val_masks, size=size, transform=val_transform, repeat=1
        )
    else:
        train_dataset = WoodDefectDataset(train_imgs, train_masks, size=size)
        val_dataset = WoodDefectDataset(val_imgs, val_masks, size=size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    return train_loader, val_loader
