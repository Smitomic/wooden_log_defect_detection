from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import random
from collections import defaultdict
from src.preprocess import preprocess_pair
from pathlib import Path
from tqdm import tqdm

class LogDefectDataset(Dataset):
    def __init__(self, image_mask_pairs, bbox_dict=None, target_size=(256,256)):
        self.samples = image_mask_pairs
        self.target_size = target_size
        self.bbox_dict = bbox_dict or {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, log_name = self.samples[idx]
        bbox = self.bbox_dict.get(log_name)
        img_tensor, mask_tensor = preprocess_pair(img_path, mask_path, bbox=bbox, target_size=self.target_size)
        return img_tensor, mask_tensor


def make_dataloaders(root_dir, batch_size=4, num_workers=0, bbox_dict=None, size=(256,256), split_mode="slice"):
    root_dir = Path(root_dir)
    image_paths = sorted(root_dir.rglob("*.jpg"))
    samples = []

    # match images to masks
    for img_path in tqdm(image_paths, desc="Scanning dataset"):
        img_path = Path(img_path)
        log_name = img_path.parts[-3] if "buk-" in img_path.parts[-3] else img_path.parts[-2]
        mask_dir_candidates = [
            img_path.parent / "PixelLabelData",
            img_path.parent / "LabelingProject" / "GroundTruthProject" / "PixelLabelData",
            img_path.parent.parent / "LabelingProject" / "GroundTruthProject" / "PixelLabelData"
        ]
        stem = img_path.stem
        mask_path = None
        for mdir in mask_dir_candidates:
            if mdir.exists():
                matches = list(mdir.glob(f"*{stem}*.png"))
                if matches:
                    mask_path = matches[0]
                    break
        if mask_path:
            samples.append((str(img_path), str(mask_path), log_name))

    print(f"Total valid samples: {len(samples)}")

    if split_mode == "slice":
        # slice-based split
        train_samples, val_samples = train_test_split(samples, test_size=0.2, random_state=42)
        print(f"Split mode: slice | Train={len(train_samples)} | Val={len(val_samples)}")
    else:
        # log-based split
        log_to_samples = defaultdict(list)
        for s in samples:
            log_to_samples[s[2]].append(s)
        logs = sorted(log_to_samples.keys())
        random.seed(42)
        n_val_logs = max(1, int(0.2 * len(logs)))
        val_logs = set(logs[-n_val_logs:])
        train_samples = [s for lg in logs if lg not in val_logs for s in log_to_samples[lg]]
        val_samples = [s for lg in val_logs for s in log_to_samples[lg]]
        print(f"Split mode: log | Train logs={len(logs)-n_val_logs} | Val logs={n_val_logs}")

    train_ds = LogDefectDataset(train_samples, bbox_dict=bbox_dict, target_size=size)
    val_ds = LogDefectDataset(val_samples, bbox_dict=bbox_dict, target_size=size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
