import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.model import DilatedSegCNN
from src.datamodule import make_dataloaders
from src.postprocess.crf import apply_dense_crf
from src.postprocess.mrf import mrf_gibbs_sampling
from src.train import compute_iou_global


def evaluate_model(
    weights_path,
    root_dir,
    num_classes=7,
    batch_size=2,
    size=256,
    use_crf=False,
    use_mrf=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader = make_dataloaders(
        root_dir=root_dir,
        batch_size=batch_size,
        size=(size, size),
        use_augmentation=False,
    )

    model = DilatedSegCNN(in_channels=1, num_classes=num_classes)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    print("Evaluating:", weights_path)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc="Evaluating"):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1)

            # MRF
            if use_mrf:
                preds_batch = []
                probs_np = probs.cpu().numpy()
                for b in range(len(probs_np)):
                    prob_map = torch.tensor(probs_np[b], dtype=torch.float32)
                    refined = mrf_gibbs_sampling(prob_map)
                    preds_batch.append(refined)
                preds = torch.tensor(np.stack(preds_batch), dtype=torch.long)

            # CRF
            elif use_crf:
                preds_batch = []
                probs_np = probs.cpu().numpy()
                imgs_np = imgs.cpu().numpy()

                for b in range(len(probs_np)):
                    img = imgs_np[b, 0]
                    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-6)
                    img_gray = (img_norm * 255).astype(np.uint8)

                    refined = apply_dense_crf(
                        img_gray,
                        probs_np[b],
                        n_classes=num_classes,
                    )
                    preds_batch.append(refined)

                preds = torch.tensor(np.stack(preds_batch), dtype=torch.long)

            # RAW
            else:
                preds = torch.argmax(probs, dim=1).cpu()

            all_preds.append(preds.reshape(-1))
            all_targets.append(masks.cpu().reshape(-1))

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    mean_iou, per_class_ious = compute_iou_global(
        all_preds, all_targets, num_classes
    )

    class_names = ["Background", "Obvod", "Hniloba", "Dutina", "HrcaZ", "HrcaN", "Trhlina"]

    print("\nFinal Per-Class IoUs:")
    for i, name in enumerate(class_names):
        print(f"{name:<10} | IoU: {per_class_ious[i]:.4f}")

    print(f"\nMean IoU: {mean_iou:.4f}")

    return mean_iou, per_class_ious

def visualize_slices(
    model,
    loader,
    device="cuda",
    n_slices=5,
    start_index=0,
    use_crf=False,
    use_mrf=False
):
    model.eval()

    # collect slices
    imgs, masks = [], []
    with torch.no_grad():
        for img, mask in loader:
            imgs.append(img)
            masks.append(mask)
            if len(imgs) * img.shape[0] >= n_slices:
                break

    imgs = torch.cat(imgs, dim=0)
    masks = torch.cat(masks, dim=0)

    imgs = imgs[start_index:start_index + n_slices].to(device)
    masks = masks[start_index:start_index + n_slices]

    # forward pass
    with torch.no_grad():
        outputs = model(imgs)
        probs = F.softmax(outputs, dim=1)
        preds_raw = torch.argmax(probs, dim=1)

        # CRF
        if use_crf:
            preds_list = []
            probs_np = probs.cpu().numpy()
            imgs_np = imgs.cpu().numpy()

            for b in range(len(imgs_np)):
                img = imgs_np[b, 0]
                img_norm = (img - img.min()) / (img.max() - img.min() + 1e-6)
                img_gray = (img_norm * 255).astype(np.uint8)

                refined = apply_dense_crf(
                    img_gray, probs_np[b], n_classes=probs.shape[1]
                )
                preds_list.append(refined)

            preds = torch.tensor(np.stack(preds_list), dtype=torch.long)

        # MRF
        elif use_mrf:
            preds_list = []
            probs_np = probs.cpu().numpy()

            for b in range(len(probs_np)):
                prob_map = torch.tensor(probs_np[b], dtype=torch.float32)
                refined = mrf_gibbs_sampling(prob_map)
                preds_list.append(refined)

            preds = torch.tensor(np.stack(preds_list), dtype=torch.long)

        # RAW
        else:
            preds = preds_raw.cpu()

    # plotting
    n = len(imgs)
    plt.figure(figsize=(12, n * 3))

    for i in range(n):
        # image
        plt.subplot(n, 3, 3 * i + 1)
        plt.imshow(imgs[i, 0].cpu(), cmap="gray")
        plt.title(f"Slice {start_index + i}")
        plt.axis("off")

        # ground truth
        plt.subplot(n, 3, 3 * i + 2)
        plt.imshow(masks[i], cmap="jet")
        plt.title("Ground Truth")
        plt.axis("off")

        # prediction
        plt.subplot(n, 3, 3 * i + 3)
        plt.imshow(preds[i], cmap="jet")
        plt.title(
            "Prediction (CRF)" if use_crf else
            "Prediction (MRF)" if use_mrf else
            "Prediction (Raw)"
        )
        plt.axis("off")

    plt.tight_layout()
    plt.show()
