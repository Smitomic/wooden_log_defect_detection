# Automated Defect Detection in Wooden Logs

Diploma thesis project - FIIT STU Bratislava  
Industrial partner: **Lignosilva**

Deep learning pipeline for automated 3D defect segmentation in wooden logs from CT scan data, with an interactive Shiny visualisation app.

---

## Overview

CT-scanned wooden logs are segmented slice-by-slice using trained convolutional models. Predictions are stacked into 3D volumes, optionally refined with MRF post-processing, and visualised as interactive Plotly meshes. The system detects four defect-relevant classes: **Bark**, **Wood**, **Knot**, and **Crack**.

Four models were trained and compared:

| Model | Split strategy | Post-processing | Best for |
|---|---|---|---|
| DilatedCNN+MRF log-split | Log-stratified | 2D MRF | Crack recall + precision |
| DilatedCNN+MRF random-split | Random 80/20 | 2D MRF | Highest crack precision |
| **UNet++ log-split** | Log-stratified | None | **Best overall (IoU, Recall, VS)** |
| UNet++ random-split | Random 80/20 | None | Balanced precision/recall |

---

## Project Structure

```
wooden_log_defect_detection/
│
├── wood_utils/                  # Shared utility package (import in all notebooks)
│   ├── config.py                # Paths, class names, MODELS registry, plot theme
│   ├── data.py                  # Pair matching, log ID resolution, grouping
│   ├── preprocess.py            # Crop, remap, z-score normalise, cache builders
│   ├── datasets.py              # WoodTrainDataset, WoodValDataset, samplers
│   ├── models.py                # DilatedSegCNN, build_unetpp, load_checkpoint
│   ├── losses.py                # FocalLoss, CombinedLoss, CLASS_WEIGHTS
│   ├── mrf.py                   # 2D + 3D MRF Gibbs sampling (vectorized)
│   ├── metrics.py               # IoU, Dice, boundary, 3D volume metrics
│   ├── training.py              # EarlyStopping, evaluate()
│   └── viz.py                   # colorise_mask, 3D mesh helpers
│
├── notebooks/
│   ├── 20_trees_EDA.ipynb                        # Exploratory data analysis
│   ├── 20_trees_training.ipynb                   # DilatedCNN log-split
│   ├── 20_trees_training-random-split.ipynb      # DilatedCNN random-split
│   ├── 20_trees_training-UNET++.ipynb            # UNet++ log-split
│   ├── 20_trees_training-UNET++_random-split.ipynb
│   ├── 20_trees_training-SegFormer.ipynb         # SegFormer (abandoned - insufficient data)
│   ├── 20_trees_evaluation.ipynb                 # 2D per-image evaluation
│   ├── cross_dataset_defect_analysis.ipynb       # Old vs new dataset defect difficulty
│   └── (project root) 20_trees_evaluation_3D.ipynb  # 3D volume evaluation
│
├── src/
│   ├── pipelines/
│   │   └── segmentation_pipeline.py   # Full inference pipeline (used by app)
│   ├── postprocess/
│   │   ├── mrf.py                     # 2D MRF
│   │   ├── mrf3d.py                   # 3D MRF (vectorized, beta=0.3, skip_classes)
│   │   └── segment_log.py             # Per-slice TIFF inference
│   └── visualization/
│       ├── mesh_viewer.py             # Plotly 3D mesh (5-class + 7-class)
│       └── volume_metrics.py          # 3D structural metrics
│
├── app/
│   └── app.py                   # Shiny app - 3D viewer + volume metrics table
│
├── data/
│   └── 20_trees/
│       ├── Images/              # CT scan slices (.tif) organised by log
│       └── Ground_truths/       # Annotations (.png, PixelLabelData format)
│
├── logs/                        # Model checkpoints (one subfolder per run)
│   ├── 20_trees/checkpoints/best.pt
│   ├── 20_trees_random-split/checkpoints/best.pt
│   ├── 20_trees_UNET++/checkpoints/best.pt
│   ├── 20_trees_UNET++_random-split/checkpoints/best.pt
│   ├── old_model/checkpoints/best.pt
│   └── old_model_mrf/checkpoints/best.pt
│
├── Dockerfile.cpu               # CPU-only build
├── Dockerfile.gpu               # CUDA build (cu121)
├── docker-compose.yml
├── run.ps1                      # Auto-detect GPU and launch (Windows)
├── run.sh                       # Auto-detect GPU and launch (Linux/macOS)
└── requirements.txt
```

---

## Class Scheme

| Index | Class | Raw mask value |
|---|---|---|
| 0 | Background | 1 |
| 1 | Bark | 2 |
| 2 | Wood | 3 |
| 3 | Knot | 4 |
| 4 | Crack | 5 |

The old 7-class models (Obvod / Hniloba / Dutina / HrcaZ / HrcaN / Trhlina) are also supported in the app via the `class_scheme` parameter in the model registry.

---

## Setup

### Option A - Docker (recommended)

**Prerequisites:**
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) - install and start it before running any commands
- For GPU builds: [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) must be installed on the host, and GPU acceleration must be enabled in Docker Desktop (Settings -> Resources -> GPU)
- After installing Docker Desktop, restart your terminal/IDE so `docker` is on the PATH

**Direct commands (recommended - no script needed):**
```powershell
# GPU build
docker compose --profile gpu up --build   # first run (builds image)
docker compose --profile gpu up           # subsequent runs
docker compose --profile gpu up -d        # run detached (background)

# CPU build
docker compose --profile cpu up --build
docker compose --profile cpu up
```

**Or use the auto-detect script (Windows):**
```powershell
# Run with bypass if execution policy blocks it:
powershell -ExecutionPolicy Bypass -File run.ps1
# Or set permanently for your user:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\run.ps1
```

**Managing the container:**
```powershell
docker compose --profile gpu logs -f    # view logs (when detached)
docker compose --profile gpu down       # stop and remove container
```

The app will be available at **http://localhost:8000**.

> **Note:** `data/` and `logs/` are mounted as read-only volumes - they are not copied into the image. Place CT scans in `data/20_trees/` and model checkpoints in `logs/` on the host before starting the container.

> **First build time:** Expect 20–90 minutes depending on internet speed - the PyTorch wheel alone is ~2.5 GB. Subsequent rebuilds after code changes take ~30 seconds (pip layers are cached).

**CUDA version:** `Dockerfile.gpu` installs PyTorch with CUDA 12.1 wheels (`cu121`). Check your driver's supported CUDA version with `nvidia-smi` (shown top-right). If it's lower, change `cu121` to e.g. `cu118` in `Dockerfile.gpu`.

**requirements.txt** is a curated list of runtime dependencies - not a full `pip freeze`. Torch and torchvision are excluded because they are installed separately in the Dockerfile with the correct CPU or CUDA build. Do not replace it with a `pip freeze` output from a local virtualenv.

---

### Option B - Local (virtualenv)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install PyTorch - choose one:
# CPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# GPU (CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install everything else:
pip install -r requirements.txt
```

> **Note:** `requirements.txt` specifies `opencv-python-headless` (required for Linux/Docker). On Windows you can replace it with `opencv-python` if you prefer, but `opencv-python-headless` also works fine locally.

**Run the app:**
```bash
shiny run --launch-browser app/app.py
```

**Run notebooks:**
```bash
jupyter notebook notebooks/
```

---

## Data

Data is **not included** in this repository (proprietary, Lignosilva).

Expected layout:
```
data/20_trees/
├── Images/
│   ├── Dub 1/   *.tif
│   ├── Dub 2/   ...
│   ├── kmen1/
│   └── ...      (20 log directories total)
└── Ground_truths/
    ├── Dub 1/GroundTruthProject/PixelLabelData/   *.png
    └── ...
```

The dataset contains 20 logs (10 Dub / oak, 10 kmen / spruce), ~64 annotated slices per log, approximately 1,280 image–mask pairs total.

---

## Shiny App

Upload any CT log TIFF file and select a model to run full segmentation with optional 3D post-processing.

**Controls:**

| Control | Description |
|---|---|
| Segmentation model | 4 new 5-class models + 2 legacy 7-class models |
| Apply 3D MRF | Smooth label noise across slices (crack class frozen) |
| Apply z-axis crack closing | Bridge isolated crack fragments across slices without lateral expansion |
| Closing extent | z=3 (bridges ≤2 missing slices), z=5, or z=7 |

**Output:**
- Interactive Plotly 3D mesh with per-class toggle
- Volume metrics table: volume (cm³), connected components, continuity, compactness

---

## Training

Each model has its own notebook. All notebooks import shared utilities from `wood_utils`:

```python
import sys, os
sys.path.insert(0, os.path.abspath(".."))
from wood_utils import *
```

**Split strategies:**

- **Log-stratified** (`*-log-split`): Val = [Dub 4, kmen6], Test = [Dub 9, kmen9], Train = remaining 16 logs. Ensures no log appears in both train and test.
- **Random** (`*-random-split`): Test = [Dub 9, kmen9], remaining pairs split 80/20 randomly by image. Allows data leakage between train/val within a log.

**Key training decisions:**
- Loss: Combined FocalLoss (γ=2) + soft Dice, equal weighting
- Class weights: Background=1, Bark=5, Wood=2, Knot=50, Crack=50
- Defect-aware sampler: 75% of each batch from defect-positive images
- Best model saved on defect IoU (not mean IoU)
- Early stopping on validation loss, patience=15

---

## Evaluation

### 2D evaluation (`notebooks/20_trees_evaluation.ipynb`)
Per-image metrics (IoU, Dice, Recall, Precision, VS, HD, ASSD) on the common test set [Dub 9, kmen9]. Global TP/FP/FN accumulation matches training notebook metric computation.

### 3D evaluation (`20_trees_evaluation_3D.ipynb`)
Slices stacked into (D, H, W) volumes using a **global bounding box** per log - fixes the lateral misalignment caused by per-slice cropping when the log bends across slices. Evaluates:
- Overlap metrics (IoU, Dice, Recall, Precision, VS)
- 3D boundary metrics (HD, ASSD) - defect classes only
- Volume structure metrics (components, continuity, compactness)
- Z-axis morphological closing experiment

### Key results (UNet++ log-split on common test set)

| Class | IoU | Recall | VS |
|---|---|---|---|
| Background | 0.992 | 0.997 | - |
| Bark | 0.642 | 0.974 | - |
| Wood | 0.976 | 0.984 | - |
| Knot | 0.643 | 0.964 | - |
| Crack | 0.454 | 0.577 | 0.918 |

Crack and knot IoU reflect intrinsic defect sparsity (crack median pixel coverage 0.01%, median instance size ~6px, heavy intensity overlap with wood tissue) rather than model or data insufficiency - the same pattern appears in a prior experiment on a separate dataset where larger, well-separated classes (rot IoU=0.93, cavity IoU=0.89) were segmented successfully while knot and crack remained at 0.60 and 0.62.

---

## wood_utils Package

All notebooks import from `wood_utils` rather than redefining shared code. To use in a notebook:

```python
import sys, os
sys.path.insert(0, os.path.abspath(".."))   # from notebooks/
from wood_utils import *                     # imports all public names
```

From the project root (e.g. `20_trees_evaluation_3D.ipynb`):
```python
sys.path.insert(0, os.path.abspath("."))
```

Key exports:

```python
# Config
IMAGES_ROOT, GT_ROOT, MODELS, CLASS_NAMES, COMMON_TEST_LOGS

# Data loading
build_pairs(images_root, gt_root)        # sorted by page number
group_by_log(pairs, images_root)

# Preprocessing
build_cache(pairs, cache_dir)            # per-slice (training + 2D eval)
build_cache_global(pairs, cache_dir, images_root)  # per-log bbox (3D eval)

# Models
DilatedSegCNN(), build_unetpp(), load_checkpoint(model_cfg)

# MRF
mrf_gibbs_sampling_2d(prob_map, beta=0.8)
mrf_gibbs_sampling_3d(prob_map, beta=0.3, skip_classes=[4])

# Metrics
compute_volume_metrics_3d(pred_vol, gt_vol, class_idx)
```

---

## Requirements

- Python 3.12
- PyTorch ≥ 2.0 (CPU or CUDA)
- segmentation-models-pytorch
- opencv-python-headless
- scikit-image, scikit-learn, scipy
- shiny, shinywidgets, plotly
- tifffile, pandas, matplotlib, tqdm

See `requirements.txt` for the full list (torch excluded - installed separately).

---

## Acknowledgements

- Dataset and domain expertise: **Lignosilva s.r.o.**
- Supervisor: **doc. Mgr. MA. Gabriela Czanner, MSc., PhD.** - FIIT STU Bratislava
- Architectures: DilatedSegCNN (custom), UNet++ via [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)