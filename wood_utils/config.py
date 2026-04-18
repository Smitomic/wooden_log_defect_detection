# Shared configuration for all 20_trees notebooks and the Shiny app pipeline.

# Data paths (relative to project root / notebook working directory)
IMAGES_ROOT = "../data/20_trees/Images"
GT_ROOT = "../data/20_trees/Ground_truths"
CACHE_DIR = "./eval_cache"
OUT_DIR = "./outputs"

# region Class definitions
# Raw mask pixel values -> model class indices
CLASS_REMAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
CLASS_NAMES = ["Background", "Bark", "Wood", "Knot", "Crack"]
NUM_CLASSES = 5
DEFECT_CLASSES = [3, 4]
CRACK_CLASS = 4
PATCH_SIZE = 256

# Plotly colours (rgb strings) - used in 3D mesh viewer and evaluation_3D
CLASS_COLORS_PLOTLY = {
    0: "rgb(74,85,104)",  # Background
    1: "rgb(141,110,99)",  # Bark
    2: "rgb(212,169,106)",  # Wood
    3: "rgb(255,111,0)",  # Knot
    4: "rgb(229,57,53)",  # Crack
}

# Matplotlib hex colours - used in 2D evaluation and EDA
CLASS_COLORS_HEX = {
    0: "#4A5568",  # Background
    1: "#8D6E63",  # Bark
    2: "#D4A96A",  # Wood
    3: "#FF6F00",  # Knot
    4: "#E53935",  # Crack
}
# endregion

# Evaluation setup
COMMON_TEST_LOGS = ["Dub 9", "kmen9"]
MAX_BOUNDARY_VOXELS = 50_000  # skip HD/ASSD for huge classes (Background/Wood)

# region Model registry
# Single source of truth for checkpoint paths and model type.
MODELS = {
    "DilatedCNN+MRF log-split": {
        "path": "../logs/20_trees/checkpoints/best.pt",
        "type": "dilated",
        "mrf": True,
        "mrf_beta": 0.8,
        "mrf3d_iterations": 3,
        "mrf3d_beta": 0.3,
        "skip_classes": [4],  # freeze Crack - MRF degrades hairline cracks
    },
    "DilatedCNN+MRF random-split": {
        "path": "../logs/20_trees_random-split/checkpoints/best.pt",
        "type": "dilated",
        "mrf": True,
        "mrf_beta": 0.8,
        "mrf3d_iterations": 3,
        "mrf3d_beta": 0.3,
        "skip_classes": [4],
    },
    "UNet++ log-split": {
        "path": "../logs/20_trees_UNET++/checkpoints/best.pt",
        "type": "unetpp",
        "mrf": False,
        "mrf_beta": 0.0,
        "mrf3d_iterations": 3,
        "mrf3d_beta": 0.3,
        "skip_classes": [4],
    },
    "UNet++ random-split": {
        "path": "../logs/20_trees_UNET++_random-split/checkpoints/best.pt",
        "type": "unetpp",
        "mrf": False,
        "mrf_beta": 0.0,
        "mrf3d_iterations": 3,
        "mrf3d_beta": 0.3,
        "skip_classes": [4],
    },
}
# endregion

# region Plot theme (matplotlib)
DARK_BG = "#0F1117"
PANEL_BG = "#1A1D27"
TEXT_COL = "#E8E8F0"
GRID_COL = "#2A2D3A"
ACCENT = "#F0A500"

PLOT_RCPARAMS = {
    "figure.facecolor": DARK_BG,
    "axes.facecolor": PANEL_BG,
    "axes.edgecolor": GRID_COL,
    "axes.labelcolor": TEXT_COL,
    "axes.titlecolor": TEXT_COL,
    "xtick.color": TEXT_COL,
    "ytick.color": TEXT_COL,
    "text.color": TEXT_COL,
    "grid.color": GRID_COL,
    "grid.linewidth": 0.5,
    "font.family": "monospace",
    "axes.spines.top": False,
    "axes.spines.right": False,
}
# endregion