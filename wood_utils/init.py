# Config constants
from .config import (
    IMAGES_ROOT, GT_ROOT, CACHE_DIR,
    CLASS_REMAP, CLASS_NAMES, NUM_CLASSES, DEFECT_CLASSES, CRACK_CLASS, PATCH_SIZE,
    CLASS_COLORS_HEX, CLASS_COLORS_PLOTLY,
    COMMON_TEST_LOGS, MAX_BOUNDARY_VOXELS,
    MODELS,
    DARK_BG, PANEL_BG, TEXT_COL, GRID_COL, ACCENT, PLOT_RCPARAMS,
)

# Data
from .data import (
    mask_stem_to_image_stem,
    get_log_id,
    build_pairs,
    group_by_log,
)

# Preprocessing
from .preprocess import (
    crop_to_foreground,
    remap_mask,
    load_and_preprocess,
    build_cache,
    build_cache_global,
)

# Datasets
from .datasets import (
    WoodTrainDataset,
    WoodValDataset,
    make_defect_aware_sampler,
    make_eval_loader,
    DEFAULT_AUGMENT,
)

# Models
from .models import (
    DilatedSegCNN,
    build_unetpp,
    load_checkpoint,
)

# Losses
from .losses import (
    CLASS_WEIGHTS,
    FocalLoss,
    CombinedLoss,
)

# MRF
from .mrf import (
    mrf_gibbs_sampling_2d,
    mrf_gibbs_sampling,      # alias for backward compat
    mrf_gibbs_sampling_3d,
)

# Metrics
from .metrics import (
    compute_metrics_global,
    compute_image_metrics,
    compute_boundary_metrics_2d,
    compute_boundary_metrics_3d,
    compute_volume_metrics_3d,
    safe_compactness,
)

# Training
from .training import (
    EarlyStopping,
    evaluate,
)

# Visualisation
from .viz import (
    hex_to_rgb,
    colorise_mask,
    apply_plot_theme,
    make_mesh_traces,
    visualise_3d_comparison,
)

__all__ = [
    # config
    "IMAGES_ROOT", "GT_ROOT", "CACHE_DIR",
    "CLASS_REMAP", "CLASS_NAMES", "NUM_CLASSES", "DEFECT_CLASSES", "CRACK_CLASS", "PATCH_SIZE",
    "CLASS_COLORS_HEX", "CLASS_COLORS_PLOTLY",
    "COMMON_TEST_LOGS", "MAX_BOUNDARY_VOXELS",
    "MODELS",
    "DARK_BG", "PANEL_BG", "TEXT_COL", "GRID_COL", "ACCENT", "PLOT_RCPARAMS",
    # data
    "mask_stem_to_image_stem", "get_log_id", "build_pairs", "group_by_log",
    # preprocess
    "crop_to_foreground", "remap_mask", "load_and_preprocess",
    "build_cache", "build_cache_global",
    # datasets
    "WoodTrainDataset", "WoodValDataset",
    "make_defect_aware_sampler", "make_eval_loader", "DEFAULT_AUGMENT",
    # models
    "DilatedSegCNN", "build_unetpp", "load_checkpoint",
    # losses
    "CLASS_WEIGHTS", "FocalLoss", "CombinedLoss",
    # mrf
    "mrf_gibbs_sampling_2d", "mrf_gibbs_sampling", "mrf_gibbs_sampling_3d",
    # metrics
    "compute_metrics_global", "compute_image_metrics",
    "compute_boundary_metrics_2d", "compute_boundary_metrics_3d",
    "compute_volume_metrics_3d", "safe_compactness",
    # training
    "EarlyStopping", "evaluate",
    # viz
    "hex_to_rgb", "colorise_mask", "apply_plot_theme",
    "make_mesh_traces", "visualise_3d_comparison",
]