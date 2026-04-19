"""
Visualisation utilities shared across evaluation and 3D evaluation notebooks.

colorise_mask       - (H,W) label array -> (H,W,3) RGB float array
hex_to_rgb          - "#RRGGBB" -> (R,G,B) floats in [0,1]
make_mesh_traces    - build Plotly Mesh3d traces from a (D,H,W) label volume
visualise_3d_comparison - side-by-side Plotly 3D: GT | model1 | model2 | ...
apply_plot_theme    - apply dark matplotlib theme from config
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skimage import measure

from .config import (
    CLASS_NAMES, CLASS_COLORS_HEX, CLASS_COLORS_PLOTLY,
    NUM_CLASSES, PLOT_RCPARAMS,
)


# region Colour helpers
def hex_to_rgb(hex_str: str) -> tuple[float, float, float]:
    """Convert "#RRGGBB" to (R, G, B) floats in [0, 1]."""
    h = hex_str.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def colorise_mask(mask_np: np.ndarray) -> np.ndarray:
    """
    Map a (H, W) integer label array to a (H, W, 3) float32 RGB image
    using ``CLASS_COLORS_HEX``.
    """
    rgb = np.zeros((*mask_np.shape, 3), dtype=np.float32)
    for c, hx in CLASS_COLORS_HEX.items():
        rgb[mask_np == c] = hex_to_rgb(hx)
    return rgb
# endregion

# region Matplotlib theme
def apply_plot_theme() -> None:
    """Apply the shared dark matplotlib theme (from config.PLOT_RCPARAMS)."""
    plt.rcParams.update(PLOT_RCPARAMS)
# endregion

# region Plotly 3D mesh helpers
def make_mesh_traces(
        volume_3d: np.ndarray,
        spacing: tuple[float, float, float] = (10, 1, 1),
        defect_only: bool = False,
        opacity_full: float = 0.15,
        opacity_defect: float = 0.70,
        show_legend: bool = True,
) -> list[go.Mesh3d]:
    """
    volume_3d    : (D, H, W) uint8 label array
    spacing      : voxel physical spacing (depth_mm, height_mm, width_mm)
    defect_only  : if True, only render Knot (3) and Crack (4)
    """
    traces: list[go.Mesh3d] = []
    volume_3d = volume_3d.astype(np.uint8)

    classes = [3, 4] if defect_only else list(range(1, NUM_CLASSES))
    opacity = opacity_defect if defect_only else opacity_full

    for c in classes:
        cls_name = CLASS_NAMES[c]
        # Bark = outer hull of whole log (include everything non-background)
        mask = (volume_3d != 0) if (c == 1 and not defect_only) else (volume_3d == c)
        if np.count_nonzero(mask) == 0:
            continue
        try:
            verts, faces, _, _ = measure.marching_cubes(
                mask.astype(np.float32), level=0.5, spacing=spacing,
            )
            traces.append(go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color=CLASS_COLORS_PLOTLY[c],
                opacity=opacity,
                name=cls_name,
                flatshading=True,
                lighting=dict(ambient=0.5, diffuse=0.5),
                legendgroup=f"class_{c}",
                showlegend=show_legend,
            ))
        except Exception as exc:
            print(f"  Skipping {cls_name}: {exc}")

    return traces


def visualise_3d_comparison(
        log_id: str,
        gt_vol: np.ndarray,
        pred_vols_dict: dict[str, np.ndarray],
        spacing: tuple[float, float, float] = (10, 1, 1),
        defect_only: bool = False,
) -> go.Figure:
    """
    Side-by-side interactive Plotly 3D: Ground Truth | model1 | model2 | …

    pred_vols_dict : ordered dict mapping display name → (D,H,W) label volume
    """
    col_titles = ["Ground Truth"] + list(pred_vols_dict.keys())
    n_cols = len(col_titles)

    fig = make_subplots(
        rows=1, cols=n_cols,
        specs=[[{"type": "mesh3d"}] * n_cols],
        subplot_titles=col_titles,
        horizontal_spacing=0.02,
    )
    scene_cfg = dict(
        aspectmode="data",
        camera=dict(eye=dict(x=-3.2, y=-4, z=0.8)),
        xaxis_title="Depth", yaxis_title="Width", zaxis_title="Height",
    )

    for trace in make_mesh_traces(gt_vol, spacing, defect_only):
        fig.add_trace(trace, row=1, col=1)

    for col_i, (model_name, pred_vol) in enumerate(pred_vols_dict.items(), start=2):
        for trace in make_mesh_traces(pred_vol, spacing, defect_only, show_legend=False):
            fig.add_trace(trace, row=1, col=col_i)

    for col_i in range(1, n_cols + 1):
        fig.update_layout(**{
            ("scene" if col_i == 1 else f"scene{col_i}"): scene_cfg
        })

    tag = "Defect-only" if defect_only else "Full segmentation"
    fig.update_layout(
        title=dict(text=f"3D {tag} — {log_id}", font=dict(size=15)),
        height=600,
        margin=dict(l=0, r=0, b=0, t=60),
        showlegend=True,
        legend_itemclick="toggle",
        legend_itemdoubleclick="toggleothers",
        uirevision="constant",
    )
    return fig
# endregion