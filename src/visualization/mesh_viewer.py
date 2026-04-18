"""
3D mesh viewer using Plotly Mesh3d + marching cubes.

Supports two class schemes:
  "new" - 5-class (0=Background, 1=Bark, 2=Wood, 3=Knot, 4=Crack)
  "old" - 7-class (0=Background, 1=Obvod, 2=Hniloba, 3=Dutina,
                   4=HrcaZ, 5=HrcaN, 6=Trhlina)

Bark/Obvod (class 1) is always rendered as the outer hull of the entire
log so the shape is visible regardless of which classes are toggled.
"""

import numpy as np
import plotly.graph_objects as go
from skimage import measure

# 5-class scheme (new models)
_NEW_COLORS = {
    1: "#8D6E63",   # Bark
    2: "#D4A96A",   # Wood
    3: "#FF6F00",   # Knot
    4: "#E53935",   # Crack
}
_NEW_LABELS = {1: "Bark", 2: "Wood", 3: "Knot", 4: "Crack"}
_NEW_OPACITY = {1: 0.15, 2: 0.10, 3: 0.70, 4: 0.80}

# 7-class scheme (old models)
_OLD_COLORS = {
    1: "#8D6E63",   # Obvod   - bark ring
    2: "#A0522D",   # Hniloba - rot (dark brown)
    3: "#4169E1",   # Dutina  - cavity (blue)
    4: "#FF6F00",   # HrcaZ   - healthy knot (amber)
    5: "#9B2335",   # HrcaN   - unhealthy knot (dark red)
    6: "#E53935",   # Trhlina - crack (red)
}
_OLD_LABELS = {
    1: "Obvod",
    2: "Hniloba",
    3: "Dutina",
    4: "Hrca (Zdravá)",
    5: "Hrca (Nezdravá)",
    6: "Trhlina",
}
_OLD_OPACITY = {1: 0.15, 2: 0.65, 3: 0.65, 4: 0.70, 5: 0.70, 6: 0.80}


def show_volume(
    volume_3d:    np.ndarray,
    spacing:      tuple = (10, 1, 1),
    title:        str   = "3D Defect Segmentation",
    class_scheme: str   = "new",
) -> go.Figure:
    """
    Build an interactive Plotly 3D mesh figure from a (D, H, W) label volume.


    volume_3d    : (D, H, W) int array with class labels
    spacing      : physical voxel spacing (depth_mm, height_mm, width_mm)
    title        : figure title
    class_scheme : "new" (5-class) or "old" (7-class)
    """
    colors  = _NEW_COLORS  if class_scheme == "new" else _OLD_COLORS
    labels  = _NEW_LABELS  if class_scheme == "new" else _OLD_LABELS
    opacity = _NEW_OPACITY if class_scheme == "new" else _OLD_OPACITY

    print("Building 3D mesh viewer…")
    volume_3d = volume_3d.astype(np.uint8)
    fig = go.Figure()

    for cls, color in colors.items():
        mask = (volume_3d != 0) if cls == 1 else (volume_3d == cls)
        if np.count_nonzero(mask) == 0:
            continue
        try:
            verts, faces, _, _ = measure.marching_cubes(
                mask.astype(np.float32), level=0.5, spacing=spacing,
            )
            fig.add_trace(go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color=color,
                opacity=opacity[cls],
                name=labels[cls],
                flatshading=True,
                lighting=dict(ambient=0.5, diffuse=0.5),
                legendgroup=f"class_{cls}",
                showlegend=True,
            ))
        except Exception as exc:
            print(f"  Skipping {labels[cls]}: {exc}")

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        scene=dict(
            xaxis_title="Length (depth)",
            yaxis_title="Width",
            zaxis_title="Height",
            aspectmode="data",
            camera=dict(eye=dict(x=-3.2, y=-4, z=0.8)),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=True,
        legend_itemclick="toggle",
        legend_itemdoubleclick="toggleothers",
        uirevision="constant",
    )
    return fig