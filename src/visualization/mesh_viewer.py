import numpy as np
import plotly.graph_objects as go
from skimage import measure

CLASS_COLORS = {
    1: '#C8B28A',  # Obvod
    2: 'red',  # Hniloba
    3: 'blue',  # Dutina
    4: 'green',  # HrcaZ
    5: 'purple',  # HrcaN
    6: 'cyan',  # Trhlina
}

def show_volume(volume_3d, spacing=(1, 1, 1)):
    print("Building high-resolution 3D mesh viewer...")

    # Ensure integer labels
    volume_3d = volume_3d.astype(np.uint8)

    fig = go.Figure()

    # Iterate through all classes except background (0)
    for cls, color in CLASS_COLORS.items():
        mask = (volume_3d == cls)
        if np.count_nonzero(mask) == 0:
            continue

        try:
            verts, faces, normals, values = measure.marching_cubes(
                mask.astype(np.float32), level=0.5, spacing=spacing
            )

            fig.add_trace(go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color=color,
                opacity=0.15,
                name=f"Class {cls}",
                flatshading=True,
                lighting=dict(ambient=0.5, diffuse=0.5),
                legendgroup=f"class_{cls}",  # Group per class
                showlegend=True,
            ))
        except Exception as e:
            print(f"Skipping class {cls} (marching cubes failed):", e)

    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.8))
        ),
        title="3D Mesh Segmentation",
        margin=dict(l=0, r=0, b=0, t=30),
        height=900,
        showlegend=True,
        legend_itemclick="toggle",  # Enable click-to-hide
        legend_itemdoubleclick="toggleothers"  # Double-click to isolate class
    )

    return fig
