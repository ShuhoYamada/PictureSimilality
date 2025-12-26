from typing import Optional

import numpy as np
import plotly.graph_objects as go


def plot_contour(contour: np.ndarray, title: str = "Contour") -> go.Figure:
    """Return a Plotly Figure showing the contour as lines+markers with equal aspect ratio."""
    pts = np.asarray(contour, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("Contour must be an (N,2) array.")

    # Close contour for visual continuity
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pts[:, 0],
            y=pts[:, 1],
            mode="lines+markers",
            marker=dict(size=4, color="#2a9d8f"),
            line=dict(width=2, color="#264653"),
            name="contour",
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title=title,
        xaxis=dict(constrain="domain"),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
        template="plotly_white",
    )

    return fig


def plot_global_map(embedding_df) -> go.Figure:
    """Placeholder: plot a 2D scatter for embeddings (expects DataFrame with x,y,label)."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=embedding_df["x"],
            y=embedding_df["y"],
            mode="markers",
            text=embedding_df.get("label", None),
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.update_layout(title="Global Map", xaxis_title="Dim 1", yaxis_title="Dim 2")
    return fig


def plot_local_comparison(ref_contour: np.ndarray, target_contour: np.ndarray, target_distances: np.ndarray, title: str = "Local Comparison") -> go.Figure:
    """Plot reference contour (dashed grey) and target contour colored by distances.

    - `target_distances` should be same length as `target_contour` and represent per-point error (mm/px).
    """
    ref = np.asarray(ref_contour, dtype=float)
    tgt = np.asarray(target_contour, dtype=float)
    d = np.asarray(target_distances, dtype=float)

    # Ensure closed reference for drawing
    if not np.allclose(ref[0], ref[-1]):
        ref_plot = np.vstack([ref, ref[0]])
    else:
        ref_plot = ref

    # For target, show markers colored by distance
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ref_plot[:, 0],
            y=ref_plot[:, 1],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            name="reference",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=tgt[:, 0],
            y=tgt[:, 1],
            mode="markers+lines",
            marker=dict(size=6, color=d, colorscale="RdBu", colorbar=dict(title="error (px)"), showscale=True),
            line=dict(width=2, color="black"),
            name="target",
            hovertemplate="index: %{pointNumber}<br>error: %{marker.color:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis=dict(constrain="domain"),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_white",
    )

    return fig