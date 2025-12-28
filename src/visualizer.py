from typing import Optional, Union, List, Tuple

import numpy as np
import plotly.graph_objects as go


# 型エイリアス
ContourWithHoles = Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]
ContourData = Union[np.ndarray, ContourWithHoles]


def plot_contour(contour: ContourData, title: str = "Contour") -> go.Figure:
    """Return a Plotly Figure showing the contour as lines+markers with equal aspect ratio.
    
    穴や島がある場合は、それぞれ別の色で表示します。
    """
    fig = go.Figure()
    
    # 穴対応かどうかチェック
    if isinstance(contour, tuple) and len(contour) >= 2:
        outer, holes, *rest = contour
        islands = rest[0] if rest else []
        
        # 外側輪郭
        outer_pts = np.asarray(outer, dtype=float)
        if not np.allclose(outer_pts[0], outer_pts[-1]):
            outer_pts = np.vstack([outer_pts, outer_pts[0]])
        
        fig.add_trace(
            go.Scatter(
                x=outer_pts[:, 0],
                y=outer_pts[:, 1],
                mode="lines+markers",
                marker=dict(size=4, color="#2a9d8f"),
                line=dict(width=2, color="#264653"),
                name="外側輪郭",
                hoverinfo="skip",
            )
        )
        
        # 穴を赤で表示
        for i, hole in enumerate(holes):
            hole_pts = np.asarray(hole, dtype=float)
            if not np.allclose(hole_pts[0], hole_pts[-1]):
                hole_pts = np.vstack([hole_pts, hole_pts[0]])
            
            fig.add_trace(
                go.Scatter(
                    x=hole_pts[:, 0],
                    y=hole_pts[:, 1],
                    mode="lines+markers",
                    marker=dict(size=3, color="#e76f51"),
                    line=dict(width=2, color="#e63946", dash="dash"),
                    name=f"穴 {i+1}",
                    hoverinfo="skip",
                )
            )
        
        # 島を緑で表示
        for i, island in enumerate(islands):
            island_pts = np.asarray(island, dtype=float)
            if not np.allclose(island_pts[0], island_pts[-1]):
                island_pts = np.vstack([island_pts, island_pts[0]])
            
            fig.add_trace(
                go.Scatter(
                    x=island_pts[:, 0],
                    y=island_pts[:, 1],
                    mode="lines+markers",
                    marker=dict(size=3, color="#457b9d"),
                    line=dict(width=2, color="#1d3557", dash="dot"),
                    name=f"島 {i+1}",
                    hoverinfo="skip",
                )
            )
    else:
        # 従来の単純な配列
        pts = np.asarray(contour, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("Contour must be an (N,2) array.")

        if not np.allclose(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[0]])

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
        showlegend=True,
        template="plotly_white",
    )

    return fig


def plot_global_map(embedding_df, show_clusters: bool = False) -> go.Figure:
    """Plot a 2D scatter for embeddings with optional cluster coloring.
    
    Parameters:
        embedding_df: DataFrame with x, y, label columns (and optionally 'cluster')
        show_clusters: クラスタごとに色分けするかどうか
    """
    fig = go.Figure()
    
    if show_clusters and "cluster" in embedding_df.columns:
        # クラスタごとに異なる色でプロット
        clusters = embedding_df["cluster"].unique()
        clusters = sorted([c for c in clusters if c >= 0])  # -1はノイズ（DBSCANの場合）
        
        # カラーパレット
        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
            "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5"
        ]
        
        for i, cluster in enumerate(clusters):
            cluster_df = embedding_df[embedding_df["cluster"] == cluster]
            color = colors[i % len(colors)]
            
            fig.add_trace(
                go.Scatter(
                    x=cluster_df["x"],
                    y=cluster_df["y"],
                    mode="markers",
                    marker=dict(size=12, color=color, line=dict(width=1, color="white")),
                    text=cluster_df["label"],
                    name=f"Cluster {cluster}",
                    hovertemplate="<b>%{text}</b><br>Cluster: " + str(cluster) + "<extra></extra>",
                )
            )
        
        # ノイズ点（DBSCANでcluster=-1の場合）
        noise_df = embedding_df[embedding_df["cluster"] == -1]
        if len(noise_df) > 0:
            fig.add_trace(
                go.Scatter(
                    x=noise_df["x"],
                    y=noise_df["y"],
                    mode="markers",
                    marker=dict(size=8, color="gray", symbol="x"),
                    text=noise_df["label"],
                    name="Noise",
                    hovertemplate="<b>%{text}</b><br>Noise<extra></extra>",
                )
            )
        
        fig.update_layout(
            title="Global Map (Clustered)",
            xaxis_title="Dim 1",
            yaxis_title="Dim 2",
            legend_title="Clusters",
            showlegend=True
        )
    else:
        # クラスタなしの通常表示
        fig.add_trace(
            go.Scatter(
                x=embedding_df["x"],
                y=embedding_df["y"],
                mode="markers",
                marker=dict(size=10, color="#2a9d8f"),
                text=embedding_df.get("label", None),
                hovertemplate="<b>%{text}</b><extra></extra>",
            )
        )
        fig.update_layout(title="Global Map", xaxis_title="Dim 1", yaxis_title="Dim 2")
    
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(scaleanchor="y", scaleratio=1),
    )
    
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
            marker=dict(size=6, color=d, colorscale="Bluered", colorbar=dict(title="error (px)"), showscale=True),
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