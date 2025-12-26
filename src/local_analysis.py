from typing import Tuple, Dict, Optional

import numpy as np
from scipy.spatial import cKDTree
from numpy.linalg import svd


def _centroid(points: np.ndarray) -> np.ndarray:
    return np.mean(points, axis=0)


def align_shapes(
    source_contour: np.ndarray,
    target_contour: np.ndarray,
    use_icp: bool = True,
    max_iters: int = 20,
    tolerance: float = 1e-4,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Align source to target.

    - Centralize both contours to centroids.
    - Compute optimal rotation using SVD (Kabsch/procrustes).
    - Optionally refine with simple ICP iterations.

    Returns (aligned_source, transform_dict) where transform_dict contains 'rotation', 'translation', 'scale' (scale=1 by default)
    """
    if source_contour is None or target_contour is None:
        raise ValueError("Both source and target contours must be provided.")

    src = np.asarray(source_contour, dtype=float).copy()
    tgt = np.asarray(target_contour, dtype=float).copy()

    # center
    src_centroid = _centroid(src)
    tgt_centroid = _centroid(tgt)
    src_centered = src - src_centroid
    tgt_centered = tgt - tgt_centroid

    def kabsch(P, Q):
        # Find rotation R that minimizes ||RP - Q||
        H = P.T @ Q
        U, S, Vt = svd(H)
        R = Vt.T @ U.T
        # Reflection correction
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        return R

    R = kabsch(src_centered, tgt_centered)
    src_aligned = (src_centered @ R.T) + tgt_centroid

    transform = {"rotation": R, "translation": tgt_centroid - (src_centroid @ R.T), "scale": 1.0}

    if use_icp:
        prev_error = np.inf
        src_iter = src.copy()
        for i in range(max_iters):
            tree = cKDTree(tgt)
            dists, idx = tree.query(src_iter)
            matched = tgt[idx]
            # compute Procrustes between src_iter and matched
            src_c = src_iter - np.mean(src_iter, axis=0)
            matched_c = matched - np.mean(matched, axis=0)
            R_iter = kabsch(src_c, matched_c)
            src_iter = (src_c @ R_iter.T) + np.mean(matched, axis=0)
            mean_error = np.mean(dists)
            if abs(prev_error - mean_error) < tolerance:
                break
            prev_error = mean_error
        src_aligned = src_iter
        # update final transform approx (rotation only) -- translation is included in src_aligned
        transform["rotation"] = R_iter
        transform["translation"] = np.mean(matched, axis=0) - (np.mean(source_contour, axis=0) @ R_iter.T)

    return src_aligned, transform


def compute_local_distance(
    source_contour: np.ndarray, target_contour: np.ndarray
) -> Tuple[np.ndarray, float, float, np.ndarray]:
    """Compute distances between source and target contours.

    Returns:
      - source_to_target_distances (np.ndarray, len(source))
      - hausdorff_distance (float)
      - chamfer_mean (float)
      - target_to_source_distances (np.ndarray, len(target))
    """
    src = np.asarray(source_contour, dtype=float)
    tgt = np.asarray(target_contour, dtype=float)

    if len(src) == 0 or len(tgt) == 0:
        raise ValueError("Contours must be non-empty.")

    tree_tgt = cKDTree(tgt)
    d_src, _ = tree_tgt.query(src)

    tree_src = cKDTree(src)
    d_tgt, _ = tree_src.query(tgt)

    hausdorff = max(np.max(d_src), np.max(d_tgt))
    chamfer = 0.5 * (np.mean(d_src) + np.mean(d_tgt))

    return d_src, hausdorff, chamfer, d_tgt
