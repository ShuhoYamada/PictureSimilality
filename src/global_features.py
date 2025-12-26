from typing import Dict, Tuple, List, Optional

import warnings

import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS, TSNE


def compute_hu_moments(contour: np.ndarray) -> np.ndarray:
    """Compute Hu moments (7,) with log transform: -sign(h) * log10(|h|).

    contour: (N,2) array
    """
    if contour is None or len(contour) < 3:
        raise ValueError("Contour must have at least 3 points to compute moments.")

    # OpenCV expects contour in int coordinates as Nx1x2 for moments if using as contour
    cnt = contour.astype(np.float64)
    # Create a mask-like moments input by converting to proper shape
    # However cv2.moments accepts a point array as well
    m = cv2.moments(cnt)
    hu = cv2.HuMoments(m).flatten()

    # Log transform with sign
    out = np.zeros_like(hu, dtype=float)
    for i, h in enumerate(hu):
        if h == 0:
            out[i] = 0.0
        else:
            out[i] = -np.sign(h) * np.log10(abs(h))
    return out


def compute_fourier_descriptors(contour: np.ndarray, num_coeffs: int = 16) -> np.ndarray:
    """Compute low-frequency Fourier descriptors invariant to translation/scale/rotation.

    Returns a real-valued vector of length `num_coeffs`.
    """
    if contour is None or len(contour) < 3:
        raise ValueError("Contour must have at least 3 points for Fourier descriptors.")

    pts = np.asarray(contour, dtype=float)
    # Represent as complex numbers
    z = pts[:, 0] + 1j * pts[:, 1]
    # Remove translation
    z -= np.mean(z)
    # FFT
    Z = np.fft.fft(z)

    # Use magnitudes of first num_coeffs (skip DC at index 0)
    # If available coefficients are fewer than requested, pad with zeros
    mags = np.abs(Z)
    if len(mags) - 1 < num_coeffs:
        coeffs = mags[1:]
        coeffs = np.pad(coeffs, (0, num_coeffs - len(coeffs)), constant_values=0.0)
    else:
        coeffs = mags[1 : num_coeffs + 1]

    # Normalize scale by the first coefficient (if non-zero)
    denom = coeffs[0] if coeffs.size > 0 else 1.0
    if denom == 0:
        denom = 1.0
    coeffs = coeffs / denom
    return coeffs.astype(float)


def compute_global_embedding(
    contours: Dict[str, np.ndarray],
    num_fourier: int = 16,
    method: str = "MDS",
    random_state: int = 0,
) -> Tuple[pd.DataFrame, List[str]]:
    """Compute feature vectors (Hu + Fourier) for each contour and return 2D embedding DataFrame.

    Returns (df, skipped_files)
    df columns: ['x','y','label']
    """
    features = []
    labels = []
    skipped: List[str] = []

    for label, contour in contours.items():
        try:
            hu = compute_hu_moments(contour)
            fd = compute_fourier_descriptors(contour, num_fourier)
            feat = np.concatenate([hu, fd])
            features.append(feat)
            labels.append(label)
        except Exception as e:
            warnings.warn(f"Skipping {label}: {e}")
            skipped.append(label)

    if not features:
        raise ValueError("No valid contours to embed.")

    X = np.vstack(features)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    method_up = method.upper()
    if method_up == "MDS":
        m = MDS(n_components=2, random_state=random_state)
        emb = m.fit_transform(Xs)
    elif method_up in ("TSNE", "T-SNE", "T SNE"):
        ts = TSNE(n_components=2, random_state=random_state)
        emb = ts.fit_transform(Xs)
    else:
        raise ValueError("Unknown method: choose 'MDS' or 'TSNE'")

    df = pd.DataFrame({"x": emb[:, 0], "y": emb[:, 1], "label": labels})
    return df, skipped
