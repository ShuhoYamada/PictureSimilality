from typing import Optional, Tuple, Union
import io

import cv2
import numpy as np
from PIL import Image


def load_and_binarize(image_input: Union[str, bytes, io.BytesIO], threshold: Optional[int] = None) -> np.ndarray:
    """Load an image (path or bytes-like) and return a binary image (uint8 0/255).

    - If threshold is None, Otsu thresholding is used.
    - Applies morphological opening to remove small noise.
    """
    # Load image into grayscale numpy array
    if isinstance(image_input, str):
        img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found at path: {image_input}")
    else:
        if isinstance(image_input, bytes):
            image_input = io.BytesIO(image_input)
        pil = Image.open(image_input).convert("L")
        img = np.asarray(pil, dtype=np.uint8)

    # Threshold
    if threshold is None:
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(img, int(threshold), 255, cv2.THRESH_BINARY)

    # Morphological opening to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return binary


def extract_contour(binary_image: np.ndarray, epsilon_factor: float = 0.01) -> Optional[np.ndarray]:
    """Extract the largest contour from a binary image and approximate it.

    Returns None if no contour is found.
    """
    contours, _ = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    # Choose the largest area contour
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cnt = contours[0]
    peri = cv2.arcLength(cnt, True)
    epsilon = max(1.0, epsilon_factor * peri)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    pts = approx.reshape(-1, 2).astype(float)
    return pts


def resample_contour(contour: np.ndarray, num_points: int = 200) -> np.ndarray:
    """Resample a contour (Nx2) to have exactly num_points uniformly along arc length."""
    if contour is None or len(contour) < 2:
        raise ValueError("Contour must have at least 2 points to resample.")

    pts = np.asarray(contour, dtype=float)
    # Ensure closed contour for arc-length computation
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    # Compute cumulative distances
    deltas = np.diff(pts, axis=0)
    seg_lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_len = cumulative[-1]
    if total_len == 0:
        raise ValueError("Contour has zero length.")

    # Equally spaced distances along the contour
    sample_dists = np.linspace(0, total_len, num_points, endpoint=False)

    # Interpolate x and y separately
    x = pts[:, 0]
    y = pts[:, 1]
    xs = np.interp(sample_dists, cumulative, x)
    ys = np.interp(sample_dists, cumulative, y)

    return np.vstack([xs, ys]).T
