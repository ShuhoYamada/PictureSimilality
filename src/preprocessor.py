from typing import Optional, Tuple, Union
import io

import cv2
import numpy as np
from PIL import Image


def load_image(image_input: Union[str, bytes, io.BytesIO]) -> np.ndarray:
    """Load an image (path or bytes-like) and return a grayscale image (uint8).
    
    Returns the original grayscale image without any processing.
    """
    if isinstance(image_input, str):
        img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found at path: {image_input}")
    else:
        if isinstance(image_input, bytes):
            image_input = io.BytesIO(image_input)
        pil = Image.open(image_input).convert("L")
        img = np.asarray(pil, dtype=np.uint8)
    return img


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


def load_and_binarize_with_original(image_input: Union[str, bytes, io.BytesIO], threshold: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load an image and return both original grayscale and binary images.

    Returns:
        Tuple of (original_grayscale, binary_image)
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

    original = img.copy()

    # Threshold
    if threshold is None:
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(img, int(threshold), 255, cv2.THRESH_BINARY)

    # Morphological opening to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return original, binary


def extract_contour(binary_image: np.ndarray, epsilon_factor: float = 0.01, include_holes: bool = True, min_hole_area: float = 100) -> Optional[np.ndarray]:
    """Extract the largest contour from a binary image and approximate it.

    Returns None if no contour is found.
    
    Parameters:
        binary_image: 二値化画像
        epsilon_factor: 輪郭近似の精度（小さいほど詳細）
        include_holes: Trueの場合、穴（内側輪郭）も含める
        min_hole_area: 穴として認識する最小面積（ノイズ除去用）
    
    Note: 
        - 対象物が黒（0）で背景が白（255）の場合、自動的に反転して処理します
        - 画像の境界に接する輪郭（外枠）は除外されます
        - 穴の中の島、島の中の穴...という階層構造も正しく処理します
    """
    img = binary_image.copy()
    h, w = img.shape[:2]
    
    # 画像の端のピクセル値を確認して、背景色を判定
    corners = [img[0, 0], img[0, w-1], img[h-1, 0], img[h-1, w-1]]
    is_inverted = np.mean(corners) > 127
    if is_inverted:
        img = cv2.bitwise_not(img)
    
    margin = 2
    
    def is_boundary_contour(cnt):
        x, y, cw, ch = cv2.boundingRect(cnt)
        return x <= margin or y <= margin or (x + cw) >= (w - margin) or (y + ch) >= (h - margin)
    
    # RETR_TREE で完全な階層構造を取得
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if not contours or hierarchy is None:
        return None
    
    hierarchy = hierarchy[0]  # shape: (N, 4) - [next, prev, child, parent]
    
    # 最上位（parent == -1）の輪郭から、境界に接しない最大のものを選択
    top_level_contours = []
    for i, cnt in enumerate(contours):
        if hierarchy[i][3] == -1:  # parent == -1 → 最上位
            if not is_boundary_contour(cnt):
                top_level_contours.append((i, cnt, cv2.contourArea(cnt)))
    
    if not top_level_contours:
        # フォールバック: 境界チェックを緩和
        for i, cnt in enumerate(contours):
            if hierarchy[i][3] == -1:
                top_level_contours.append((i, cnt, cv2.contourArea(cnt)))
    
    if not top_level_contours:
        return None
    
    # 最大面積の輪郭を選択
    top_level_contours.sort(key=lambda x: x[2], reverse=True)
    main_idx, main_cnt, _ = top_level_contours[0]
    
    # 輪郭を近似する関数
    def approximate_contour(cnt):
        peri = cv2.arcLength(cnt, True)
        eps = max(1.0, epsilon_factor * peri)
        approx = cv2.approxPolyDP(cnt, eps, True)
        return approx.reshape(-1, 2).astype(float)
    
    pts_outer = approximate_contour(main_cnt)
    
    if not include_holes:
        return pts_outer
    
    # 階層構造を再帰的にたどって、全ての子輪郭を収集
    def collect_all_descendants(parent_idx):
        """指定した輪郭の全ての子孫を収集（階層レベル付き）"""
        descendants = []
        child_idx = hierarchy[parent_idx][2]  # first child
        
        while child_idx != -1:
            area = cv2.contourArea(contours[child_idx])
            if area >= min_hole_area:
                # 階層の深さを計算
                depth = 1
                p = parent_idx
                while hierarchy[p][3] != -1:
                    depth += 1
                    p = hierarchy[p][3]
                
                descendants.append({
                    'idx': child_idx,
                    'contour': contours[child_idx],
                    'area': area,
                    'depth': depth,
                    'is_hole': (depth % 2 == 1)  # 奇数階層は穴、偶数階層は島
                })
                
                # 子の子も再帰的に収集
                descendants.extend(collect_all_descendants(child_idx))
            
            child_idx = hierarchy[child_idx][0]  # next sibling
        
        return descendants
    
    all_inner_contours = collect_all_descendants(main_idx)
    
    if not all_inner_contours:
        return pts_outer
    
    # 全ての輪郭を収集（外側 + 穴 + 島 + ...）
    all_pts_list = [('outer', pts_outer, False)]  # (type, points, is_hole)
    
    for info in all_inner_contours:
        pts = approximate_contour(info['contour'])
        all_pts_list.append((
            'hole' if info['is_hole'] else 'island',
            pts,
            info['is_hole']
        ))
    
    # 全ての点を連結
    combined = []
    for i, (ctype, pts, is_hole) in enumerate(all_pts_list):
        if i > 0:
            # 前の輪郭から現在の輪郭への接続
            combined.append(all_pts_list[0][1][0])  # 外側の始点に戻る
            combined.append(pts[0])  # 現在の輪郭の始点へ
        combined.extend(pts.tolist())
    
    # 最後に外側の始点に戻る
    combined.append(all_pts_list[0][1][0])
    
    return np.array(combined, dtype=float)


def extract_contour_hierarchical(binary_image: np.ndarray, epsilon_factor: float = 0.01, min_area: float = 100) -> Optional[dict]:
    """Extract contours with full hierarchy information.
    
    Returns a dictionary with:
        - 'outer': 外側輪郭の点列
        - 'holes': 穴の輪郭リスト
        - 'islands': 島（穴の中の対象物）の輪郭リスト
        - 'all_contours': 全ての輪郭情報のリスト
    """
    img = binary_image.copy()
    h, w = img.shape[:2]
    
    corners = [img[0, 0], img[0, w-1], img[h-1, 0], img[h-1, w-1]]
    if np.mean(corners) > 127:
        img = cv2.bitwise_not(img)
    
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if not contours or hierarchy is None:
        return None
    
    hierarchy = hierarchy[0]
    margin = 2
    
    def is_boundary_contour(cnt):
        x, y, cw, ch = cv2.boundingRect(cnt)
        return x <= margin or y <= margin or (x + cw) >= (w - margin) or (y + ch) >= (h - margin)
    
    def approximate_contour(cnt):
        peri = cv2.arcLength(cnt, True)
        eps = max(1.0, epsilon_factor * peri)
        approx = cv2.approxPolyDP(cnt, eps, True)
        return approx.reshape(-1, 2).astype(float)
    
    # 最上位輪郭を探す
    main_idx = None
    max_area = 0
    for i, cnt in enumerate(contours):
        if hierarchy[i][3] == -1 and not is_boundary_contour(cnt):
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                main_idx = i
    
    if main_idx is None:
        return None
    
    result = {
        'outer': approximate_contour(contours[main_idx]),
        'holes': [],
        'islands': [],
        'all_contours': []
    }
    
    def collect_descendants(parent_idx, depth=0):
        child_idx = hierarchy[parent_idx][2]
        while child_idx != -1:
            area = cv2.contourArea(contours[child_idx])
            if area >= min_area:
                pts = approximate_contour(contours[child_idx])
                is_hole = (depth % 2 == 0)  # depth 0からの子は穴
                
                info = {
                    'points': pts,
                    'area': area,
                    'depth': depth + 1,
                    'is_hole': is_hole
                }
                
                result['all_contours'].append(info)
                if is_hole:
                    result['holes'].append(pts)
                else:
                    result['islands'].append(pts)
                
                collect_descendants(child_idx, depth + 1)
            
            child_idx = hierarchy[child_idx][0]
    
    collect_descendants(main_idx)
    
    return result


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
