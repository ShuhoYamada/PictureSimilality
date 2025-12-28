from typing import Dict, Tuple, List, Optional

import warnings

import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS, TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering


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
        m = MDS(n_components=2, random_state=random_state, normalized_stress='auto')
        emb = m.fit_transform(Xs)
    elif method_up in ("TSNE", "T-SNE", "T SNE"):
        # perplexityはサンプル数-1以下でなければならない
        perplexity = min(30, max(1, len(Xs) - 1))
        ts = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
        emb = ts.fit_transform(Xs)
    else:
        raise ValueError("Unknown method: choose 'MDS' or 'TSNE'")

    df = pd.DataFrame({"x": emb[:, 0], "y": emb[:, 1], "label": labels})
    return df, skipped


def cluster_shapes(
    df: pd.DataFrame,
    features: Optional[np.ndarray] = None,
    method: str = "kmeans",
    n_clusters: int = 3,
    eps: float = 0.5,
    min_samples: int = 2,
) -> pd.DataFrame:
    """形状をクラスタリングする
    
    Parameters:
        df: 埋め込み結果のDataFrame（x, y, label列を含む）
        features: 元の特徴量ベクトル（指定しない場合はx,yを使用）
        method: クラスタリング手法 ('kmeans', 'dbscan', 'hierarchical')
        n_clusters: クラスタ数（kmeansとhierarchicalで使用）
        eps: DBSCANの近傍半径
        min_samples: DBSCANの最小サンプル数
    
    Returns:
        クラスタラベルが追加されたDataFrame（'cluster'列）
    """
    df = df.copy()
    
    # 特徴量の準備
    if features is not None:
        X = features
    else:
        X = df[["x", "y"]].values
    
    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    method_lower = method.lower()
    
    if method_lower == "kmeans":
        n_clusters = min(n_clusters, len(X))  # サンプル数以下に制限
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clusterer.fit_predict(X_scaled)
    
    elif method_lower == "dbscan":
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clusterer.fit_predict(X_scaled)
    
    elif method_lower in ("hierarchical", "agglomerative"):
        n_clusters = min(n_clusters, len(X))
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clusterer.fit_predict(X_scaled)
    
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    df["cluster"] = labels
    return df


def compute_cluster_stats(df: pd.DataFrame) -> pd.DataFrame:
    """クラスタごとの統計情報を計算
    
    Returns:
        クラスタごとの統計情報（サンプル数、中心座標など）
    """
    if "cluster" not in df.columns:
        return pd.DataFrame()
    
    stats = df.groupby("cluster").agg({
        "x": ["mean", "std", "count"],
        "y": ["mean", "std"],
        "label": lambda x: list(x)
    }).reset_index()
    
    # カラム名をフラット化
    stats.columns = ["cluster", "x_mean", "x_std", "count", "y_mean", "y_std", "members"]
    
    return stats


def compute_feature_vector(contour: np.ndarray, num_fourier: int = 16) -> Optional[np.ndarray]:
    """単一の輪郭から特徴量ベクトルを計算
    
    Returns:
        特徴量ベクトル (Hu moments + Fourier descriptors)
    """
    try:
        hu = compute_hu_moments(contour)
        fd = compute_fourier_descriptors(contour, num_fourier)
        return np.concatenate([hu, fd])
    except Exception:
        return None


def find_similar_shapes(
    query_name: str,
    contours: Dict[str, np.ndarray],
    num_fourier: int = 16,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """クエリ画像に類似した形状を検索
    
    Parameters:
        query_name: 検索対象の画像名
        contours: 全ての輪郭データ {ファイル名: 輪郭点}
        num_fourier: フーリエ係数数
        top_k: 上位何件を返すか
    
    Returns:
        [(ファイル名, 類似度スコア), ...] のリスト（類似度が高い順）
        類似度スコアは0-1の範囲（1が最も類似）
    """
    if query_name not in contours:
        return []
    
    # 全ての特徴量を計算
    features = {}
    for name, contour in contours.items():
        feat = compute_feature_vector(contour, num_fourier)
        if feat is not None:
            features[name] = feat
    
    if query_name not in features:
        return []
    
    query_feat = features[query_name]
    
    # 標準化
    all_feats = np.array(list(features.values()))
    scaler = StandardScaler()
    scaler.fit(all_feats)
    
    query_scaled = scaler.transform(query_feat.reshape(1, -1))[0]
    
    # 各画像との距離を計算
    distances = []
    for name, feat in features.items():
        if name == query_name:
            continue
        feat_scaled = scaler.transform(feat.reshape(1, -1))[0]
        # ユークリッド距離
        dist = np.linalg.norm(query_scaled - feat_scaled)
        distances.append((name, dist))
    
    # 距離でソート（小さい順）
    distances.sort(key=lambda x: x[1])
    
    # 距離を類似度スコアに変換 (0-1、1が最も類似)
    if distances:
        max_dist = max(d[1] for d in distances) if distances else 1.0
        if max_dist == 0:
            max_dist = 1.0
        similarities = [(name, 1.0 - (dist / (max_dist + 1e-6))) for name, dist in distances]
    else:
        similarities = []
    
    return similarities[:top_k]


def compute_pairwise_similarity(
    contours: Dict[str, np.ndarray],
    num_fourier: int = 16,
    max_samples: int = 100,
    progress_callback=None,
) -> Tuple[pd.DataFrame, bool]:
    """全ペア間の類似度を計算
    
    Parameters:
        contours: 輪郭データの辞書
        num_fourier: フーリエ係数数
        max_samples: ヒートマップ表示用の最大サンプル数（超える場合はサンプリング）
        progress_callback: 進捗コールバック関数
    
    Returns:
        (類似度行列のDataFrame, サンプリングされたかどうか)
    """
    # 全ての特徴量を計算
    names = []
    feat_list = []
    
    total = len(contours)
    for i, (name, contour) in enumerate(contours.items()):
        feat = compute_feature_vector(contour, num_fourier)
        if feat is not None:
            names.append(name)
            feat_list.append(feat)
        if progress_callback and i % 100 == 0:
            progress_callback(i / total * 0.5)  # 前半50%
    
    if len(names) < 2:
        return pd.DataFrame(), False
    
    # サンプリングが必要かチェック
    sampled = False
    if len(names) > max_samples:
        # ランダムサンプリング
        np.random.seed(42)
        indices = np.random.choice(len(names), max_samples, replace=False)
        indices = sorted(indices)
        names = [names[i] for i in indices]
        feat_list = [feat_list[i] for i in indices]
        sampled = True
    
    # 標準化
    X = np.array(feat_list)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if progress_callback:
        progress_callback(0.7)
    
    # ペアワイズ距離を計算
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(X_scaled, X_scaled, metric='euclidean')
    
    if progress_callback:
        progress_callback(0.9)
    
    # 距離を類似度に変換
    max_dist = dist_matrix.max() if dist_matrix.max() > 0 else 1.0
    sim_matrix = 1.0 - (dist_matrix / (max_dist + 1e-6))
    
    # DataFrameに変換
    df = pd.DataFrame(sim_matrix, index=names, columns=names)
    
    if progress_callback:
        progress_callback(1.0)
    
    return df, sampled


def compute_similarity_for_query(
    query_name: str,
    contours: Dict[str, np.ndarray],
    num_fourier: int = 16,
) -> pd.DataFrame:
    """特定の画像に対する全画像の類似度を計算（軽量版）
    
    Returns:
        query_nameに対する各画像の類似度DataFrame
    """
    if query_name not in contours:
        return pd.DataFrame()
    
    # 全ての特徴量を計算
    names = []
    feat_list = []
    
    for name, contour in contours.items():
        feat = compute_feature_vector(contour, num_fourier)
        if feat is not None:
            names.append(name)
            feat_list.append(feat)
    
    if len(names) < 2 or query_name not in names:
        return pd.DataFrame()
    
    # 標準化
    X = np.array(feat_list)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # クエリのインデックスを取得
    query_idx = names.index(query_name)
    query_vec = X_scaled[query_idx].reshape(1, -1)
    
    # クエリと全画像の距離を計算
    from scipy.spatial.distance import cdist
    distances = cdist(query_vec, X_scaled, metric='euclidean')[0]
    
    # 距離を類似度に変換
    max_dist = distances.max() if distances.max() > 0 else 1.0
    similarities = 1.0 - (distances / (max_dist + 1e-6))
    
    # DataFrameに変換
    df = pd.DataFrame({
        "filename": names,
        "similarity": similarities
    })
    df = df.sort_values("similarity", ascending=False)
    
    return df


def export_full_similarity_matrix(
    contours: Dict[str, np.ndarray],
    num_fourier: int = 16,
    output_path: str = None,
) -> bytes:
    """全ペア類似度をCSVとしてエクスポート（大規模データ対応）
    
    Returns:
        CSV形式のバイト列
    """
    import io
    
    # 全ての特徴量を計算
    names = []
    feat_list = []
    
    for name, contour in contours.items():
        feat = compute_feature_vector(contour, num_fourier)
        if feat is not None:
            names.append(name)
            feat_list.append(feat)
    
    if len(names) < 2:
        return b""
    
    # 標準化
    X = np.array(feat_list)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # チャンク処理で類似度を計算（メモリ効率化）
    chunk_size = 500
    
    # CSVヘッダー
    buffer = io.StringIO()
    buffer.write("," + ",".join(names) + "\n")
    
    from scipy.spatial.distance import cdist
    
    # 全体の最大距離を先に計算（サンプリングで推定）
    sample_size = min(100, len(X_scaled))
    sample_indices = np.random.choice(len(X_scaled), sample_size, replace=False)
    sample_dists = cdist(X_scaled[sample_indices], X_scaled[sample_indices], metric='euclidean')
    max_dist = sample_dists.max() if sample_dists.max() > 0 else 1.0
    
    # チャンクごとに処理
    for i in range(0, len(X_scaled), chunk_size):
        chunk = X_scaled[i:i+chunk_size]
        dist_chunk = cdist(chunk, X_scaled, metric='euclidean')
        sim_chunk = 1.0 - (dist_chunk / (max_dist + 1e-6))
        
        for j, row in enumerate(sim_chunk):
            row_name = names[i + j]
            row_str = ",".join([f"{v:.4f}" for v in row])
            buffer.write(f"{row_name},{row_str}\n")
    
    csv_bytes = buffer.getvalue().encode('utf-8')
    return csv_bytes
