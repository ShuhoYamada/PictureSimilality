from typing import Optional
import io

import streamlit as st
import numpy as np

from src.preprocessor import load_and_binarize, load_and_binarize_with_original, extract_contour, resample_contour
from src.visualizer import plot_contour


st.set_page_config(page_title="2次元形状類似度解析", layout="wide")
st.title("2次元形状類似度解析")

st.sidebar.header("入力設定")
# Allow multiple files for global view, but keep ability to inspect single image
uploaded_files = st.sidebar.file_uploader(
    "画像をアップロード (TIFF/PNG/JPG) — Global/Local 両対応",
    accept_multiple_files=True,
    type=["tif", "tiff", "png", "jpg", "jpeg"],
)
threshold = st.sidebar.slider("閾値 (0=Otsu)", 0, 255, 0)
epsilon_factor = st.sidebar.slider("approx epsilon factor", 0.001, 0.05, 0.01)
num_points = st.sidebar.slider("リサンプリング点数", 50, 1000, 200)
include_holes = st.sidebar.checkbox("穴（内側輪郭）を含める", value=True)
min_hole_area = st.sidebar.slider("穴の最小面積 (px²)", 10, 1000, 100) if include_holes else 100
num_fourier = st.sidebar.slider("フーリエ係数数 (num_coeffs)", 4, 128, 16)
method = st.sidebar.selectbox("埋め込み手法", ["MDS", "TSNE"])

# クラスタリング設定
st.sidebar.header("クラスタリング設定")
enable_clustering = st.sidebar.checkbox("クラスタリングを有効にする", value=False)
if enable_clustering:
    cluster_method = st.sidebar.selectbox(
        "クラスタリング手法",
        ["K-means", "DBSCAN", "階層的クラスタリング"],
        help="K-means: 指定した数のクラスタに分割\nDBSCAN: 密度ベース（自動でクラスタ数決定）\n階層的: 階層的に統合"
    )
    if cluster_method == "K-means":
        n_clusters = st.sidebar.slider("クラスタ数", 2, 10, 3)
        cluster_params = {"method": "kmeans", "n_clusters": n_clusters}
    elif cluster_method == "DBSCAN":
        eps = st.sidebar.slider("近傍半径 (eps)", 0.1, 2.0, 0.5, 0.1)
        min_samples = st.sidebar.slider("最小サンプル数", 1, 10, 2)
        cluster_params = {"method": "dbscan", "eps": eps, "min_samples": min_samples}
    else:  # 階層的クラスタリング
        n_clusters = st.sidebar.slider("クラスタ数", 2, 10, 3)
        cluster_params = {"method": "hierarchical", "n_clusters": n_clusters}
else:
    cluster_params = None


@st.cache_data
def _process_image(file_bytes: bytes, threshold: int, epsilon_factor: float, num_points: int, include_holes: bool = True, min_hole_area: int = 100):
    """画像を処理して輪郭データを取得
    
    Returns:
        include_holes=True の場合: ((outer, holes, islands), binary)
        include_holes=False の場合: (outer, binary)
    """
    binary = load_and_binarize(file_bytes, threshold if threshold > 0 else None)
    contour_data = extract_contour(binary, epsilon_factor, include_holes=include_holes, min_hole_area=min_hole_area)
    if contour_data is None:
        return None, binary
    
    if include_holes and isinstance(contour_data, tuple):
        # タプルの場合: (outer, holes, islands)
        outer, holes, islands = contour_data
        outer_resampled = resample_contour(outer, num_points)
        # 穴と島もリサンプリング（点数は比例配分）
        holes_resampled = []
        for hole in holes:
            hole_pts = max(20, int(num_points * len(hole) / max(len(outer), 1)))
            holes_resampled.append(resample_contour(hole, hole_pts))
        islands_resampled = []
        for island in islands:
            island_pts = max(20, int(num_points * len(island) / max(len(outer), 1)))
            islands_resampled.append(resample_contour(island, island_pts))
        return (outer_resampled, holes_resampled, islands_resampled), binary
    else:
        # 単純な配列の場合
        resampled = resample_contour(contour_data, num_points)
        return resampled, binary


@st.cache_data
def _process_image_with_original(file_bytes: bytes, threshold: int, epsilon_factor: float, num_points: int, include_holes: bool = True, min_hole_area: int = 100):
    """元画像も一緒に返す処理関数"""
    original, binary = load_and_binarize_with_original(file_bytes, threshold if threshold > 0 else None)
    contour_data = extract_contour(binary, epsilon_factor, include_holes=include_holes, min_hole_area=min_hole_area)
    if contour_data is None:
        return None, original, binary
    
    if include_holes and isinstance(contour_data, tuple):
        outer, holes, islands = contour_data
        outer_resampled = resample_contour(outer, num_points)
        holes_resampled = [resample_contour(h, max(20, int(num_points * len(h) / max(len(outer), 1)))) for h in holes]
        islands_resampled = [resample_contour(isl, max(20, int(num_points * len(isl) / max(len(outer), 1)))) for isl in islands]
        return (outer_resampled, holes_resampled, islands_resampled), original, binary
    else:
        resampled = resample_contour(contour_data, num_points)
        return resampled, original, binary


@st.cache_data
def _process_multiple(files_data: tuple, threshold: int, epsilon: float, num_points: int, num_fourier: int, method: str, include_holes: bool = True, min_hole_area: int = 100):
    """複数画像を処理する（ファイルデータはバイト列のタプルとして渡す）"""
    from src.global_features import compute_global_embedding

    contours = {}
    skipped = []
    for name, data in files_data:
        try:
            contour, _ = _process_image(data, threshold if threshold > 0 else None, epsilon, num_points, include_holes, min_hole_area)
            if contour is None:
                skipped.append(name)
            else:
                contours[name] = contour
        except Exception:
            skipped.append(name)

    if len(contours) < 2:
        return None, skipped

    df, skipped_more = compute_global_embedding(contours, num_fourier=num_fourier, method=method)
    skipped += skipped_more
    return df, skipped


# 画像データを保持するためのキャッシュ関数
@st.cache_data
def _get_contours_and_images(files_data: tuple, threshold: int, epsilon: float, num_points: int, include_holes: bool, min_hole_area: int):
    """輪郭と元画像データを取得
    
    Returns:
        contours: {name: contour_data} - contour_dataは(outer, holes, islands)タプルまたはnp.ndarray
        images: {name: original_image}
        skipped: list of skipped file names
    """
    contours = {}
    images = {}  # 元画像を保持
    skipped = []
    
    for name, data in files_data:
        try:
            original, binary = load_and_binarize_with_original(data, threshold if threshold > 0 else None)
            contour_data = extract_contour(binary, epsilon, include_holes=include_holes, min_hole_area=min_hole_area)
            if contour_data is None:
                skipped.append(name)
            else:
                # 輪郭データをリサンプリング
                if include_holes and isinstance(contour_data, tuple):
                    outer, holes, islands = contour_data
                    outer_resampled = resample_contour(outer, num_points)
                    holes_resampled = [resample_contour(h, max(20, int(num_points * len(h) / max(len(outer), 1)))) for h in holes]
                    islands_resampled = [resample_contour(isl, max(20, int(num_points * len(isl) / max(len(outer), 1)))) for isl in islands]
                    contours[name] = (outer_resampled, holes_resampled, islands_resampled)
                else:
                    contours[name] = resample_contour(contour_data, num_points)
                images[name] = original  # 元画像を保存
        except Exception:
            skipped.append(name)
    
    return contours, images, skipped


tabs = st.tabs(["Global Map", "Single Image", "Local Comparison", "類似画像検索"])

# --- Global Map Tab
with tabs[0]:
    st.header("Global Map")
    if not uploaded_files or len(uploaded_files) < 2:
        st.info("複数の画像をサイドバーからアップロードしてください（2枚以上）。")
    else:
        # アップロード情報を表示
        st.info(f"📁 アップロードされた画像: {len(uploaded_files)}枚")
        
        # セッション状態の初期化
        if 'global_map_analyzed' not in st.session_state:
            st.session_state.global_map_analyzed = False
        if 'global_map_df' not in st.session_state:
            st.session_state.global_map_df = None
        if 'global_map_skipped' not in st.session_state:
            st.session_state.global_map_skipped = []
        
        # 解析ボタン
        col_btn1, col_btn2 = st.columns([1, 3])
        with col_btn1:
            analyze_global = st.button("🔬 解析開始", type="primary", use_container_width=True, key="global_analyze")
        with col_btn2:
            if st.session_state.global_map_analyzed and st.session_state.global_map_df is not None:
                st.success(f"✅ 解析済み: {len(st.session_state.global_map_df)}枚")
        
        # 解析ボタンが押されたら処理を開始
        if analyze_global:
            st.session_state.global_map_analyzed = False
            
            # ファイルデータを準備
            files_data = []
            for f in uploaded_files:
                f.seek(0)
                files_data.append((f.name, f.read()))
            
            # プログレスバーで進捗を表示
            progress_bar = st.progress(0, text="画像を読み込み中...")
            status_text = st.empty()
            
            # 輪郭を抽出
            contours = {}
            skipped = []
            total = len(files_data)
            
            for i, (name, data) in enumerate(files_data):
                try:
                    contour, _ = _process_image(data, threshold, epsilon_factor, num_points, include_holes, min_hole_area)
                    if contour is None:
                        skipped.append(name)
                    else:
                        contours[name] = contour
                except Exception:
                    skipped.append(name)
                
                if (i + 1) % 100 == 0 or i == total - 1:
                    progress = (i + 1) / total
                    progress_bar.progress(progress, text=f"画像を読み込み中... ({i + 1}/{total})")
            
            status_text.text(f"読み込み完了: {len(contours)}枚 (スキップ: {len(skipped)}枚)")
            
            if len(contours) >= 2:
                # 埋め込み計算
                progress_bar.progress(0.5, text="埋め込みを計算中...")
                from src.global_features import compute_global_embedding
                
                df, skipped_more = compute_global_embedding(contours, num_fourier=num_fourier, method=method)
                skipped.extend(skipped_more)
                
                # クラスタリングを適用
                if enable_clustering and cluster_params is not None:
                    progress_bar.progress(0.8, text="クラスタリング中...")
                    from src.global_features import cluster_shapes
                    df = cluster_shapes(df, **cluster_params)
                
                st.session_state.global_map_df = df
                st.session_state.global_map_skipped = skipped
                st.session_state.global_map_analyzed = True
                
                progress_bar.progress(1.0, text="完了!")
                st.rerun()
            else:
                st.error("有効な輪郭を持つ画像が2枚以上見つかりませんでした。")
        
        # 解析済みの場合、結果を表示
        if st.session_state.global_map_analyzed and st.session_state.global_map_df is not None:
            df = st.session_state.global_map_df
            skipped = st.session_state.global_map_skipped
            
            st.markdown("---")
            
            from src.visualizer import plot_global_map
            fig = plot_global_map(df, show_clusters=enable_clustering)
            st.plotly_chart(fig, use_container_width=True)
            
            # クラスタリング結果を表示
            if enable_clustering and "cluster" in df.columns:
                st.subheader("📊 クラスタリング結果")
                
                n_clusters = df["cluster"].nunique()
                noise_count = len(df[df["cluster"] == -1]) if -1 in df["cluster"].values else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("クラスタ数", n_clusters - (1 if noise_count > 0 else 0))
                with col2:
                    st.metric("分類済み", len(df) - noise_count)
                with col3:
                    if noise_count > 0:
                        st.metric("ノイズ（未分類）", noise_count)
                
                with st.expander("🗂️ クラスタ別メンバー一覧"):
                    for cluster_id in sorted(df["cluster"].unique()):
                        if cluster_id == -1:
                            st.write("**ノイズ（未分類）:**")
                        else:
                            st.write(f"**Cluster {cluster_id}:**")
                        members = df[df["cluster"] == cluster_id]["label"].tolist()
                        st.write(", ".join(members))
                        st.markdown("---")
            
            with st.expander("📊 埋め込み座標データを表示"):
                st.dataframe(df)

            if skipped:
                st.warning(f"以下のファイルは輪郭抽出に失敗しました: {skipped}")

# --- Single Image Tab
with tabs[1]:
    st.header("Single Image Inspection")
    st.write("単一の画像をアップロードして輪郭を確認できます。")
    single = st.file_uploader("単一画像 (Single inspection)", accept_multiple_files=False, type=["tif", "tiff", "png", "jpg", "jpeg"], key="single")
    
    if single is None:
        st.info("ここでは1枚の画像を選んで輪郭を確認できます。")
    else:
        # セッション状態の初期化
        if 'single_analyzed' not in st.session_state:
            st.session_state.single_analyzed = False
        if 'single_result' not in st.session_state:
            st.session_state.single_result = None
        
        # 解析ボタン
        col_btn1, col_btn2 = st.columns([1, 3])
        with col_btn1:
            analyze_single = st.button("🔬 解析開始", type="primary", use_container_width=True, key="single_analyze")
        with col_btn2:
            if st.session_state.single_analyzed:
                st.success("✅ 解析済み")
        
        if analyze_single:
            try:
                single.seek(0)
                file_bytes = single.read()
                contour, original, binary = _process_image_with_original(file_bytes, threshold if threshold > 0 else None, epsilon_factor, num_points, include_holes, min_hole_area)
                
                st.session_state.single_result = {
                    'contour': contour,
                    'original': original,
                    'binary': binary,
                    'name': single.name
                }
                st.session_state.single_analyzed = True
                st.rerun()
            except Exception as e:
                st.exception(e)
        
        # 解析済みの場合、結果を表示
        if st.session_state.single_analyzed and st.session_state.single_result:
            result = st.session_state.single_result
            
            st.markdown("---")
            
            # 元画像と処理後画像を並べて表示
            st.subheader("画像比較")
            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.image(result['original'], caption="元画像 (Original)", use_container_width=True)
            with img_col2:
                st.image(result['binary'], caption="二値化画像 (Binarized)", use_container_width=True)
            
            st.markdown("---")
            
            if result['contour'] is None:
                st.error("輪郭が検出できませんでした。閾値設定や画像を確認してください。")
            else:
                st.subheader("抽出された輪郭")
                fig = plot_contour(result['contour'], title=result['name'])
                st.plotly_chart(fig, use_container_width=True)

# --- Local Comparison Tab
with tabs[2]:
    st.header("Local Comparison")
    st.write("2枚の画像を選択して、位置合わせ（重心 + Procrustes / ICP）と差分ヒートマップを表示します。")

    col1, col2 = st.columns(2)
    with col1:
        ref = st.file_uploader("基準画像 (reference)", accept_multiple_files=False, type=["tif", "tiff", "png", "jpg", "jpeg"], key="ref")
    with col2:
        tgt = st.file_uploader("比較画像 (target)", accept_multiple_files=False, type=["tif", "tiff", "png", "jpg", "jpeg"], key="tgt")

    icp_checkbox = st.checkbox("ICP による微調整を行う", value=True)
    run = st.button("解析開始 (Align & Compute)")

    if run:
        if ref is None or tgt is None:
            st.error("基準画像と比較画像の両方を選択してください。")
        else:
            try:
                b1 = ref.read()
                b2 = tgt.read()
                ref_contour, ref_orig, ref_bin = _process_image_with_original(b1, threshold if threshold > 0 else None, epsilon_factor, num_points, include_holes, min_hole_area)
                tgt_contour, tgt_orig, tgt_bin = _process_image_with_original(b2, threshold if threshold > 0 else None, epsilon_factor, num_points, include_holes, min_hole_area)

                # 元画像と処理後画像を並べて表示
                st.subheader("入力画像の比較")
                
                # 基準画像: 元画像と二値化画像
                st.write(f"**基準画像: {ref.name}**")
                ref_col1, ref_col2 = st.columns(2)
                with ref_col1:
                    st.image(ref_orig, caption="元画像 (Original)", use_container_width=True)
                with ref_col2:
                    st.image(ref_bin, caption="二値化画像 (Binarized)", use_container_width=True)
                
                # 比較画像: 元画像と二値化画像
                st.write(f"**比較画像: {tgt.name}**")
                tgt_col1, tgt_col2 = st.columns(2)
                with tgt_col1:
                    st.image(tgt_orig, caption="元画像 (Original)", use_container_width=True)
                with tgt_col2:
                    st.image(tgt_bin, caption="二値化画像 (Binarized)", use_container_width=True)
                
                st.markdown("---")

                if ref_contour is None or tgt_contour is None:
                    st.error("いずれかの画像で輪郭が検出できませんでした。閾値や画像を確認してください。")
                else:
                    from src.local_analysis import align_shapes, compute_local_distance
                    from src.visualizer import plot_local_comparison

                    aligned_tgt, transform = align_shapes(tgt_contour, ref_contour, use_icp=icp_checkbox)

                    # Compute distances: source=ref, target=aligned_tgt
                    _, hausdorff, chamfer, target_to_source = compute_local_distance(ref_contour, aligned_tgt)

                    st.subheader("輪郭の位置合わせ結果")
                    fig = plot_local_comparison(ref_contour, aligned_tgt, target_to_source, title=f"{ref.name} ↔ {tgt.name}")
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("---")
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.metric("Hausdorff distance (px)", f"{hausdorff:.3f}")
                    with metric_col2:
                        st.metric("Chamfer mean (px)", f"{chamfer:.3f}")

                    with st.expander("詳細: 変換行列・変換量"):
                        st.write(transform)

            except Exception as e:
                st.exception(e)

# --- Similar Image Search Tab
with tabs[3]:
    st.header("類似画像検索")
    st.write("画像を選択すると、似ている画像を類似度順に表示します。")
    
    if not uploaded_files or len(uploaded_files) < 2:
        st.info("複数の画像をサイドバーからアップロードしてください（2枚以上）。")
    else:
        # アップロード情報を表示
        st.info(f"📁 アップロードされた画像: {len(uploaded_files)}枚")
        
        # セッション状態の初期化
        if 'similarity_analyzed' not in st.session_state:
            st.session_state.similarity_analyzed = False
        if 'similarity_contours' not in st.session_state:
            st.session_state.similarity_contours = None
        if 'similarity_images' not in st.session_state:
            st.session_state.similarity_images = None
        if 'similarity_features' not in st.session_state:
            st.session_state.similarity_features = None
        if 'similarity_skipped' not in st.session_state:
            st.session_state.similarity_skipped = []
        
        # 解析ボタン
        col_btn1, col_btn2 = st.columns([1, 3])
        with col_btn1:
            analyze_button = st.button("🔬 解析開始", type="primary", use_container_width=True)
        with col_btn2:
            if st.session_state.similarity_analyzed:
                st.success(f"✅ 解析済み: {len(st.session_state.similarity_contours)}枚")
        
        # 解析ボタンが押されたら処理を開始
        if analyze_button:
            st.session_state.similarity_analyzed = False  # リセット
            
            # ファイルデータを準備
            files_data = []
            for f in uploaded_files:
                f.seek(0)
                files_data.append((f.name, f.read()))
            files_data_tuple = tuple(files_data)
            
            # プログレスバーで進捗を表示
            progress_bar = st.progress(0, text="画像を読み込み中...")
            status_text = st.empty()
            
            # 輪郭と画像を取得（バッチ処理）
            contours = {}
            images = {}
            skipped = []
            total = len(files_data_tuple)
            
            for i, (name, data) in enumerate(files_data_tuple):
                try:
                    from src.preprocessor import load_and_binarize_with_original, extract_contour, resample_contour
                    original, binary = load_and_binarize_with_original(data, threshold if threshold > 0 else None)
                    contour_data = extract_contour(binary, epsilon_factor, include_holes=include_holes, min_hole_area=min_hole_area)
                    if contour_data is None:
                        skipped.append(name)
                    else:
                        if include_holes and isinstance(contour_data, tuple):
                            outer, holes, islands = contour_data
                            outer_resampled = resample_contour(outer, num_points)
                            holes_resampled = [resample_contour(h, max(20, int(num_points * len(h) / max(len(outer), 1)))) for h in holes]
                            islands_resampled = [resample_contour(isl, max(20, int(num_points * len(isl) / max(len(outer), 1)))) for isl in islands]
                            contours[name] = (outer_resampled, holes_resampled, islands_resampled)
                        else:
                            contours[name] = resample_contour(contour_data, num_points)
                        images[name] = original
                except Exception:
                    skipped.append(name)
                
                # 進捗更新（100件ごと）
                if (i + 1) % 100 == 0 or i == total - 1:
                    progress = (i + 1) / total
                    progress_bar.progress(progress, text=f"画像を読み込み中... ({i + 1}/{total})")
            
            status_text.text(f"読み込み完了: {len(contours)}枚 (スキップ: {len(skipped)}枚)")
            
            if len(contours) >= 2:
                # 特徴量を計算
                progress_bar.progress(0, text="特徴量を計算中...")
                from src.global_features import compute_all_features
                
                features = {}
                total_contours = len(contours)
                for i, (name, contour) in enumerate(contours.items()):
                    try:
                        from src.global_features import compute_feature_vector
                        feat = compute_feature_vector(contour, num_fourier, use_holes=True)
                        if feat is not None:
                            features[name] = feat
                    except Exception:
                        pass
                    
                    if (i + 1) % 100 == 0 or i == total_contours - 1:
                        progress = (i + 1) / total_contours
                        progress_bar.progress(progress, text=f"特徴量を計算中... ({i + 1}/{total_contours})")
                
                # セッション状態に保存
                st.session_state.similarity_contours = contours
                st.session_state.similarity_images = images
                st.session_state.similarity_features = features
                st.session_state.similarity_skipped = skipped
                st.session_state.similarity_analyzed = True
                
                progress_bar.progress(1.0, text="完了!")
                st.success(f"✅ 解析完了: {len(contours)}枚の画像から{len(features)}個の特徴量を抽出しました")
                st.rerun()
            else:
                st.error("有効な輪郭を持つ画像が2枚以上見つかりませんでした。")
        
        # 解析済みの場合、結果を表示
        if st.session_state.similarity_analyzed and st.session_state.similarity_contours:
            contours = st.session_state.similarity_contours
            images = st.session_state.similarity_images
            precomputed_features = st.session_state.similarity_features
            
            st.markdown("---")
            
            # 検索対象の画像を選択
            available_images = list(contours.keys())
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("🔍 検索対象を選択")
                query_image = st.selectbox(
                    "基準画像",
                    available_images,
                    key="query_select"
                )
                
                top_k = st.slider("表示する類似画像数", 1, min(10, len(available_images) - 1), 5)
                
                # 選択した画像を表示
                if query_image and query_image in images:
                    st.image(images[query_image], caption=f"選択中: {query_image}", use_container_width=True)
            
            with col2:
                if query_image:
                    from src.global_features import find_similar_shapes
                    
                    # 類似画像を検索（事前計算済み特徴量を使用）
                    similar = find_similar_shapes(
                        query_image, contours, num_fourier, top_k,
                        precomputed_features=precomputed_features
                    )
                    
                    if similar:
                        st.subheader(f"📊 類似画像 TOP {len(similar)}")
                        
                        # グリッド表示
                        cols_per_row = 3
                        for i in range(0, len(similar), cols_per_row):
                            cols = st.columns(cols_per_row)
                            for j, col in enumerate(cols):
                                idx = i + j
                                if idx < len(similar):
                                    name, score = similar[idx]
                                    with col:
                                        if name in images:
                                            st.image(images[name], use_container_width=True)
                                        st.markdown(f"**{idx + 1}. {name}**")
                                        st.progress(score, text=f"類似度: {score:.1%}")
                    else:
                        st.warning("類似画像が見つかりませんでした。")
            
            # 類似度マトリックス（オプション）
            st.markdown("---")
            st.subheader("📈 類似度マトリックス")
            
            num_images = len(contours)
            st.info(f"📊 対象画像数: {num_images}枚")
            
            if num_images > 500:
                st.warning(f"⚠️ 画像数が多いため（{num_images}枚）、ヒートマップ表示は最大100枚のサンプルに制限されます。完全なマトリックスはCSVでダウンロードできます。")
            
            col_matrix1, col_matrix2 = st.columns(2)
            
            with col_matrix1:
                show_heatmap = st.checkbox("ヒートマップを表示", value=num_images <= 100)
            
            with col_matrix2:
                max_heatmap_samples = st.slider(
                    "ヒートマップ最大サンプル数",
                    min_value=20,
                    max_value=200,
                    value=100,
                    help="ヒートマップに表示する最大画像数。メモリとパフォーマンスのため制限しています。"
                )
            
            from src.global_features import compute_pairwise_similarity, export_full_similarity_matrix
            
            # ヒートマップ表示
            if show_heatmap:
                with st.spinner(f"類似度を計算中... ({min(num_images, max_heatmap_samples)}枚)"):
                    progress_bar = st.progress(0)
                    
                    def update_progress(p):
                        progress_bar.progress(min(p, 1.0))
                    
                    sim_matrix, was_sampled = compute_pairwise_similarity(
                        contours, num_fourier, 
                        max_samples=max_heatmap_samples,
                        progress_callback=update_progress
                    )
                    progress_bar.empty()
                
                if not sim_matrix.empty:
                    if was_sampled:
                        st.info(f"🎲 表示用に{max_heatmap_samples}枚をランダムサンプリングしました。")
                    
                    import plotly.express as px
                    
                    fig = px.imshow(
                        sim_matrix.values,
                        x=sim_matrix.columns,
                        y=sim_matrix.index,
                        color_continuous_scale="RdYlGn",
                        aspect="auto",
                        title="類似度マトリックス（緑=類似、赤=非類似）"
                    )
                    fig.update_layout(
                        xaxis_title="",
                        yaxis_title="",
                        xaxis=dict(tickangle=45),
                        height=max(400, min(800, len(sim_matrix) * 5))
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # CSVダウンロード（完全版）
            st.markdown("### 📥 完全な類似度マトリックスをダウンロード")
            
            if st.button("類似度マトリックスを生成（CSV）", key="generate_full_matrix"):
                with st.spinner(f"全{num_images}枚の類似度を計算中...（大規模データの場合、数分かかることがあります）"):
                    csv_bytes = export_full_similarity_matrix(contours, num_fourier)
                
                if csv_bytes:
                    st.download_button(
                        "📥 CSVをダウンロード",
                        csv_bytes,
                        "similarity_matrix_full.csv",
                        "text/csv",
                        key="download_full_matrix"
                    )
                    st.success(f"✅ {num_images}x{num_images}の類似度マトリックスを生成しました。")
            
            if skipped:
                st.warning(f"以下のファイルは輪郭抽出に失敗しました: {skipped}")