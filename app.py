from typing import Optional
import io

import streamlit as st
import numpy as np

from src.preprocessor import load_and_binarize, extract_contour, resample_contour
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
num_fourier = st.sidebar.slider("フーリエ係数数 (num_coeffs)", 4, 128, 16)
method = st.sidebar.selectbox("埋め込み手法", ["MDS", "TSNE"])


@st.cache_data
def _process_image(file_bytes: bytes, threshold: int, epsilon_factor: float, num_points: int):
    binary = load_and_binarize(file_bytes, threshold if threshold > 0 else None)
    contour = extract_contour(binary, epsilon_factor)
    if contour is None:
        return None, binary
    resampled = resample_contour(contour, num_points)
    return resampled, binary


@st.cache_data
def _process_multiple(files: tuple, threshold: int, epsilon: float, num_points: int, num_fourier: int, method: str):
    from src.global_features import compute_global_embedding

    contours = {}
    skipped = []
    for f in files:
        try:
            b = f.read()
            contour, _ = _process_image(b, threshold if threshold > 0 else None, epsilon, num_points)
            if contour is None:
                skipped.append(f.name)
            else:
                contours[f.name] = contour
        except Exception:
            skipped.append(getattr(f, "name", "<unknown>"))

    if not contours:
        return None, skipped

    df, skipped_more = compute_global_embedding(contours, num_fourier=num_fourier, method=method)
    skipped += skipped_more
    return df, skipped


tabs = st.tabs(["Global Map", "Single Image", "Local Comparison"])

# --- Global Map Tab
with tabs[0]:
    st.header("Global Map")
    if not uploaded_files or len(uploaded_files) < 2:
        st.info("複数の画像をサイドバーからアップロードしてください（2枚以上）。")
    else:
        df_skipped = _process_multiple(tuple(uploaded_files), threshold, epsilon_factor, num_points, num_fourier, method)
        # _process_multiple returns either (df, skipped) or (None, skipped) or None
        if df_skipped is None or df_skipped[0] is None:
            skipped = [] if df_skipped is None else df_skipped[1]
            st.error("有効な輪郭を持つ画像が見つかりませんでした。パラメータを調整してください。")
            if skipped:
                st.warning(f"以下のファイルは輪郭抽出に失敗しました: {skipped}")
        else:
            df, skipped = df_skipped
            from src.visualizer import plot_global_map

            fig = plot_global_map(df)
            st.plotly_chart(fig, use_container_width=True)

            if skipped:
                st.warning(f"以下のファイルは輪郭抽出に失敗しました: {skipped}")

# --- Single Image Tab
with tabs[1]:
    st.header("Single Image Inspection")
    st.write("左のサイドバーから単一ファイルを選択して輪郭を確認できます。")
    single = st.file_uploader("単一画像 (Single inspection)", accept_multiple_files=False, type=["tif", "tiff", "png", "jpg", "jpeg"], key="single")
    if single is None:
        st.info("ここでは1枚の画像を選んで輪郭を確認できます。")
    else:
        try:
            file_bytes = single.read()
            contour, binary = _process_image(file_bytes, threshold if threshold > 0 else None, epsilon_factor, num_points)
            if contour is None:
                st.error("輪郭が検出できませんでした。閾値設定や画像を確認してください。")
                st.image(binary, caption="Binarized image", use_column_width=True)
            else:
                fig = plot_contour(contour, title=single.name)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("---")
                st.subheader("バイナリ画像")
                st.image(binary, caption="Binarized image", use_column_width=True)
        except Exception as e:
            st.exception(e)

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
                ref_contour, ref_bin = _process_image(b1, threshold if threshold > 0 else None, epsilon_factor, num_points)
                tgt_contour, tgt_bin = _process_image(b2, threshold if threshold > 0 else None, epsilon_factor, num_points)

                if ref_contour is None or tgt_contour is None:
                    st.error("いずれかの画像で輪郭が検出できませんでした。閾値や画像を確認してください。")
                else:
                    from src.local_analysis import align_shapes, compute_local_distance
                    from src.visualizer import plot_local_comparison

                    aligned_tgt, transform = align_shapes(tgt_contour, ref_contour, use_icp=icp_checkbox)

                    # Compute distances: source=ref, target=aligned_tgt
                    _, hausdorff, chamfer, target_to_source = compute_local_distance(ref_contour, aligned_tgt)

                    fig = plot_local_comparison(ref_contour, aligned_tgt, target_to_source, title=f"{ref.name} ↔ {tgt.name}")
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("---")
                    st.metric("Hausdorff distance (px)", f"{hausdorff:.3f}")
                    st.metric("Chamfer mean (px)", f"{chamfer:.3f}")

                    with st.expander("詳細: 変換行列・変換量"):
                        st.write(transform)

            except Exception as e:
                st.exception(e)
