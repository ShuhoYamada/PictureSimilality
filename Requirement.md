# 2次元形状類似度解析・可視化Webシステム 開発仕様書

## 1. プロジェクト概要

### 目的

多数のTIFF画像（部品断面図等）を入力とし、形状の「全体的な類似度」による俯瞰マップの作成と、「局所的な形状差」のヒートマップ可視化を行うWebアプリケーションを構築する。

### コアコンセプト

- **全体分析 (Global):** Huモーメントとフーリエ記述子を融合し、形状群の大局的なクラスタリングを行う。
- **局所分析 (Local):** ICPアルゴリズム等による位置合わせを行い、ハウスドルフ距離ベースの差異を可視化する。

---

## 2. 技術スタック

- **言語:** Python 3.10+
- **Webフレームワーク:** Streamlit (UI構築)
- **画像処理:** OpenCV (`cv2`), Pillow
- **数値計算:** NumPy, SciPy (spatial.distance, cKDTree)
- **データ分析:** Scikit-learn (MDS/t-SNE, StandardScaler, PCA)
- **可視化:** Plotly (インタラクティブなグラフ描画)

---

## 3. モジュール構成

プロジェクトは以下のモジュール構成で実装すること。

```
project_root/
├── app.py              # Streamlit Webアプリケーション (エントリーポイント)
├── src/
│   ├── preprocessor.py    # 画像読み込み、ベクトル化
│   ├── global_features.py # 特徴量抽出 (Hu, Fourier)、次元圧縮
│   ├── local_analysis.py  # 位置合わせ (ICP)、距離計算
│   └── visualizer.py      # Plotlyによるグラフ生成ロジック
└── requirements.txt
```

---

## 4. 詳細機能要件

### A. 前処理モジュール (`src/preprocessor.py`)

#### クラス/関数要件

**`load_and_binarize(image_file, threshold)`:**
- StreamlitのアップロードファイルまたはパスからBytesIOを受け取る。
- グレースケール変換後、大津の2値化または指定閾値で2値化。
- `cv2.morphologyEx` (Opening) でノイズ除去を行う。

**`extract_contour(binary_image, epsilon_factor)`:**
- `cv2.findContours` で輪郭抽出。最大面積の輪郭のみを採用する。
- `cv2.approxPolyDP` で頂点数を削減（近似）する。`epsilon` パラメータで精度を調整可能にする。

**`resample_contour(contour, num_points)`:**
- 比較精度向上のため、輪郭の頂点数が指定数（例: 200点）になるように線形補間（リサンプリング）を行う処理を含めることが望ましい。

---

### B. 全体特徴抽出モジュール (`src/global_features.py`)

#### クラス/関数要件

**`compute_hu_moments(contour)`:**
- `cv2.HuMoments` を計算。
- 対数変換 (`-sign(h) × log₁₀|h|`) を行いスケールを合わせる。

**`compute_fourier_descriptors(contour, num_coeffs)`:**
- 輪郭座標を複素数化しFFT適用。
- 並進・回転・スケール不変性を持たせる正規化を行う（DC成分除去、絶対値化、第一成分で除算）。
- 低周波側の `num_coeffs` 個の係数を抽出。

**`compute_global_embedding(contours_dict)`:**
- 全形状に対してHuモーメントとフーリエ記述子を計算し、連結して1つのベクトルにする。
- **重要:** `sklearn.preprocessing.StandardScaler` を適用し、特徴量間の重みを正規化する。
- MDS (多次元尺度構成法) または t-SNE を適用し、2次元座標に圧縮して返す。

---

### C. 局所比較・アライメントモジュール (`src/local_analysis.py`)

#### クラス/関数要件

**`align_shapes(source_contour, target_contour)`:**
- **重心移動:** 両方の重心を原点 (0,0) に合わせる。
- **回転補正:** プロクラステス解析 (Procrustes Analysis) または簡易ICPを用いて、source を target に最も重なるように回転させる。

**`compute_local_distance(source_contour, target_contour)`:**
- アライメント済みの形状に対し、`scipy.spatial.cKDTree` を使用。
- source の各点から target の最も近い点へのユークリッド距離を計算。
- **戻り値:** 各点の距離配列、最大距離（Hausdorff）、平均距離（Chamfer）。

---

### D. 可視化モジュール (`src/visualizer.py`)

> **必須要件:** すべて Plotly (`plotly.graph_objects`) を使用し、Figureオブジェクトを返すこと。

**`plot_global_map(embedding_df)`:**
- MDS/t-SNEの結果を散布図（Scatter plot）にする。
- ホバー情報（Hover text）にファイル名を表示。

**`plot_local_comparison(ref_contour, target_contour, distances)`:**
- **基準形状:** グレーの点線で描画。
- **ターゲット形状:** 散布図マーカー（Markers+Lines）で描画。
- **ヒートマップ:** ターゲット形状のマーカー色を `distances` に基づいて着色（Colorscale: Bluered等）。
- **カラーバー:** 誤差の大きさ（mm/px）を示すカラーバーを付与。
- アスペクト比は `scaleanchor="x", scaleratio=1` で固定し、形状が歪まないようにする。

---

### E. Webアプリケーション (`app.py`)

#### Streamlit実装要件

**サイドバー:**
- `st.file_uploader` (Multiple files, TIFF対応)。
- パラメータ設定（近似精度、フーリエ係数数など）。
- 「解析開始」ボタン。

**キャッシング:**
- 重い処理（画像読み込み、特徴量計算）には `@st.cache_data` を使用し、UI操作ごとの再計算を防ぐ。

**タブ構成:**

| タブ | 機能 |
|------|------|
| **Tab 1: 全体マップ (Global View)** | `visualizer.plot_global_map` の結果を表示 (`st.plotly_chart`)。 |
| **Tab 2: 詳細比較 (Local Comparison)** | 2つのセレクトボックスで「基準画像」と「比較画像」を選択。`local_analysis` で位置合わせと距離計算を実行。一致率スコア等の数値指標を表示。`visualizer.plot_local_comparison` の結果を表示。 |

---

## 5. 開発プロセス指示（AIエージェント向け）

以下のステップ順にコードを実装・提示してください。

1. **Step 1:** `src/preprocessor.py` と `src/visualizer.py` の基本部分を作成し、1枚の画像をアップロードして輪郭をPlotlyで表示するだけの `app.py` を作成する（動作確認用）。

2. **Step 2:** `src/global_features.py` を実装し、複数画像のアップロード、特徴量計算、MDSによる散布図表示機能を `app.py` に追加する。

3. **Step 3:** `src/local_analysis.py` を実装し、ICPによる位置合わせと、距離に応じた色分け表示機能を実装して完成させる。

---

## 制約事項

- 型ヒント（Type Hints）を付与すること。
- エラーハンドリング（輪郭が検出できない画像への対処など）を含めること。
