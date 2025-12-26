# 2次元形状類似度解析・可視化 Web システム

✅ 本リポジトリは、複数のTIFF画像（部品断面図等）を入力に、形状の**全体的類似度（Global）**と**局所差分（Local）**を可視化する Streamlit Web アプリケーションの実装です。

---

## 🔧 目次
- [特徴](#-特徴)
- [セットアップ](#-セットアップ)
- [使い方（簡易）](#-使い方簡易)
- [コマンド一覧](#-コマンド一覧)
- [ファイル構成](#-ファイル構成)
- [注意点 / トラブルシューティング](#-注意点--トラブルシューティング)
- [開発 / テスト](#-開発--テスト)

---

## ✨ 特徴
- **Global:** Huモーメントとフーリエ記述子を連結 → StandardScaler → MDS/t-SNE により 2D 埋め込みを可視化
- **Local:** 重心合わせ + Procrustes（Kabsch）／簡易ICP による位置合わせ → cKDTree ベースで点ごとの誤差を算出（Hausdorff / Chamfer）→ Plotly でヒートマップ可視化
- **UI:** Streamlit で操作可能（複数ファイルアップロード、パラメータ調整、タブ構成）

---

## 🧰 セットアップ（Windows 例）
1. Python3.10+ を用意（推奨: 3.10-3.13）
2. 仮想環境作成・有効化:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

3. 依存をインストール:

```powershell
pip install -r requirements.txt
```

> 補足: プレビュー画像を PNG 出力したい場合は `kaleido` を追加で入れてください（README の `requirements.txt` はオプションを含めています）。

---

## ▶️ 使い方（簡易）
1. アプリ起動:

```powershell
streamlit run app.py
```

2. ブラウザで表示された UI のサイドバーから画像をアップロードし、必要なパラメータ（閾値、近似精度、リサンプリング点数、フーリエ係数、埋め込み手法など）を設定します。

3. タブの使い分け:
- **Global Map:** 複数画像を一度に解析して 2D マップを表示（ホバーでファイル名確認）
- **Single Image:** 単一画像の二値化・輪郭を確認
- **Local Comparison:** 2 枚を選択して「解析開始」を押すと位置合わせとヒートマップを表示。Hausdorff / Chamfer を出力

---

## ⛏ コマンド一覧
- アプリ起動: `streamlit run app.py`
- プレビュー HTML 生成（開発用スクリプトを追加しています）: see `scripts/` (将来的に追加)
- 簡易 HTTP サーバ（プレビュー表示）:

```powershell
python -m http.server --directory previews 8000
# ブラウザで http://localhost:8000/local_comparison.html を開く
```

---

## 📁 ファイル構成（主要）
```
project_root/
├── app.py                 # Streamlit UI
├── src/
│   ├── preprocessor.py    # 画像読み込み・2値化・輪郭抽出・リサンプリング
│   ├── global_features.py # Hu, Fourier, StandardScaler, MDS/TSNE
│   ├── local_analysis.py  # 重心合わせ / Procrustes / ICP, cKDTree 距離計算
│   └── visualizer.py      # Plotly グラフ生成
├── previews/              # 開発中に生成したプレビュー HTML
└── requirements.txt
```

---

## ⚠️ 注意点 / トラブルシューティング
- **輪郭が検出できない場合**: 閾値（Threshold）を 0（Otsu）→ 手動に変更、あるいは画像を反転（白背景／黒背景の違い）して試してください。`num_points` を増やすと局所差が滑らかになりますが計算量が増えます。
- **距離の単位**: 出力はピクセル単位 (px) です。実際の長さ（mm など）に変換するには画像のスケール情報が必要です。
- **パフォーマンス**: 大量の画像や高解像度では特徴抽出・埋め込み（特に t-SNE）が重いです。`@st.cache_data` を活用して再計算を抑制しています。

---

## 🧪 開発・テスト
今後の作業候補（推奨）:
- サンプル画像セットを `examples/` に追加してデモを簡易化
- `pytest` によるユニットテスト（主要関数）を追加

---

## 💡 追加/カスタマイズ案
- 距離を mm 単位へ換算するための設定画面（スケール指定）を追加
- 並列処理 / バッチ処理で大量画像対応

---

## ライセンス / 貢献
貢献は歓迎します。必要に応じて PR を送ってください。

---

**作成済み:** `src/preprocessor.py`, `src/global_features.py`, `src/local_analysis.py`, `src/visualizer.py`, `app.py`

必要なら README をさらに縮める / 日本語表現を整える / サンプル画像とテストを追加します — 希望があれば教えてください。