<!-- Copyright (c) 2026 Yota Yamamoto -->
<!-- SPDX-License-Identifier: MIT -->

## License / ライセンス

このリポジトリは MIT License で公開しています。  
著作権者は **Yota Yamamoto** です。

- ライセンス本文: `LICENSE`
- SPDX: `MIT`

# INSIGHTA

INSIGHTA は、ローカル PC 上でデータを読み込み、データテーブルと複数グラフを行き来しながらサンプルを選択し、特徴・差分・異常兆候・モデル性能を探索するための Windows 向けアプリです。  
UI は Plotly Dash、配布は PyInstaller に対応しています。

## 1. 何ができるか

### データ読み込み
- CSV
- Excel (`.xlsx`, `.xls`, `.xlsm`)
- SQL データベース
  - SQL Server
  - MySQL
  - SQLite
  - Oracle Database
  - PostgreSQL
- PI データ
  - PI DA タグ
  - PI AF 属性
  - PI AF イベントフレーム

### 探索的データ解析（EDA）
- データテーブルのサーバー側フィルタ、ソート、ページング
- 散布図、ヒストグラム、箱ひげ図、散布図行列
- グラフとテーブルのリンクドブラッシング
- グラフの別ウィンドウ表示
- 選択群と非選択群の比較サマリ、原因候補ランキング

### データ前処理
- 列型の自動判定と手動上書き
- 欠損値行の除外
- 選択中サンプルを欠損扱いにする外れ値除外
- タイムラグ列追加
- 単純移動平均（平滑化）
- 指数移動平均（指数平滑化）
- 季節分解（STL 分解）
- 特徴量追加（四則演算、`log()`、`exp()`）
- 学習データ基準の標準化
- 学習用 / テスト用データ分割

### モデリング
- 教師なし学習
  - PCA
  - PCA 異常予兆検知（T2 / Q）
  - ICA
- 時系列
  - ARIMA
  - SARIMA
  - EWMA
  - CUSUM
- 回帰
  - 重回帰
  - PLS 回帰
  - LightGBM 回帰
  - ランダムフォレスト回帰
- 分類
  - ロジスティック回帰
  - LightGBM 分類
  - 決定木分類
  - ランダムフォレスト分類
- 学習済みモデルの保存 / 再読込
- CV による推奨ハイパーパラメータ計算

補足:
- 時系列モデルでは、目的変数に数値列を選びます。
- 並び順は `前後分割の順序列` を使います。時刻列や連番列を設定してください。

### 結果可視化
- 回帰: 実測値と予測値の重ね合わせ、yy プロット、`R2`、`RMSE`、`MAE`
- 分類: 混同行列、ROC、`AUC`
- 時系列
  - ARIMA / SARIMA: 学習区間とテスト区間の実測値・予測値、yy プロット、`R2`、`RMSE`、`MAE`、`AIC`、`BIC`
  - EWMA: 実測値、EWMA 管理図、信号件数 / 信号率
  - CUSUM: 実測値、CUSUM 管理図、正側 / 負側信号件数
- 重要度 / 寄与度
  - 回帰係数、VIP、Permutation importance、Gain importance、SHAP など
  - PCA: Loading の 2 乗
  - PCA 異常予兆検知: T2 寄与度、Q 寄与度

## 2. リポジトリ構成

```text
INSIGHTA/
├─ app.py
├─ INSIGHTA.spec
├─ README.md
├─ BUILD_WINDOWS.md
├─ LICENSE
├─ requirements.txt
├─ assets/
├─ data/
├─ release/
│  └─ INSIGHTA/
│     ├─ INSIGHTA.exe
│     └─ README.md
├─ src/
│  ├─ callbacks.py
│  ├─ data_io.py
│  ├─ db_connectors.py
│  ├─ figures.py
│  ├─ layout.py
│  ├─ model_runner.py
│  ├─ modeling.py
│  ├─ pi_af_sdk.py
│  ├─ preprocess.py
│  ├─ ranking.py
│  ├─ server_cache.py
│  ├─ state.py
│  ├─ table_backend.py
│  ├─ ui_config.py
│  └─ utils.py
└─ tests/
```

補足:
- `release/INSIGHTA/INSIGHTA.exe` は配布用の実行ファイルです。
- `data/sample.csv` / `data/sample.xlsx` は動作確認用の小さなサンプルです。
- 大容量アップロード検証用のローカル生成ファイルは `.gitignore` で除外しています。

## 3. 動作環境

### 開発 / ソースコード実行
- Windows 64bit
- Miniforge で作成した conda 環境 `insighta`
- Python 3.11
- ブラウザ: Edge / Chrome 推奨
- `statsmodels` は時系列モデルと STL 分解に使用します

### 配布版
- Windows 64bit
- Python の事前インストール不要

補足:
- SQL Server 接続には ODBC Driver が必要です。
- PI AF / PI DA 取得には PI AF Client と AF SDK が必要です。
- PI 系機能は Windows 前提です。

## 4. セットアップ

INSIGHTA の開発・実行・再ビルドは、Miniforge で作成した conda 環境 `insighta` を前提にします。  
Python 本体は conda で入れ、ライブラリは `requirements.txt` を `pip` で入れる想定です。

```powershell
conda create -n insighta python=3.11 -y
conda activate insighta
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pyinstaller
```

### VS Code で Miniforge 環境が切り替わらない場合

PowerShell で `C:/Users/.../Scripts/activate` を直接実行すると、`conda activate` の内部状態が壊れることがあります。  
VS Code のターミナルでは次だけを使ってください。

```powershell
conda activate insighta
```

確認コマンド:

```powershell
python -c "import sys; print(sys.executable)"
```

期待値の例:

```text
C:\Users\<ユーザー名>\miniforge3\envs\insighta\python.exe
```

## 5. 起動方法

### ソースコード版

```powershell
conda activate insighta
python app.py
```

ブラウザで次を開きます。

- `http://127.0.0.1:8050`

### 配布版 exe

- `release/INSIGHTA/INSIGHTA.exe` を実行
- 数秒後にブラウザで `http://127.0.0.1:8050` が自動で開く
- 自動で開かない場合は手動で同 URL を開く

終了方法:
- 画面上部の `アプリを終了` ボタン
- またはソースコード版では `Ctrl + C`

## 6. 基本操作

### 6.1 データ読み込み

画面上部の各入力手段からデータを読み込みます。

- `CSV / Excel をアップロード`
- `DB（SQL）から読み込む`
- `PI（AF SDK）から読み込む`

読み込み後のデータテーブルは、すべての読み込み UI の下に表示されます。

### 6.2 データテーブル

- サーバー側フィルタ / ソート / ページング
- 行選択
- 一括選択チェックボックス
- 加工後データの CSV / Excel 出力

補足:
- 大容量データでは、全件を一括描画せず、現在ページだけをブラウザへ返します。
- 選択状態はページをまたいでも維持されます。
- `全選択` は、現在のフィルタ結果全体を対象に動作します。

### 6.3 グラフ

必要なグラフだけ表示できます。

- 散布図
- ヒストグラム
- 箱ひげ図
- 散布図行列

グラフは別ウィンドウで開きます。  
同期するのはサンプル選択状態だけで、各グラフの軸や列の設定は独立です。

### 6.4 選択操作

- 通常ドラッグ: 置換選択
- `Ctrl` を押しながらドラッグ: 加算選択
- テーブル選択とグラフ選択は相互反映
- `選択をクリア` で全解除

## 7. 大容量データ対応

INSIGHTA は、大きめの CSV / Excel / SQL / PI データに対して、次の方針でブラウザ負荷を抑えています。

- 読み込んだ DataFrame 本体はブラウザ `dcc.Store` ではなく、サーバー側メモリキャッシュに保持
- DataTable はサーバー側フィルタ / ソート / ページング
- 散布図は必要に応じて `Scattergl` を使用
- テーブル描画 callback とグラフ描画 callback を分離し、ページ送りだけで重いグラフ再計算を起こさない構造

注意:
- Excel 読み込みそのものは `pandas/openpyxl` の処理時間が残るため、50MB 級では読込に数分かかる場合があります。
- その後の画面更新については、以前よりかなり軽くなるよう改善しています。

## 8. PI / SQL の補足

### SQL
- SQL Server は ODBC Driver 17/18 for SQL Server を確認してください。
- Oracle / PostgreSQL / MySQL は接続用ドライバやクライアント設定が必要です。

### PI
- PI AF Client / AF SDK が必要です。
- PI DA タグ、PI AF 属性、PI AF イベントフレームに対応しています。
- `pythonnet` はアプリ側で `netfx` を強制する構成にしています。

## 9. 開発メモ

### テスト実行

```powershell
conda activate insighta
python -m pytest -q
```

### 主な設計方針

- `callbacks.py`
  - Dash callback 定義
- `table_backend.py`
  - DataTable のサーバー側フィルタ / ソート / ページング
- `server_cache.py`
  - 大容量データのサーバー側保持
- `figures.py`
  - グラフ生成の純関数寄りヘルパー
- `state.py`
  - `dcc.Store` 用 state とデータ参照
- `model_runner.py`
  - モデル学習と可視化データ生成

## 10. PyInstaller ビルド

通常は次で十分です。

```powershell
conda activate insighta
python .\build_windows.py
```

テスト省略時:

```powershell
conda activate insighta
python .\build_windows.py --skip-tests
```

ビルドの詳細手順は `BUILD_WINDOWS.md` を参照してください。

## 11. 配布物と GitHub 公開

- `release/INSIGHTA/INSIGHTA.exe` は配布用 exe です。
- `dist/` と `build/` はコミットしません。
- `release/INSIGHTA/INSIGHTA.exe` は Git LFS 管理です。
- 秘密情報やローカル生成データは `.gitignore` で除外しています。

## 12. ライセンス

本リポジトリのソースコードおよび配布物は MIT License です。  
著作権者は **Yota Yamamoto** です。  
詳細は `LICENSE` を参照してください。
