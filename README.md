<!-- Copyright (c) 2026 Yota Yamamoto -->
<!-- SPDX-License-Identifier: MIT -->

## License / ライセンス

このリポジトリは MIT License で公開しています。
著作権者は **Yota Yamamoto** です。

- ライセンス本文: `LICENSE`
- SPDX: `MIT`

# INSIGHTA

INSIGHTA は、データを読み込み、データテーブルと複数グラフを行き来しながらサンプルを選択し、特徴・差分・異常兆候・モデル性能を探索するためのローカル実行型アプリです。  
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
- データテーブルのフィルタ、ソート、複数行選択
- 散布図、ヒストグラム、箱ひげ図、散布図行列
- グラフとテーブルのリンクドブラッシング
- グラフの別ウィンドウ表示
- 選択群と非選択群の比較サマリ、原因候補ランキング

### データ前処理
- 列型の自動判定と手動上書き
- 欠損値行の除外
- 選択中サンプルを欠損扱いにする外れ値除外
- タイムラグ列追加
- 特徴量追加（四則演算、`log()`、`exp()`）
- 学習データ基準の標準化
- 学習用 / テスト用データ分割

### モデリング
- 教師なし学習
  - PCA
  - PCA 異常予兆検知（T2 / Q）
  - ICA
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

### 結果可視化
- 回帰: 実測値と予測値の重ね合わせ、yy プロット、`R2`、`RMSE`、`MAE`
- 分類: 混同行列、ROC、`AUC`
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
│  ├─ state.py
│  └─ utils.py
└─ tests/
```

## 3. 動作環境

### 開発 / ソースコード実行
- Windows 64bit
- Miniforge で作成した conda 環境 `insighta`
- Python 3.11
- ブラウザ: Edge / Chrome 推奨

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

- 全行スクロール表示
- フィルタ / ソート
- 行選択
- 一括選択チェックボックス
- 加工後データの CSV / Excel 出力

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

## 7. データ入力の詳細

### 7.1 CSV / Excel

- CSV: `pandas.read_csv`
- Excel: `pandas.read_excel`
- `id` 列が無い場合は自動付与
- 列型は自動判定され、後から UI で修正可能

### 7.2 SQL

対応 DBMS:
- SQL Server
- MySQL
- SQLite
- Oracle Database
- PostgreSQL

主な流れ:
1. DBMS を選択
2. 接続情報を入力
3. 接続してテーブル一覧を取得
4. SQL を作成または編集
5. 実行して読み込む

注意:
- SQL Server は ODBC Driver 17/18 が必要です。
- Oracle / PostgreSQL / MySQL も環境に応じたクライアント設定が必要な場合があります。

### 7.3 PI データ

取得対象:
- PI DA タグ
- PI AF 属性
- PI AF イベントフレーム

#### PI DA タグ
- 複数タグはカンマ区切り、改行区切り、全角読点区切りに対応
- 複数タグ取得時は `timestamp` を行、タグ名を列にした横持ちテーブルへ整形
- 取得方法は Snapshot / Recorded / Interpolated / Summary に対応

#### PI AF 属性
- `AF サーバー名`
- `AF データベース名`
- `AF エレメント名` またはパス
- `AF 属性名一覧`

複数属性取得時は `timestamp` を行、属性名を列にした横持ちテーブルへ整形します。  
名前入力は全角区切りや簡易パスも許容するよう正規化しています。

#### PI AF イベントフレーム
- `AF サーバー名`
- `AF データベース名`
- `イベントフレームテンプレート名`
- `イベント生成分析名一覧`
- `開始時刻` / `終了時刻`

イベントフレーム取得時は、条件に一致したイベントを行単位で返します。

PI 系の補足:
- `pythonnet` runtime はアプリ内部で `netfx` を強制しています。
- PI DA は取得できるが PI AF で失敗するケースに対応するため、DB 名やエレメント名の解決を柔軟化しています。
- それでも失敗する場合は、AF SDK / PI AF Client のインストール有無、32/64bit 不一致、AF サーバー名と DB 名の指定を確認してください。

## 8. データ前処理

### 列型変更
- `float`
- `int`
- `string`
- `category`
- `datetime`

変換できない値は欠損値になります。

### 欠損 / 外れ値扱い
- 欠損値を含む行を除外
- 選択中サンプルを欠損扱いにして分析対象から除外

### 特徴量追加
例:
- `diff = temp - pressure`
- `ratio = a / b`
- `log_temp = log(temp)`
- `exp_x = exp(x)`

### タイムラグ列追加
例:
- `temp: 1,-1`
- `pressure: 3`

## 9. モデリング

### 9.1 共通フロー
1. 目的変数と説明変数を選ぶ
2. 必要ならラグ列・特徴量を追加する
3. 学習 / テスト分割を設定する
4. 必要なら標準化する
5. `CVで推奨ハイパーパラメータを計算`
6. 推奨値を確認し、最終ハイパーパラメータを調整する
7. `モデルを学習して評価`

### 9.2 分割方法
- ランダム
- 層別ランダム
- 前後

### 9.3 PCA 異常予兆検知（T2 / Q）

このモデルは目的変数を使いません。  
学習データから PCA を作り、主成分空間内の逸脱と再構成残差の逸脱を監視します。

- `T2 統計量`
  - 主成分空間内で通常サンプルからどれだけ離れているか
- `Q 統計量`
  - PCA で再構成できなかった残差の大きさ

#### 主成分数
- `主成分数決定の累積寄与率閾値 (%)` を指定可能
- 初期値は `90`
- CV 推奨値または手動指定の `n_components` を使って学習

#### 管理限界
T2 と Q は別々に設定します。

- `T2 注意管理限界 (%)`
- `T2 異常管理限界 (%)`
- `Q 注意管理限界 (%)`
- `Q 異常管理限界 (%)`

考え方:
- `100未満`: その百分位の閾値を学習データから計算
- `100以上`: 90% 閾値を基準に倍率換算
  - 例: `180` は `90% 閾値の 2 倍`

#### 寄与度
- `T2 寄与度`: PCA 空間での逸脱に効いた変数
- `Q 寄与度`: 再構成残差に効いた変数
- 寄与度の集計対象は、現在 UI で選択中のサンプルだけです
- 選択中サンプルの中で異常管理限界を超えたサンプル群を優先して集計します
- 超過サンプルが無い場合は、選択中で統計量が最大のサンプルを基準にします

## 10. モデル別の主な出力

### 回帰
- 実測値と予測値の重ね合わせ
- yy プロット
- `R2`
- `RMSE`
- `MAE`

### 分類
- 混同行列
- ROC 曲線
- `AUC`

### 重要度
- 重回帰: 標準化回帰係数
- PLS 回帰: VIP
- LightGBM: SHAP、Gain importance
- ランダムフォレスト: SHAP、Permutation importance
- 決定木: Information Gain、図式化
- PCA: Loading の 2 乗
- PCA T2/Q: T2 寄与度、Q 寄与度
- ICA: Mixing matrix

## 11. テスト

```powershell
python -m pytest -q
```

主な対象:
- ランキング計算
- 前処理
- モデリング
- DB 接続ユーティリティ
- PI データ取得補助ロジック

## 12. PyInstaller ビルド

README では、Miniforge の conda 環境 `insighta` を `conda activate insighta` で有効化した状態だけを前提にします。

```powershell
conda activate insighta
python .\build_windows.py
```

このスクリプトは次を自動で実行します。

- 必要なら `pytest` 実行
- 古い `dist/` と `build/` を削除
- `INSIGHTA.spec` を絶対パス指定してビルド
- ビルド失敗時はその場で停止
- `dist/INSIGHTA.exe` を `release/INSIGHTA/INSIGHTA.exe` にコピー

テストを省略する場合:

```powershell
conda activate insighta
python .\build_windows.py --skip-tests
```

`build_windows.ps1` は補助ラッパーですが、README では主手順として扱いません。  
詳細な再ビルド手順は `BUILD_WINDOWS.md` を参照してください。

## 13. GitHub での exe 取り扱い

- `release/INSIGHTA/INSIGHTA.exe` は Git LFS で管理しています
- `dist/` と `build/` はコミット対象外です
- 通常の開発ではソースコード中心で管理し、配布用 exe を更新した時だけ `release/INSIGHTA/INSIGHTA.exe` を更新します

確認コマンド:

```powershell
git lfs ls-files
```

## 14. よくあるトラブル

### Excel が読めない
- 実行中の Python が想定の仮想環境と違う可能性があります
- `conda activate insighta` 後に `python -c "import sys, openpyxl; print(sys.executable); print(openpyxl.__version__)"` で確認してください

### `http://127.0.0.1:8050` が開けない
- 既に別の INSIGHTA が起動中の可能性があります
- 先に起動済みプロセスを止めてください

### PI AF 取得に失敗する
- AF SDK / PI AF Client のインストール確認
- Python / AF SDK の 64bit 一致確認
- AF サーバー名、データベース名、エレメント名またはパスの再確認

### SQL Server に接続できない
- ODBC Driver 17/18 for SQL Server を確認してください

## 15. ライセンス

本リポジトリおよび配布物は MIT License です。  
詳細は `LICENSE` を参照してください。

