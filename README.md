# INSIGHTA

INSIGHTA は、データを多様なグラフで描画しながら GUI 上でサンプル（行）を選択し、特徴や差分を探索するためのブラウザ型 EDA / モデリング支援ツールです。  
Plotly Dash を使ってローカルPCで動作します。

## 1. できること（概要）

- データ読み込み
  - CSV (`.csv`)
  - Excel (`.xlsx`, `.xls`, `.xlsm`)
  - SQL DBMS（接続してテーブル一覧取得・SQL実行）
    - SQL Server
    - MySQL
    - SQLite
    - Oracle Database
    - PostgreSQL
- EDA（対話的可視化）
  - 散布図 / ヒストグラム / 箱ひげ図 / 散布図行列
  - データテーブル（フィルタ・ソート・複数行選択）
  - グラフ選択とテーブル選択の連動（リンクドブラッシング）
  - 別ウィンドウでグラフ表示（選択状態を共有）
- データ前処理（GUI）
  - 列型の手動変更（変換不可値は欠損化）
  - 欠損行除外
  - 選択中データを欠損扱い（外れ値扱い）
  - タイムラグ列追加
  - 特徴量追加（四則演算 / `log()` / `exp()`）
- 統計モデリング
  - 学習/テスト分割（ランダム / 層別ランダム / 前後）
  - 学習データ基準の標準化
  - 教師なし: PCA / ICA
  - 回帰: 重回帰 / PLS回帰 / LightGBM / ランダムフォレスト回帰
  - 分類: ロジスティクス回帰 / LightGBM / 決定木 / ランダムフォレスト
  - 学習データCVによるハイパーパラメータ推奨（最終値はユーザーが決定）
  - 学習済みモデルの保存 / 再読込（`joblib`）
- 結果可視化
  - 回帰: 実測値 vs 予測値重ね合わせ / yyプロット / `R2`, `RMSE`, `MAE`
  - 分類: 混同行列 / ROC / `AUC`
  - 変数重要度（モデル別）
- 出力
  - 加工・分析後のデータテーブルを CSV / Excel 出力

## 2. 画面の考え方（ざっくり）

- 上部: データ読み込み、グラフ設定表示、選択クリア
- データテーブル: 常に分析対象データを確認（フィルタ/ソート/選択）
- 分析前処理の設定: 列型・欠損/外れ値扱い
- 統計モデリング共通設定: 特徴量追加 → 分割/標準化
- 統計モデリング実行: モデル選択、CV推奨、学習・評価、モデル保存/再読込
- グラフ設定/表示: 別ウィンドウで開いても選択状態を共有

## 3. 必要環境

- Python 3.11 以上
- Windows（確認環境）
  - 他OSでも理論上動作しますが、SQL Server 接続まわり（ODBC/driver）は環境依存があります

## 4. セットアップ

### 4.1 仮想環境（venv）の場合

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 4.2 Miniforge / conda の場合（推奨）

```bash
conda create -n insighta python=3.11 -y
conda activate insighta
pip install -r requirements.txt
```

補足:
- LightGBM / SHAP / DBドライバ系の導入に時間がかかることがあります。
- SQL DBMS を使わない場合でも `requirements.txt` はそのまま入れて問題ありません。

## 5. 起動方法

```bash
python app.py
```

ブラウザで以下を開きます:

- `http://127.0.0.1:8050`

## 6. 基本的な使い方（最短手順）

### 6.1 ファイルから読み込む（CSV / Excel）

1. `CSV / Excel をアップロード` をクリック
2. ファイルを選択
3. データテーブルに内容が表示されることを確認

### 6.2 DB（SQL）から読み込む

1. `DB（SQL）から読み込む` を開く
2. DBMS を選ぶ（SQL Server / MySQL / SQLite / Oracle / PostgreSQL）
3. 接続情報を入力して `接続`
4. テーブル一覧から選択
5. `SELECT文を作成`
6. 必要なら SQL を編集して `SQLを実行して読み込み`

### 6.3 EDA（グラフ選択）

1. `グラフ設定を表示` を押す
2. 必要なグラフ種別にチェックを入れる
3. 別ウィンドウリンク（散布図 / ヒストグラム / 箱ひげ図 / 散布図行列）を開く
4. グラフ上でドラッグ選択
5. テーブル、サマリ、ランキング、他グラフへ選択状態が反映されることを確認

選択操作:
- 通常ドラッグ: 置換選択
- `Ctrl`（Mac は `Command`）+ ドラッグ: 加算選択
- 全解除: `選択をクリア`

## 7. データ前処理と分析対象の制御

### 7.1 分析前処理の設定

- 列型の手動変更（`float` / `int` / `string` / `category` / `datetime`）
- 変換できない値は欠損値になります
- 欠損行の除外
- 選択中データを欠損扱いにする（外れ値扱い）

### 7.2 統計モデリング共通設定

このセクションは「モデルを学習する前の共通設定」です。

順序の意図:
1. ラグ列追加 / 特徴量追加（新しい変数を作る）
2. 学習/テスト分割
3. 標準化

設定できる内容:
- タイムラグ列追加（例: `temp: 1,-1`）
- 特徴量追加（例: `temp_diff = temp - pressure`）
- 学習/テスト分割方法
- 学習データ比率
- 乱数シード
- 層別列 / 順序列
- 学習データ基準の標準化

## 8. モデリング機能の使い方

### 8.1 実行手順

1. `統計モデリング実行` を開く
2. モデル手法を選択
3. 目的変数（教師ありモデルのみ）を選択
4. 説明変数を選択
5. `CVで推奨ハイパーパラメータを計算` を押す
6. JSON を確認・必要なら編集
7. `モデルを学習して評価` を押す

### 8.2 学習済みモデルの保存 / 再読込

- 保存: `学習済みモデルを保存`
- 再読込: `学習済みモデルを再読込` から `.joblib` を選択

補足:
- 再読込時、保存されている結果パネル（評価指標・図）が復元表示されます。

## 9. 変数重要度（モデル別）

INSIGHTA では、モデルに応じて以下の重要度を表示します。

- 重回帰: 標準化回帰係数（β）
- PLS回帰: VIP
- LightGBM回帰: SHAP値 / Gain importance
- ランダムフォレスト回帰: SHAP値 / Permutation importance
- ロジスティック回帰分類: 標準化回帰係数（β）
- LightGBM分類: SHAP値 / Gain importance
- 決定木分類: SHAP値 / Information Gain / 決定木の図式化
- ランダムフォレスト分類: SHAP値 / Permutation importance
- PCA: Loading の 2乗
- ICA: Mixing matrix

注意:
- SHAP値の計算には `shap` ライブラリが必要です。
- 一部の重要度は前処理後（One-Hot後）の特徴量単位で表示されます。

## 10. データ出力

データテーブル上部から、加工・分析後のデータを出力できます。

- `CSV出力`
- `Excel出力`

出力対象は「現在データテーブルに表示されているデータ（加工後）」です。

## 11. SQL/DB 利用時の注意

DBMSごとに追加ライブラリ/ドライバが必要です（`requirements.txt` に含めていますが、環境側のDBドライバ設定が必要な場合があります）。

- SQL Server
  - Python側: `pyodbc`
  - OS側: ODBC Driver 17/18 for SQL Server
- MySQL
  - `pymysql`
- PostgreSQL
  - `psycopg2-binary`
- Oracle Database
  - `oracledb`
- SQLite
  - Python標準の `sqlite3` を使用（SQLite DB ファイルパスを指定）

接続できない場合の確認ポイント:
- ホスト名 / ポート番号
- DB名（Oracle はサービス名）
- スキーマ名（必要なDBMSのみ）
- ユーザー名 / パスワード
- ネットワーク疎通（VPN, firewall 等）
- SQL Server の ODBC Driver インストール有無

## 12. テスト

```bash
pytest -q
```

## 13. 開発メモ（構成）

主要ファイル:
- `app.py` : エントリポイント
- `src/layout.py` : 画面レイアウト
- `src/callbacks.py` : Dashコールバック
- `src/data_io.py` : CSV/Excel 読み込み
- `src/db_connectors.py` : 複数DBMS接続
- `src/preprocess.py` : 列型変更・欠損/外れ値扱い
- `src/modeling.py` : 分割/標準化/ラグ/特徴量追加
- `src/model_runner.py` : モデル学習・CV推奨・評価・重要度
- `src/figures.py` : EDAグラフ生成
- `src/ranking.py` : 選択群 vs 非選択群の原因候補ランキング

## 14. トラブルシュート（よくあるもの）

- Excel読込で `openpyxl` エラーが出る
  - 実行中の Python 環境に `openpyxl` が入っているか確認してください。
  - `python -c "import sys,openpyxl; print(sys.executable); print(openpyxl.__version__)"`
- ポート `8050` が使用中
  - 既存のアプリを終了するか、ポートを変更して起動してください。
- SHAP重要度が出ない
  - `shap` 未インストール、またはモデル/環境依存で計算失敗の可能性があります（メッセージ表示されます）。
