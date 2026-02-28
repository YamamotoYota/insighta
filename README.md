## License / ライセンス

このリポジトリは MIT License で公開します。
著作権者（Author / Copyright Holder）は **Yota Yamamoto** です。

- ライセンス本文: `LICENSE`
- SPDX: `MIT`

# INSIGHTA

INSIGHTA は、データを多様なグラフで描画しながら GUI 上でサンプル（行）を選択し、特徴や差分を探索するためのブラウザ型 EDA / モデリング支援ツールです。  
Plotly Dash を使ってローカルPCで動作します。

## 1. できること

- データ読み込み
  - CSV (`.csv`)
  - Excel (`.xlsx`, `.xls`, `.xlsm`)
  - SQL DBMS（接続してテーブル一覧取得・SQL実行）
    - SQL Server
    - MySQL
    - SQLite
    - Oracle Database
    - PostgreSQL
  - PI AF SDK（PI DataLink相当）
    - PI DAタグ（Snapshot / Recorded / Interpolated / Summary）
    - PI AF属性データ（エレメント名 + 属性名で取得。PI DAタグ同様の行形式）
    - PI AFイベントフレーム（テンプレート + 対象期間 + イベント生成分析で取得）
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
  - 分類: ロジスティック回帰 / LightGBM / 決定木 / ランダムフォレスト
  - 学習データCVによるハイパーパラメータ推奨（最終値はユーザーが決定）
  - 学習済みモデルの保存 / 再読込（`joblib`）
- 結果可視化
  - 回帰: 実測値 vs 予測値重ね合わせ / yyプロット / `R2`, `RMSE`, `MAE`
  - 分類: 混同行列 / ROC / `AUC`
  - 変数重要度（モデル別）
- 出力
  - 加工・分析後のデータテーブルを CSV / Excel 出力

## 2. 画面の考え方

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

### 4.2 Miniforge / conda の場合

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

## 6. 基本的な使い方

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


### 6.3 PI（AF SDK）から読み込む

1. `PI（AF SDK）から読み込む` を開く
2. `取得対象` を選択
   - `PI DAタグ`
   - `PI AF属性`
   - `PI AFイベントフレーム`
3. 共通項目を入力
   - `PI Data Archiveサーバー名`（PI DAタグ時）
   - `AFサーバー名` / `AFデータベース名`（AF系時）
   - 開始時刻 / 終了時刻
4. 取得対象ごとの必須項目を入力
   - PI DAタグ: `PIタグ一覧`
   - PI AF属性: `AFエレメント名` + `AF属性名一覧`
   - PI AFイベントフレーム: `イベントフレームテンプレート名` + `イベント生成分析名一覧`
5. `PIデータを読み込み` を押す

時刻指定の例:
- `*-1d`（現在から1日前）
- `*`（現在）
- `2026-01-01 00:00:00`

補足:
- PI AF属性は、PI DAタグと同様の行形式（`tag`, `timestamp`, `value` など）で取得されます。
- PI AFイベントフレームは、テンプレート・期間・分析名で絞り込んだイベント行として取得されます。
### 6.4 EDA（グラフ選択）

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

## 11. SQL/DB と PI 利用時の注意

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


PI AF SDK（PI DataLink相当）を使う場合の前提:
- Windows + PI AF Client（PI System Explorer）導入済み
- `OSIsoft.AFSDK` が参照可能
- Python側: `pythonnet`

PI/AF接続で失敗する場合の確認ポイント:
- PI Data Archive サーバー名（または既定サーバー設定）
- AFサーバー名 / AFデータベース名
- PIタグ名、AFエレメント名、AF属性名の存在
- イベントフレームテンプレート名、イベント生成分析名の存在
- 時間指定（`*-1d` / `*` / 固定時刻文字列）の書式
- クライアントPCのAF SDKインストール状態

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
- `src/db_connectors.py` : 複数DBMS接続`n- `src/pi_af_sdk.py` : PI DA / PI AF（属性・イベントフレーム）取得
- `src/preprocess.py` : 列型変更・欠損/外れ値扱い
- `src/modeling.py` : 分割/標準化/ラグ/特徴量追加
- `src/model_runner.py` : モデル学習・CV推奨・評価・重要度
- `src/figures.py` : EDAグラフ生成
- `src/ranking.py` : 選択群 vs 非選択群の原因候補ランキング

## 14. トラブルシュート

- Excel読込で `openpyxl` エラーが出る
  - 実行中の Python 環境に `openpyxl` が入っているか確認してください。
  - `python -c "import sys,openpyxl; print(sys.executable); print(openpyxl.__version__)"`
- ポート `8050` が使用中
  - 既存のアプリを終了するか、ポートを変更して起動してください。
- SHAP重要度が出ない
  - `shap` 未インストール、またはモデル/環境依存で計算失敗の可能性があります（メッセージ表示されます）。

## 15. 配布版ビルド手順（開発者向け / PyInstaller）

このアプリは Tkinter へ作り直さず、Dash（Webアプリ）のまま `PyInstaller` で `INSIGHTA.exe` 化しています。
配布版はローカルPC上で `127.0.0.1:8050` のサーバーを起動し、既定ブラウザを自動で開く方式です。

### 15.1 前提

- Windows 64bit
- Python 3.11 系
- 依存関係インストール済み（`requirements.txt`）
- `INSIGHTA.spec` が存在すること

推奨（今回の実績）:
- Miniforge / conda 環境 `insighta`

### 15.2 PyInstaller のインストール

```bash
conda run -n insighta python -m pip install pyinstaller
```

### 15.3 ビルド実行

```bash
conda run -n insighta python -m PyInstaller --noconfirm --clean INSIGHTA.spec
```

出力先:
- `dist/INSIGHTA.exe`

### 15.4 配布用フォルダの更新

配布用フォルダは `release/INSIGHTA` を使います（利用者向け）。

- `release/INSIGHTA/INSIGHTA.exe`
- `release/INSIGHTA/README.md`（配布専用README）

ビルド後に `dist/INSIGHTA.exe` を `release/INSIGHTA/INSIGHTA.exe` にコピーして更新してください。

例:

```powershell
Copy-Item -Path dist\INSIGHTA.exe -Destination release\INSIGHTA\INSIGHTA.exe -Force
```

### 15.5 起動確認

```bash
.\dist\INSIGHTA.exe
```

- 数秒待つと、既定ブラウザで `http://127.0.0.1:8050` が自動で開きます。
- 開かない場合はブラウザで `http://127.0.0.1:8050` を手動で開いて確認してください。

### 15.6 重要な注意点

1. `INSIGHTA.exe` を起動したまま再ビルドしない
- `dist/INSIGHTA.exe` または `release/INSIGHTA/INSIGHTA.exe` が起動中だと、PyInstaller が `PermissionError` で失敗します。
- 再ビルド前に実行中の `INSIGHTA.exe` を終了してください。

2. conda/miniforge 環境では DLL 同梱が重要
- `INSIGHTA.spec` では conda 環境の `Library/bin` と `DLLs` から DLL を収集する設定にしています。
- これが無いと `_ctypes` などの DLL ロードエラーで起動失敗することがあります。

3. DB接続の実行時要件は別途必要な場合がある
- 例: SQL Server の ODBC Driver（OS側）
- `PyInstaller` で Python ライブラリを同梱しても、OS側ドライバは別途必要なことがあります。

### 15.7 補足

- `app.py`
  - frozen実行時の `assets` 解決（`sys._MEIPASS`）
  - frozen実行時のブラウザ自動起動
  - frozen実行時は `debug=False`
- `INSIGHTA.spec`
  - `assets/` と `data/` の同梱
  - `lightgbm`, `shap` の収集
  - DBドライバ/PI連携 hidden import（`pyodbc`, `pymysql`, `psycopg2`, `oracledb`, `pythonnet`, `clr_loader`）
  - `console=False`（黒いコンソール非表示）





## 16. GitHub同期（`release/INSIGHTA/INSIGHTA.exe`）

`release/INSIGHTA/INSIGHTA.exe` は約 156MB です。  
GitHub の通常 Git には **100MB のハード上限** があるため、通常コミットでは push できません。

このため、本リポジトリでは `INSIGHTA.exe` を Git LFS で管理する前提です（`.gitattributes` 設定済み）。

### 16.1 初回セットアップ（開発PCごと）

```bash
git lfs install
```

### 16.2 追跡設定（確認）

```bash
git lfs track "release/INSIGHTA/INSIGHTA.exe"
```

補足:
- 本リポジトリには `.gitattributes` で同設定を追加済みです。

### 16.3 ビルド後に同期する手順

```bash
git add .gitattributes release/INSIGHTA/INSIGHTA.exe
git commit -m "Update INSIGHTA.exe"
git push
```

### 16.4 注意点

- Git LFS のストレージ容量・転送量には上限があります（利用プラン依存）。
- 容量/転送制限を避ける場合は、実行ファイルは GitHub Releases に添付する運用が現実的です。



