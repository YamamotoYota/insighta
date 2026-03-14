<!-- Copyright (c) 2026 Yota Yamamoto -->
<!-- SPDX-License-Identifier: MIT -->

# INSIGHTA 再ビルド手順（Windows / 再現性重視）

この手順書は、別 PC でも同じ手順で `INSIGHTA.exe` を再ビルドできるようにした手順です。  
推奨は `environment.yml` を使う方法で、`venv` / `conda` のどちらでも実施できます。

## 0. 前提

- Windows PC
- リポジトリ一式
- Python バージョンは `3.11`
- 64bit Python

必要な主なファイル:
- `app.py`
- `INSIGHTA.spec`
- `build_windows.py`
- `requirements.txt`
- `requirements-build.txt`
- `requirements-optional-pi.txt`
- `environment.yml`
- `src/`
- `assets/`

補足:
- 時系列モデル（ARIMA / SARIMA / EWMA / CUSUM）と STL 分解は `statsmodels` を使います。
- PI 機能を使う場合だけ `pythonnet` と PI AF Client / AF SDK が追加で必要です。

## 1. リポジトリを配置

```powershell
git clone <YOUR_REPOSITORY_URL> C:\work\INSIGHTA
cd C:\work\INSIGHTA
```

## 2. 推奨: `environment.yml` から環境を作成

```powershell
conda env create -f environment.yml
conda activate insighta
```

## 3. 代替: `venv` または既存 Python に入れる場合

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-build.txt
```

PI 機能も配布物に含めたい場合は追加で次を実行します。

```powershell
python -m pip install -r requirements-optional-pi.txt
```

確認:

```powershell
python -c "import sys; print(sys.executable)"
```

期待値の例:

```text
C:\path\to\python.exe
```

## 4. 通常実行で動作確認

```powershell
python app.py
```

確認項目:
- ブラウザで既定の `http://127.0.0.1:8050` が開く
- INSIGHTA の画面が表示される
- CSV / Excel / SQL / PI のいずれかでデータ読み込みができる

必要なら次の環境変数で接続先を変えられます。

- `INSIGHTA_HOST`
- `INSIGHTA_PORT`

## 5. 推奨ビルド方法

```powershell
python .\build_windows.py
```

このスクリプトは以下を自動で実行します。

- `pytest` 実行
- 古い `dist/` と `build/` を削除
- `INSIGHTA.spec` を使ってビルド
- ビルド失敗時はその場で停止
- `dist/INSIGHTA.exe` を `release/INSIGHTA/INSIGHTA.exe` にコピー

テストを省略する場合:

```powershell
python .\build_windows.py --skip-tests
```

## 6. exe の起動確認

```powershell
.\release\INSIGHTA\INSIGHTA.exe
```

期待される挙動:
- 数秒後にブラウザが自動起動する
- 既定の `http://127.0.0.1:8050` に INSIGHTA が表示される
- 終了ボタンで停止できる

## 7. 最近の大容量データ対応について

現在の INSIGHTA は、大容量データで次の構成を取っています。

- 読み込んだ DataFrame 本体はサーバー側メモリキャッシュに保持
- DataTable はサーバー側フィルタ / ソート / ページング
- テーブル描画とグラフ描画を分離

そのため、ビルド後の動作確認では次も見ておくと安全です。

- Excel 読み込み後にテーブルが表示される
- DataTable のページ送りが動く
- ページ送りでアプリ全体が固まりにくい
- グラフ選択とテーブル選択が相互反映する

## 8. GitHub に反映する場合の注意

- `dist/` と `build/` はコミットしません
- `release/INSIGHTA/INSIGHTA.exe` は Git LFS 管理です
- Git LFS 未導入環境では、exe の push に失敗します
- exe のローカル再ビルド自体は Git LFS なしでも可能です

確認コマンド:

```powershell
git lfs ls-files
```

## 9. よくあるトラブル

### build が成功したように見えるのに exe が更新されていない
- 古い `dist/INSIGHTA.exe` をコピーしている可能性があります
- `build_windows.py` は毎回 `dist/` と `build/` を消してから開始するため、この問題を防げます

### `PermissionError` でビルド失敗する
- 既に `INSIGHTA.exe` が起動中です
- 実行中プロセスを止めてから再ビルドしてください

### Excel 読み込みで `openpyxl` エラー
- 想定外の Python 環境を使っている可能性があります
- 有効化した環境で次を確認してください

```powershell
python -c "import sys, pandas, openpyxl; print(sys.executable); print(pandas.__version__); print(openpyxl.__version__)"
```

### SQL 接続ができない
- DBMS ごとに追加ドライバが必要です
- SQL Server は ODBC Driver 17/18 を確認してください

### PI AF / PI DA が取得できない
- `pythonnet` が入っていないと PI 機能は使えません
- PI AF Client / AF SDK のインストールと 64bit 一致を確認してください
- PI 機能は Windows 前提です
