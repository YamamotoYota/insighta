<!-- Copyright (c) 2026 Yota Yamamoto -->
<!-- SPDX-License-Identifier: MIT -->

# INSIGHTA 再ビルド手順（Windows / 1ページ版）

この手順書は、別PC環境で `INSIGHTA.exe` を再ビルドするための最短手順です。  
前提は **Windows + Miniforge/conda** です。

## 0. 事前に用意するもの

- Windows PC
- Miniforge（または Miniconda / Anaconda）
- このリポジトリ一式（`app.py`, `src/`, `assets/`, `data/`, `requirements.txt`, `INSIGHTA.spec` を含む）

注意:
- `dist/`, `build/`, `.venv/` は不要です（再生成されます）
- Windows向けEXEは Windows 上でビルドしてください（クロスビルド非推奨）

## 1. 作業フォルダへ移動

例（GitHubから取得する場合）:

```powershell
git clone <YOUR_REPOSITORY_URL> C:\work\eda_rev01
cd C:\work\eda_rev01
```

ZIPで持ち込んだ場合も、展開先フォルダで同様に `cd` してください。

## 2. conda 環境を作成

```powershell
conda create -n insighta python=3.11 -y
```

## 3. 依存関係をインストール

### 3-1. conda activate が使える場合

```powershell
conda activate insighta
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pyinstaller
```

### 3-2. conda activate を使わない場合（推奨代替）

```powershell
conda run -n insighta python -m pip install --upgrade pip
conda run -n insighta python -m pip install -r requirements.txt
conda run -n insighta python -m pip install pyinstaller
```

## 4. （推奨）通常実行で動作確認

```powershell
conda run -n insighta python app.py
```

確認:
- ブラウザで `http://127.0.0.1:8050` が開く
- アプリ画面が表示される

終了:
- アプリ画面の「終了」ボタン
- または `Ctrl + C`

## 5. PyInstaller で EXE をビルド

このプロジェクトでは `.spec` ファイルを使ってビルドします（推奨）。

```powershell
conda run -n insighta python -m PyInstaller --noconfirm --clean INSIGHTA.spec
```

出力先:
- `dist\INSIGHTA.exe`

## 6. 配布用フォルダを更新（必要な場合）

```powershell
Copy-Item -Path dist\INSIGHTA.exe -Destination release\INSIGHTA\INSIGHTA.exe -Force
```

配布用フォルダ:
- `release\INSIGHTA\INSIGHTA.exe`
- `release\INSIGHTA\README.md`

## 7. EXE の起動確認（推奨）

```powershell
.\release\INSIGHTA\INSIGHTA.exe
```

期待される挙動:
- 黒いコンソールは出ない
- 数秒以内にブラウザが自動起動する
- `http://127.0.0.1:8050` に INSIGHTA が表示される

自動で開かない場合:
- ブラウザで `http://127.0.0.1:8050` を手動で開く

## 8. よくあるエラーと対処

### A. `PermissionError` でビルド失敗する

原因:
- `INSIGHTA.exe` が起動中（ロックされている）

対処:
- 実行中の `INSIGHTA.exe` を終了してから再ビルドする

### B. Excel 読み込みで `openpyxl` エラーが出る

原因:
- `openpyxl` を入れた環境と、実際に実行している Python 環境が違う

確認コマンド:

```powershell
conda run -n insighta python -c "import sys, pandas, openpyxl; print(sys.executable); print(pandas.__version__); print(openpyxl.__version__)"
```

### C. `Address already in use (127.0.0.1:8050)` が出る

原因:
- 既に別の INSIGHTA / Python プロセスが 8050 番ポートを使用中

対処:
- 既存プロセスを終了してから再実行する

### D. SQL接続ができない

注意:
- Python ライブラリだけでなく、OS 側ドライバが必要な場合があります
- 例: SQL Server は ODBC Driver が必要

## 9. 公開運用の注意（GitHub）

- `dist/` や `build/` は通常コミットしない（`.gitignore` 済み）
- `release\INSIGHTA\INSIGHTA.exe` も通常コミットしない（`.gitignore` 済み）
- 配布用EXEは **GitHub Releases** に添付する運用を推奨

## 10. 参考（この手順で使う主なファイル）

- `app.py`
- `INSIGHTA.spec`
- `requirements.txt`
- `src/`
- `assets/`
- `data/`
- `release/INSIGHTA/README.md`

