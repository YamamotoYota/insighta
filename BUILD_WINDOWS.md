<!-- Copyright (c) 2026 Yota Yamamoto -->
<!-- SPDX-License-Identifier: MIT -->

# INSIGHTA 再ビルド手順（Windows / 1ページ版）

この手順書は、別 PC 環境で `INSIGHTA.exe` を再ビルドするための最短手順です。  
前提は Windows + Miniforge / conda です。

## 0. 事前に用意するもの

- Windows PC
- Miniforge、Miniconda、または Anaconda
- このリポジトリ一式

必要な主なファイル:
- `app.py`
- `INSIGHTA.spec`
- `requirements.txt`
- `src/`
- `assets/`
- `data/`

補足:
- `dist/` と `build/` は再生成されます。
- Windows 向け exe は Windows 上でビルドしてください。

## 1. リポジトリを配置

例:

```powershell
git clone <YOUR_REPOSITORY_URL> C:\work\INSIGHTA
cd C:\work\INSIGHTA
```

## 2. conda 環境を作成

```powershell
conda create -n insighta python=3.11 -y
```

## 3. 依存関係をインストール

### `conda activate` が使える場合

```powershell
conda activate insighta
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pyinstaller
```

### `conda activate` を避ける場合

```powershell
conda run -n insighta python -m pip install --upgrade pip
conda run -n insighta python -m pip install -r requirements.txt
conda run -n insighta python -m pip install pyinstaller
```

## 4. 通常実行で動作確認

```powershell
conda run -n insighta python app.py
```

確認項目:
- ブラウザで `http://127.0.0.1:8050` が開く
- INSIGHTA の画面が表示される

終了:
- 画面上部の `アプリを終了`
- または `Ctrl + C`

## 5. PyInstaller で exe をビルド

```powershell
conda run -n insighta python -m PyInstaller --noconfirm --clean INSIGHTA.spec
```

生成物:
- `dist\INSIGHTA.exe`

## 6. 配布用フォルダを更新

```powershell
Copy-Item -Path dist\INSIGHTA.exe -Destination release\INSIGHTA\INSIGHTA.exe -Force
```

## 7. exe の起動確認

```powershell
.\release\INSIGHTA\INSIGHTA.exe
```

期待される挙動:
- 数秒後にブラウザが自動起動する
- `http://127.0.0.1:8050` に INSIGHTA が表示される
- 終了ボタンで停止できる

## 8. GitHub に反映する場合の注意

- `dist/` と `build/` はコミットしません
- `release/INSIGHTA/INSIGHTA.exe` は Git LFS 管理です
- Git LFS 未導入環境では、exe の push に失敗します

確認コマンド:

```powershell
git lfs ls-files
```

## 9. よくあるトラブル

### `PermissionError` でビルド失敗する
- 既に `INSIGHTA.exe` が起動中です
- 実行中プロセスを止めてから再ビルドしてください

### `Address already in use (127.0.0.1:8050)`
- 既に別の INSIGHTA が起動中です
- 先に停止してください

### Excel 読み込みで `openpyxl` エラー
- 実行中の Python 環境が違う可能性があります
- 次で確認してください

```powershell
conda run -n insighta python -c "import sys, pandas, openpyxl; print(sys.executable); print(pandas.__version__); print(openpyxl.__version__)"
```

### SQL 接続ができない
- DBMS ごとに追加ドライバが必要です
- SQL Server は ODBC Driver 17/18 を確認してください

### PI AF / PI DA が取得できない
- PI AF Client / AF SDK のインストールと 64bit 一致を確認してください
