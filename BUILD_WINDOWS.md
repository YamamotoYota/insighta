<!-- Copyright (c) 2026 Yota Yamamoto -->
<!-- SPDX-License-Identifier: MIT -->

# INSIGHTA 再ビルド手順（Windows / Miniforge 前提）

この手順書は、Miniforge で作成した conda 環境 `insighta` を使って `INSIGHTA.exe` を再ビルドするための手順です。  
Python 本体は conda で入れ、ライブラリは `requirements.txt` を `pip` で入れる前提に絞っています。

## 0. 前提

- Windows PC
- Miniforge
- リポジトリ一式
- conda 環境名は `insighta`
- Python バージョンは `3.11`

必要な主なファイル:
- `app.py`
- `INSIGHTA.spec`
- `build_windows.py`
- `requirements.txt`
- `src/`
- `assets/`
- `data/`

## 1. リポジトリを配置

```powershell
git clone <YOUR_REPOSITORY_URL> C:\work\INSIGHTA
cd C:\work\INSIGHTA
```

## 2. conda 環境を作成

```powershell
conda create -n insighta python=3.11 -y
```

## 3. 依存関係をインストール

```powershell
conda activate insighta
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pyinstaller
```

確認:

```powershell
python -c "import sys; print(sys.executable)"
```

期待値の例:

```text
C:\Users\<ユーザー名>\miniforge3\envs\insighta\python.exe
```

## 4. 通常実行で動作確認

```powershell
conda activate insighta
python app.py
```

確認項目:
- ブラウザで `http://127.0.0.1:8050` が開く
- INSIGHTA の画面が表示される

## 5. 推奨ビルド方法

```powershell
conda activate insighta
python .\build_windows.py
```

このスクリプトは以下を自動で実行します。

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

## 6. exe の起動確認

```powershell
.\release\INSIGHTA\INSIGHTA.exe
```

期待される挙動:
- 数秒後にブラウザが自動起動する
- `http://127.0.0.1:8050` に INSIGHTA が表示される
- 終了ボタンで停止できる

## 7. GitHub に反映する場合の注意

- `dist/` と `build/` はコミットしません
- `release/INSIGHTA/INSIGHTA.exe` は Git LFS 管理です
- Git LFS 未導入環境では、exe の push に失敗します
- exe の再ビルドだけなら Git LFS は不要です

確認コマンド:

```powershell
git lfs ls-files
```

## 8. よくあるトラブル

### build が成功したように見えるのに exe が更新されていない
- 古い `dist/INSIGHTA.exe` をコピーしている可能性があります
- `build_windows.py` は毎回 `dist/` と `build/` を消してから開始するため、この問題を防げます

### `PermissionError` でビルド失敗する
- 既に `INSIGHTA.exe` が起動中です
- 実行中プロセスを止めてから再ビルドしてください

### Excel 読み込みで `openpyxl` エラー
- `insighta` 環境以外の Python を使っている可能性があります
- `conda activate insighta` 後に次で確認してください

```powershell
python -c "import sys, pandas, openpyxl; print(sys.executable); print(pandas.__version__); print(openpyxl.__version__)"
```

### SQL 接続ができない
- DBMS ごとに追加ドライバが必要です
- SQL Server は ODBC Driver 17/18 を確認してください

### PI AF / PI DA が取得できない
- PI AF Client / AF SDK のインストールと 64bit 一致を確認してください
