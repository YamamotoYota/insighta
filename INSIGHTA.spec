# -*- mode: python ; coding: utf-8 -*-
# Copyright (c) 2026 Yota Yamamoto
# SPDX-License-Identifier: MIT
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all


def _dll_binaries_from(folder: str) -> list[tuple[str, str]]:
    """Collect dll files from folder for conda environment compatibility."""
    path = Path(folder)
    if not path.exists():
        return []
    return [(str(p), '.') for p in path.glob('*.dll') if p.is_file()]

env_root = Path(sys.executable).resolve().parent

datas = [('assets', 'assets'), ('data', 'data')]
binaries = []
binaries += _dll_binaries_from(str(env_root / 'Library' / 'bin'))
binaries += _dll_binaries_from(str(env_root / 'DLLs'))
hiddenimports = ['pyodbc', 'pymysql', 'psycopg2', 'oracledb']
tmp_ret = collect_all('lightgbm')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('shap')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='INSIGHTA',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
