# -*- mode: python ; coding: utf-8 -*-
# Copyright (c) 2026 Yota Yamamoto
# SPDX-License-Identifier: MIT
import sys
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)


PROJECT_ROOT = Path(SPECPATH).resolve()


def _existing_data_dir(name: str) -> list[tuple[str, str]]:
    """Return PyInstaller data tuple only when the directory exists."""
    path = PROJECT_ROOT / name
    if not path.exists():
        return []
    return [(str(path), name)]


def _dll_binaries_from(folder: Path) -> list[tuple[str, str]]:
    """Collect DLL files from the active Python environment."""
    if not folder.exists():
        return []
    return [(str(path), ".") for path in folder.glob("*.dll") if path.is_file()]


def _safe_collect_all(package_name: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[str]]:
    """Collect package files without failing when the package is unavailable."""
    try:
        return collect_all(package_name)
    except Exception:
        return [], [], []


def _safe_collect_data_files(package_name: str) -> list[tuple[str, str]]:
    """Collect package data files without hard build failure."""
    try:
        return collect_data_files(package_name)
    except Exception:
        return []


def _safe_collect_dynamic_libs(package_name: str) -> list[tuple[str, str]]:
    """Collect package dynamic libraries without hard build failure."""
    try:
        return collect_dynamic_libs(package_name)
    except Exception:
        return []


def _safe_collect_submodules(package_name: str) -> list[str]:
    """Collect package submodules without hard build failure."""
    try:
        return collect_submodules(package_name)
    except Exception:
        return []


env_root = Path(sys.executable).resolve().parent

datas: list[tuple[str, str]] = []
datas += _existing_data_dir("assets")
datas += _existing_data_dir("data")

binaries: list[tuple[str, str]] = []
binaries += _dll_binaries_from(env_root / "Library" / "bin")
binaries += _dll_binaries_from(env_root / "DLLs")

hiddenimports: list[str] = [
    "pyodbc",
    "pymysql",
    "psycopg2",
    "oracledb",
    "clr",
]

# Optional/runtime-loaded modules used by INSIGHTA features.
hiddenimports += _safe_collect_submodules("openpyxl")
hiddenimports += _safe_collect_submodules("xlrd")
hiddenimports += _safe_collect_submodules("sqlalchemy.dialects")
hiddenimports += _safe_collect_submodules("pythonnet")
datas += _safe_collect_data_files("openpyxl")
binaries += _safe_collect_dynamic_libs("pythonnet")

for package_name in ("lightgbm", "shap"):
    collected_datas, collected_binaries, collected_hiddenimports = _safe_collect_all(package_name)
    datas += collected_datas
    binaries += collected_binaries
    hiddenimports += collected_hiddenimports


a = Analysis(
    [str(PROJECT_ROOT / "app.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=sorted(set(hiddenimports)),
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
    name="INSIGHTA",
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

