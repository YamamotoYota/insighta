# Copyright (c) 2026 Yota Yamamoto
# SPDX-License-Identifier: MIT

"""Windows build helper for INSIGHTA."""

from __future__ import annotations

import argparse
import importlib.util
import platform
import shutil
import struct
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_command(command: list[str], cwd: Path) -> None:
    """Run command and fail fast on non-zero exit."""
    print("$", " ".join(command))
    subprocess.run(command, cwd=str(cwd), check=True)


def verify_build_environment() -> None:
    """Validate Python runtime used for Windows packaging."""
    if sys.platform != "win32":
        raise RuntimeError("Windows 向けビルドは Windows 上で実行してください。")
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"Python 3.11 系でビルドしてください。現在: {platform.python_version()}")
    if struct.calcsize("P") * 8 != 64:
        raise RuntimeError("配布用 exe は 64bit Python 環境でビルドしてください。")


def warn_missing_optional_modules() -> None:
    """Print informative warnings for optional features not included in the build env."""
    if importlib.util.find_spec("pythonnet") is None:
        print("WARNING: pythonnet が見つかりません。今回の exe では PI 機能を使用できない可能性があります。")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build INSIGHTA executable on Windows.")
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest before build.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    spec_path = repo_root / "INSIGHTA.spec"
    dist_dir = repo_root / "dist"
    build_dir = repo_root / "build"
    dist_exe = dist_dir / "INSIGHTA.exe"
    release_dir = repo_root / "release" / "INSIGHTA"
    release_exe = release_dir / "INSIGHTA.exe"

    print("[1/4] Python executable")
    print(sys.executable)
    verify_build_environment()
    warn_missing_optional_modules()

    if not args.skip_tests:
        print("[2/4] Run tests")
        run_command([sys.executable, "-m", "pytest", "-q"], cwd=repo_root)

    for path in (dist_dir, build_dir):
        if path.exists():
            shutil.rmtree(path)

    print("[3/4] Build executable")
    run_command([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", str(spec_path)], cwd=repo_root)

    if not dist_exe.exists():
        raise FileNotFoundError("Build finished but dist/INSIGHTA.exe was not created.")

    release_dir.mkdir(parents=True, exist_ok=True)

    print("[4/4] Copy to release folder")
    shutil.copy2(dist_exe, release_exe)

    print("\nBuild completed.")
    for path in (dist_exe, release_exe):
        updated = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"- {path} | size={path.stat().st_size} | updated={updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
