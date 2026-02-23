"""Data loading and normalization utilities."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import pandas as pd

ID_COLUMN = "id"
CSV_ENCODING_CANDIDATES: tuple[str, ...] = ("utf-8-sig", "utf-8", "cp932", "shift_jis")
EXCEL_EXTENSIONS: set[str] = {".xlsx", ".xls", ".xlsm"}
CSV_EXTENSIONS: set[str] = {".csv"}


class DataLoadError(ValueError):
    """Raised when dataset parsing fails."""


def _read_csv_bytes_with_fallback(csv_bytes: bytes) -> pd.DataFrame:
    """Read CSV bytes trying common Japanese/UTF encodings."""
    errors: list[str] = []
    for encoding in CSV_ENCODING_CANDIDATES:
        try:
            return pd.read_csv(io.BytesIO(csv_bytes), encoding=encoding)
        except Exception as exc:  # pragma: no cover - parser errors vary by file
            errors.append(f"{encoding}: {type(exc).__name__}")

    attempted = ", ".join(CSV_ENCODING_CANDIDATES)
    detail = "; ".join(errors)
    raise DataLoadError(
        f"CSVの読み込みに失敗しました。試行したエンコーディング: {attempted}. 詳細: {detail}"
    )


def _read_excel_bytes(excel_bytes: bytes, suffix: str) -> pd.DataFrame:
    """Read Excel bytes based on extension."""
    engine = "xlrd" if suffix == ".xls" else "openpyxl"
    try:
        return pd.read_excel(io.BytesIO(excel_bytes), engine=engine)
    except ImportError as exc:
        # pandas.read_excel can raise ImportError for transitive dependencies too.
        raise DataLoadError(
            f"{suffix} の読み込みで ImportError が発生しました ({engine}): {exc}"
        ) from exc
    except Exception as exc:  # pragma: no cover - parser errors vary
        raise DataLoadError(f"Excelファイルの読み込みに失敗しました: {suffix}") from exc


def _decode_upload_contents(contents: str) -> tuple[str, bytes]:
    """Decode Dash upload payload into header and raw bytes."""
    try:
        header, encoded = contents.split(",", 1)
    except ValueError as exc:
        raise DataLoadError("アップロードデータの形式が不正です。") from exc

    if "base64" not in header:
        raise DataLoadError("アップロードデータがbase64形式ではありません。")

    try:
        decoded = base64.b64decode(encoded)
    except Exception as exc:  # pragma: no cover - malformed payload varies
        raise DataLoadError("アップロードデータのデコードに失敗しました。") from exc

    return header, decoded


def load_dataset_from_path(path: str | Path) -> pd.DataFrame:
    """Load dataset from local file path (csv/xlsx/xls/xlsm)."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    try:
        raw = file_path.read_bytes()
    except Exception as exc:  # pragma: no cover - OS errors vary
        raise DataLoadError(f"ファイルを開けませんでした: {path}") from exc

    if suffix in CSV_EXTENSIONS:
        return _read_csv_bytes_with_fallback(raw)
    if suffix in EXCEL_EXTENSIONS:
        return _read_excel_bytes(raw, suffix)

    raise DataLoadError(f"未対応の拡張子です: {suffix}")


def load_dataset_from_upload(contents: str, filename: str | None) -> pd.DataFrame:
    """Load uploaded dataset (csv/xlsx/xls/xlsm)."""
    _header, decoded = _decode_upload_contents(contents)
    suffix = Path(filename or "").suffix.lower()

    if suffix in EXCEL_EXTENSIONS:
        return _read_excel_bytes(decoded, suffix)
    if suffix in CSV_EXTENSIONS:
        return _read_csv_bytes_with_fallback(decoded)

    # 拡張子が無い場合や不正な場合は、まずCSVとして試行する。
    try:
        return _read_csv_bytes_with_fallback(decoded)
    except DataLoadError:
        pass

    for excel_suffix in (".xlsx", ".xlsm", ".xls"):
        try:
            return _read_excel_bytes(decoded, excel_suffix)
        except DataLoadError:
            continue

    raise DataLoadError(
        "アップロードファイルを読み込めませんでした。対応拡張子: csv, xlsx, xls, xlsm"
    )


def load_csv_from_path(path: str | Path) -> pd.DataFrame:
    """Backward-compatible CSV loader."""
    return _read_csv_bytes_with_fallback(Path(path).read_bytes())


def load_csv_from_upload(contents: str) -> pd.DataFrame:
    """Backward-compatible CSV loader from dcc.Upload payload."""
    _header, decoded = _decode_upload_contents(contents)
    return _read_csv_bytes_with_fallback(decoded)


def ensure_id_column(df: pd.DataFrame, id_col: str = ID_COLUMN) -> pd.DataFrame:
    """Ensure a unique, non-null ID column exists."""
    if id_col not in df.columns:
        working = df.copy()
        working.insert(0, id_col, np.arange(1, len(working) + 1).astype(str))
        return working

    working = df.copy()
    raw_id_series = working[id_col]
    id_series = raw_id_series.astype(str)
    if raw_id_series.isna().any() or (not id_series.is_unique):
        working[id_col] = np.arange(1, len(working) + 1).astype(str)
    else:
        working[id_col] = id_series
    return working


def infer_column_types(
    df: pd.DataFrame,
    id_col: str = ID_COLUMN,
) -> tuple[list[str], list[str]]:
    """Infer numeric and categorical columns."""
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != id_col]
    categorical_cols = [
        col
        for col in df.columns
        if col != id_col and col not in numeric_cols
    ]
    return numeric_cols, categorical_cols


def prepare_dataframe(df: pd.DataFrame, id_col: str = ID_COLUMN) -> pd.DataFrame:
    """Validate and normalize dataframe before storing."""
    if df.empty:
        raise DataLoadError("データ行がありません。")

    prepared = ensure_id_column(df, id_col=id_col)
    if prepared.columns.duplicated().any():
        raise DataLoadError("重複した列名があります。")
    return prepared
