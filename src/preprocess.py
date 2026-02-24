# Copyright (c) 2026 Yota Yamamoto
# SPDX-License-Identifier: MIT

"""Preprocessing helpers for user-driven dtype overrides and row filtering."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .data_io import infer_column_types
from .utils import normalize_id_list

SUPPORTED_CAST_TYPES: tuple[str, ...] = ("auto", "float", "int", "string", "category", "datetime")

CAST_TYPE_LABELS: dict[str, str] = {
    "auto": "自動",
    "float": "小数",
    "int": "整数",
    "string": "文字列",
    "category": "カテゴリ",
    "datetime": "日時",
}

DTYPE_KIND_LABELS: dict[str, str] = {
    "float": "小数",
    "int": "整数",
    "string": "文字列",
    "category": "カテゴリ",
    "datetime": "日時",
    "bool": "真偽値",
}


def infer_series_kind(series: pd.Series) -> str:
    """Infer a simple kind label from pandas dtype."""
    if pd.api.types.is_integer_dtype(series.dtype):
        return "int"
    if pd.api.types.is_float_dtype(series.dtype):
        return "float"
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return "datetime"
    if pd.api.types.is_bool_dtype(series.dtype):
        return "bool"
    if pd.api.types.is_categorical_dtype(series.dtype):
        return "category"
    return "string"


def infer_series_kind_label(series: pd.Series) -> str:
    """Return display label for inferred dtype kind."""
    kind = infer_series_kind(series)
    return DTYPE_KIND_LABELS.get(kind, "文字列")


def normalize_type_overrides(
    overrides: dict[str, Any] | None,
    columns: Iterable[str],
    *,
    id_col: str = "id",
) -> dict[str, str]:
    """Normalize raw override map into supported `{column: cast_type}` map."""
    normalized: dict[str, str] = {}
    valid_columns = {col for col in columns if col != id_col}
    for col, cast_type in (overrides or {}).items():
        if col not in valid_columns:
            continue
        cast_name = str(cast_type or "auto").lower()
        if cast_name not in SUPPORTED_CAST_TYPES:
            cast_name = "auto"
        normalized[col] = cast_name
    return normalized


def _cast_to_float(series: pd.Series) -> pd.Series:
    """Cast series to float with invalid values coerced to missing."""
    return pd.to_numeric(series, errors="coerce")


def _cast_to_int(series: pd.Series) -> pd.Series:
    """Cast series to nullable integer with non-integer values coerced to missing."""
    numeric = pd.to_numeric(series, errors="coerce")
    # 整数として表現できない値(例: 1.5)は欠損扱いにする。
    valid_or_missing = numeric.isna() | np.isclose(numeric % 1, 0, equal_nan=True)
    cleaned = numeric.where(valid_or_missing, np.nan)
    return cleaned.round(0).astype("Int64")


def _cast_to_string(series: pd.Series) -> pd.Series:
    """Cast series to pandas string dtype."""
    return series.astype("string")


def _cast_to_category(series: pd.Series) -> pd.Series:
    """Cast series to category via string for stable label handling."""
    return series.astype("string").astype("category")


def _cast_to_datetime(series: pd.Series) -> pd.Series:
    """Cast series to datetime with invalid values coerced to missing."""
    return pd.to_datetime(series, errors="coerce")


def apply_type_overrides(
    df: pd.DataFrame,
    overrides: dict[str, Any] | None,
    *,
    id_col: str = "id",
) -> pd.DataFrame:
    """Apply user-defined dtype overrides and coerce invalid values to missing."""
    working = df.copy()
    normalized = normalize_type_overrides(overrides, working.columns, id_col=id_col)

    for col, cast_type in normalized.items():
        if cast_type == "auto":
            continue
        if cast_type == "float":
            working[col] = _cast_to_float(working[col])
        elif cast_type == "int":
            working[col] = _cast_to_int(working[col])
        elif cast_type == "string":
            working[col] = _cast_to_string(working[col])
        elif cast_type == "category":
            working[col] = _cast_to_category(working[col])
        elif cast_type == "datetime":
            working[col] = _cast_to_datetime(working[col])

    return working


def apply_analysis_filters(
    df: pd.DataFrame,
    *,
    exclude_missing_rows: bool,
    selected_ids: Iterable[object] | None,
    treat_selected_as_missing: bool,
    id_col: str = "id",
) -> pd.DataFrame:
    """Apply analysis filters (selected-as-missing / drop missing rows)."""
    working = df.copy()
    selected = normalize_id_list(selected_ids)

    if treat_selected_as_missing and selected and id_col in working.columns:
        selected_mask = working[id_col].astype(str).isin(set(selected))
        target_cols = [col for col in working.columns if col != id_col]
        if target_cols:
            working.loc[selected_mask, target_cols] = np.nan

    if exclude_missing_rows:
        target_cols = [col for col in working.columns if col != id_col]
        if target_cols:
            working = working.dropna(axis=0, how="any", subset=target_cols)
    return working


def build_runtime_metadata(df: pd.DataFrame, *, id_col: str = "id") -> dict[str, Any]:
    """Build runtime metadata from dataframe."""
    numeric_cols, categorical_cols = infer_column_types(df, id_col=id_col)
    return {
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "columns": list(df.columns),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }

