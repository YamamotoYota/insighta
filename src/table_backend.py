# Copyright (c) 2026 Yota Yamamoto
# SPDX-License-Identifier: MIT

"""Server-side Dash DataTable filtering, sorting, and paging helpers."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def normalize_table_page_size(value: Any) -> int:
    """Normalize DataTable page size for backend paging."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = 200
    return max(1, min(parsed, 1000))


def normalize_table_page_current(value: Any) -> int:
    """Normalize DataTable page index."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = 0
    return max(parsed, 0)


_TABLE_FILTER_OPERATORS = [
    ["ge ", ">="],
    ["le ", "<="],
    ["lt ", "<"],
    ["gt ", ">"],
    ["ne ", "!="],
    ["eq ", "="],
    ["contains "],
    ["datestartswith "],
]


def split_filter_part(filter_part: str) -> tuple[str | None, str | None, Any]:
    """Split one Dash DataTable filter expression."""
    for operator_type in _TABLE_FILTER_OPERATORS:
        for operator in operator_type:
            if operator not in filter_part:
                continue
            name_part, value_part = filter_part.split(operator, 1)
            name = name_part[name_part.find("{") + 1 : name_part.rfind("}")]
            value = value_part.strip()
            if value and value[0] == value[-1] and value[0] in ("'", '"', '`'):
                value = value[1:-1].replace(f"\\{value[0]}", value[0])
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass
            return name, operator_type[0].strip(), value
    return None, None, None


def apply_filter_part(df: pd.DataFrame, filter_part: str) -> pd.DataFrame:
    """Apply one Dash DataTable filter part to dataframe."""
    column, operator_key, raw_value = split_filter_part(filter_part)
    if not column or column not in df.columns or not operator_key:
        return df

    series = df[column]
    if operator_key == "contains":
        return df.loc[series.astype(str).str.contains(str(raw_value), case=False, na=False)]
    if operator_key == "datestartswith":
        return df.loc[series.astype(str).str.startswith(str(raw_value), na=False)]

    if pd.api.types.is_datetime64_any_dtype(series):
        compare_series = pd.to_datetime(series, errors="coerce")
        compare_value = pd.to_datetime(raw_value, errors="coerce")
    elif pd.api.types.is_numeric_dtype(series):
        compare_series = pd.to_numeric(series, errors="coerce")
        try:
            compare_value = float(raw_value)
        except (TypeError, ValueError):
            return df.iloc[0:0]
    else:
        compare_series = series.astype(str)
        compare_value = str(raw_value)

    if pd.isna(compare_value):
        return df.iloc[0:0]

    operation_map = {
        "eq": compare_series == compare_value,
        "ne": compare_series != compare_value,
        "lt": compare_series < compare_value,
        "le": compare_series <= compare_value,
        "gt": compare_series > compare_value,
        "ge": compare_series >= compare_value,
    }
    mask = operation_map.get(operator_key)
    if mask is None:
        return df
    return df.loc[mask.fillna(False) if hasattr(mask, "fillna") else mask]


def apply_table_filter_query(df: pd.DataFrame, filter_query: str | None) -> pd.DataFrame:
    """Apply Dash DataTable filter_query server-side."""
    query = str(filter_query or "").strip()
    if not query:
        return df

    result = df
    for filter_part in [part.strip() for part in query.split(" && ") if part.strip()]:
        result = apply_filter_part(result, filter_part)
    return result


def apply_table_sort(df: pd.DataFrame, sort_by: list[dict[str, Any]] | None) -> pd.DataFrame:
    """Apply Dash DataTable multi-sort server-side."""
    sort_cols = [item.get("column_id") for item in (sort_by or []) if item.get("column_id") in df.columns]
    if not sort_cols:
        return df
    ascending = [str(item.get("direction", "asc")) != "desc" for item in (sort_by or []) if item.get("column_id") in df.columns]
    return df.sort_values(sort_cols, ascending=ascending, kind="mergesort", na_position="last")


def apply_table_view(df: pd.DataFrame, filter_query: str | None, sort_by: list[dict[str, Any]] | None) -> pd.DataFrame:
    """Build the filtered/sorted DataTable view server-side."""
    filtered = apply_table_filter_query(df, filter_query)
    return apply_table_sort(filtered, sort_by)


def slice_table_page(df: pd.DataFrame, page_current: Any, page_size: Any) -> tuple[pd.DataFrame, int]:
    """Slice one DataTable page and return the page count."""
    size = normalize_table_page_size(page_size)
    current = normalize_table_page_current(page_current)
    page_count = math.ceil(len(df) / size) if len(df) else 0
    start = current * size
    end = start + size
    return df.iloc[start:end].copy(), page_count
