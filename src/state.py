# Copyright (c) 2026 Yota Yamamoto
# SPDX-License-Identifier: MIT

"""State serialization for dcc.Store."""

from __future__ import annotations

import io
from typing import Any

import pandas as pd

from .data_io import infer_column_types


def dataframe_to_json(df: pd.DataFrame) -> str:
    """Serialize dataframe for dcc.Store.

    Note:
        This MVP stores dataframe JSON directly in dcc.Store.
        For larger datasets, this can be replaced with server-side cache and
        this payload can carry only a cache key.
    """
    return df.to_json(orient="split", date_format="iso")


def dataframe_from_json(payload: str) -> pd.DataFrame:
    """Deserialize dataframe from dcc.Store payload."""
    return pd.read_json(io.StringIO(payload), orient="split")


def build_empty_data_state(app_run_id: str | None = None) -> dict[str, Any]:
    """Build empty state before any CSV upload."""
    return {
        "source_name": None,
        "df_json": None,
        "app_run_id": app_run_id,
        "metadata": {
            "row_count": 0,
            "column_count": 0,
            "columns": [],
            "numeric_cols": [],
            "categorical_cols": [],
        },
    }


def build_current_data_state(
    df: pd.DataFrame,
    source_name: str,
    app_run_id: str | None = None,
) -> dict[str, Any]:
    """Build current_data payload."""
    numeric_cols, categorical_cols = infer_column_types(df)
    metadata = {
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "columns": list(df.columns),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }
    return {
        "source_name": source_name,
        "df_json": dataframe_to_json(df),
        "app_run_id": app_run_id,
        "metadata": metadata,
    }


def build_default_ui_config(metadata: dict[str, Any]) -> dict[str, Any]:
    """Create default shared UI config (window-independent settings)."""
    return {
        "visible_graphs": ["scatter", "hist", "box", "matrix"],
        "show_graphs": False,
        "type_overrides": {},
        "exclude_missing_rows": False,
        "treat_selected_as_missing": False,
        "split_method": "random",
        "train_ratio": 0.8,
        "split_seed": 42,
        "split_stratify_col": None,
        "split_order_col": None,
        "apply_standardize": False,
        "lag_config_text": "",
        "feature_config_text": "",
    }


def build_default_view_config(metadata: dict[str, Any]) -> dict[str, Any]:
    """Create default per-window graph config from metadata."""
    numeric_cols: list[str] = list(metadata.get("numeric_cols", []))
    all_cols: list[str] = list(metadata.get("columns", []))

    non_id_cols = [col for col in all_cols if col != "id"]
    plot_cols = non_id_cols or all_cols

    x_col = plot_cols[0] if len(plot_cols) >= 1 else None
    y_col = plot_cols[1] if len(plot_cols) >= 2 else x_col
    hist_col = plot_cols[0] if len(plot_cols) >= 1 else None
    box_col = numeric_cols[0] if len(numeric_cols) >= 1 else hist_col
    return {
        "x_col": x_col,
        "y_col": y_col,
        "hist_col": hist_col,
        "box_col": box_col,
        "matrix_cols": numeric_cols[: min(len(numeric_cols), 4)],
    }
