# Copyright (c) 2026 Yota Yamamoto
# SPDX-License-Identifier: MIT

"""State serialization for dcc.Store."""

from __future__ import annotations

import io
from typing import Any

import pandas as pd

from .data_io import ID_COLUMN, infer_column_types
from .server_cache import has_dataframe, load_dataframe, store_dataframe
from .ui_config import visible_graph_keys


def dataframe_to_json(df: pd.DataFrame) -> str:
    """Serialize dataframe for backward-compatible small-payload support."""
    return df.to_json(orient="split", date_format="iso")


def dataframe_from_json(payload: str) -> pd.DataFrame:
    """Deserialize dataframe from legacy dcc.Store payload."""
    return pd.read_json(io.StringIO(payload), orient="split")


def has_current_dataset(current_data: dict[str, Any] | None) -> bool:
    """Return True when current_data carries a valid dataframe reference."""
    current = current_data or {}
    cache_key = str(current.get('cache_key') or '').strip()
    legacy_json = current.get('df_json')
    return bool((cache_key and has_dataframe(cache_key)) or legacy_json)


def dataframe_from_state(current_data: dict[str, Any]) -> pd.DataFrame:
    """Load dataframe from server-side cache or legacy JSON payload."""
    cache_key = str((current_data or {}).get('cache_key') or '').strip()
    if cache_key:
        try:
            return load_dataframe(cache_key)
        except KeyError as exc:
            raise ValueError('サーバー側データキャッシュが失われました。データを再度読み込んでください。') from exc

    payload = (current_data or {}).get('df_json')
    if payload:
        return dataframe_from_json(payload)

    raise ValueError('データが読み込まれていません。')


def build_empty_data_state(app_run_id: str | None = None) -> dict[str, Any]:
    """Build empty state before any CSV upload."""
    return {
        'source_name': None,
        'cache_key': None,
        'df_json': None,
        'app_run_id': app_run_id,
        'metadata': {
            'row_count': 0,
            'column_count': 0,
            'columns': [],
            'numeric_cols': [],
            'categorical_cols': [],
        },
    }


def build_current_data_state(
    df: pd.DataFrame,
    source_name: str,
    app_run_id: str | None = None,
) -> dict[str, Any]:
    """Build current_data payload with server-side dataframe cache."""
    numeric_cols, categorical_cols = infer_column_types(df)
    metadata = {
        'row_count': int(len(df)),
        'column_count': int(df.shape[1]),
        'columns': list(df.columns),
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
    }
    cache_key = store_dataframe(df, app_run_id)
    return {
        'source_name': source_name,
        'cache_key': cache_key,
        'df_json': None,
        'app_run_id': app_run_id,
        'metadata': metadata,
    }


def build_default_ui_config() -> dict[str, Any]:
    """Create default shared UI config (window-independent settings)."""
    return {
        'visible_graphs': visible_graph_keys(),
        'show_graphs': False,
        'type_overrides': {},
        'exclude_missing_rows': False,
        'treat_selected_as_missing': False,
        'split_method': 'random',
        'train_ratio': 0.8,
        'split_seed': 42,
        'split_stratify_col': None,
        'split_order_col': None,
        'apply_standardize': False,
        'lag_config_text': '',
        'feature_config_text': '',
    }


def build_default_view_config(metadata: dict[str, Any]) -> dict[str, Any]:
    """Create default per-window graph config from metadata."""
    numeric_cols: list[str] = list(metadata.get('numeric_cols', []))
    all_cols: list[str] = list(metadata.get('columns', []))

    non_id_cols = [col for col in all_cols if col != ID_COLUMN]
    plot_cols = non_id_cols or all_cols

    x_col = plot_cols[0] if len(plot_cols) >= 1 else None
    y_col = plot_cols[1] if len(plot_cols) >= 2 else x_col
    hist_col = plot_cols[0] if len(plot_cols) >= 1 else None
    box_col = numeric_cols[0] if len(numeric_cols) >= 1 else hist_col
    return {
        'x_col': x_col,
        'y_col': y_col,
        'hist_col': hist_col,
        'box_col': box_col,
        'matrix_cols': numeric_cols[: min(len(numeric_cols), 4)],
    }
