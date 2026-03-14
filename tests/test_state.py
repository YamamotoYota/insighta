# Copyright (c) 2026 Yota Yamamoto
# SPDX-License-Identifier: MIT

"""Tests for shared state defaults and server-side dataset cache."""

from __future__ import annotations

import pandas as pd

from src.server_cache import cache_stats, clear_app_run_cache
from src.state import (
    build_current_data_state,
    build_default_ui_config,
    build_default_view_config,
    dataframe_from_state,
    has_current_dataset,
)
from src.ui_config import visible_graph_keys


def teardown_function() -> None:
    clear_app_run_cache('test-run')



def test_build_default_ui_config_uses_shared_graph_keys() -> None:
    config = build_default_ui_config()
    assert config['visible_graphs'] == visible_graph_keys()
    assert config['show_graphs'] is False
    assert config['apply_standardize'] is True



def test_build_default_view_config_keeps_id_out_of_primary_plot_defaults() -> None:
    view = build_default_view_config(
        {
            'columns': ['id', 'temp', 'pressure', 'machine'],
            'numeric_cols': ['temp', 'pressure'],
        }
    )
    assert view['x_col'] == 'temp'
    assert view['y_col'] == 'pressure'
    assert view['hist_col'] == 'temp'
    assert view['box_col'] == 'temp'



def test_build_current_data_state_uses_server_cache_not_browser_json() -> None:
    df = pd.DataFrame({'id': ['1', '2'], 'temp': [10.0, 12.0]})
    state = build_current_data_state(df, 'uploaded.xlsx', app_run_id='test-run')

    assert state['cache_key'] == 'test-run:current-data'
    assert state['df_json'] is None
    assert has_current_dataset(state) is True
    restored = dataframe_from_state(state)
    assert restored.equals(df)
    assert cache_stats()['entries'] == 1
