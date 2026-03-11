# Copyright (c) 2026 Yota Yamamoto
# SPDX-License-Identifier: MIT

"""Tests for shared state defaults."""

from __future__ import annotations

from src.state import build_default_ui_config, build_default_view_config
from src.ui_config import visible_graph_keys


def test_build_default_ui_config_uses_shared_graph_keys() -> None:
    config = build_default_ui_config()
    assert config["visible_graphs"] == visible_graph_keys()
    assert config["show_graphs"] is False


def test_build_default_view_config_keeps_id_out_of_primary_plot_defaults() -> None:
    view = build_default_view_config(
        {
            "columns": ["id", "temp", "pressure", "machine"],
            "numeric_cols": ["temp", "pressure"],
        }
    )
    assert view["x_col"] == "temp"
    assert view["y_col"] == "pressure"
    assert view["hist_col"] == "temp"
    assert view["box_col"] == "temp"
