# Copyright (c) 2026 Yota Yamamoto
# SPDX-License-Identifier: MIT

"""Tests for preprocessing helpers."""

from __future__ import annotations

import pandas as pd

from src.preprocess import apply_analysis_filters, apply_type_overrides


def test_apply_type_overrides_coerces_invalid_values_to_missing() -> None:
    df = pd.DataFrame(
        {
            "id": ["1", "2", "3", "4"],
            "machine": ["A", "B", "A", "C"],
            "value": ["10", "20.5", "x", "30"],
            "timestamp": ["2025-01-01", "invalid", "2025-01-03", None],
        }
    )

    converted = apply_type_overrides(
        df,
        {
            "value": "int",
            "machine": "category",
            "timestamp": "datetime",
        },
    )

    assert str(converted["value"].dtype) == "Int64"
    assert converted["value"].isna().sum() == 2  # "20.5" と "x"
    assert str(converted["machine"].dtype) == "category"
    assert converted["timestamp"].isna().sum() == 2


def test_apply_analysis_filters_selected_as_missing_and_dropna() -> None:
    df = pd.DataFrame(
        {
            "id": ["1", "2", "3"],
            "x": [1.0, 2.0, None],
            "group": ["a", "b", "c"],
        }
    )

    filtered = apply_analysis_filters(
        df,
        exclude_missing_rows=True,
        selected_ids=["1"],
        treat_selected_as_missing=True,
    )

    # id=1 は選択中を欠損扱いで除外、id=3 は元から欠損があるため除外される。
    assert filtered["id"].tolist() == ["2"]

