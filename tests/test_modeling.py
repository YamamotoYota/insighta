# Copyright (c) 2026 Yota Yamamoto
# SPDX-License-Identifier: MIT

"""Tests for modeling preparation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.modeling import apply_modeling_preparation


def test_modeling_preparation_adds_lag_and_feature_columns() -> None:
    df = pd.DataFrame(
        {
            "id": [str(i) for i in range(1, 7)],
            "time": [1, 2, 3, 4, 5, 6],
            "temp": [10.0, 12.0, 11.0, 13.0, 15.0, 14.0],
            "pressure": [100.0, 101.0, 99.0, 102.0, 103.0, 104.0],
        }
    )

    modeled, meta = apply_modeling_preparation(
        df,
        split_method="sequential",
        train_ratio=0.5,
        random_seed=42,
        stratify_col=None,
        order_col="time",
        standardize=False,
        lag_text="temp: 1,-1",
        feature_text="temp_diff = temp - pressure",
    )

    assert "temp_lag1" in modeled.columns
    assert "temp_lead1" in modeled.columns
    assert "temp_diff" in modeled.columns
    assert meta["train_count"] == 3
    assert meta["test_count"] == 3


def test_modeling_preparation_standardizes_using_train_data() -> None:
    df = pd.DataFrame(
        {
            "id": [str(i) for i in range(1, 9)],
            "time": [1, 2, 3, 4, 5, 6, 7, 8],
            "value": [10.0, 12.0, 14.0, 16.0, 50.0, 52.0, 54.0, 56.0],
        }
    )

    modeled, meta = apply_modeling_preparation(
        df,
        split_method="sequential",
        train_ratio=0.5,
        random_seed=1,
        stratify_col=None,
        order_col="time",
        standardize=True,
        lag_text="",
        feature_text="",
    )

    assert "value" in meta["standardized_cols"]
    train_values = modeled.iloc[:4]["value"]
    assert abs(float(train_values.mean())) < 1e-9
    assert np.isclose(float(train_values.std(ddof=0)), 1.0)
