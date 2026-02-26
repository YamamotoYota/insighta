# Copyright (c) 2026 Yota Yamamoto
# SPDX-License-Identifier: MIT

"""Tests for ranking functions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ranking import rank_candidate_causes


def test_rank_candidate_causes_numeric_and_categorical() -> None:
    df = pd.DataFrame(
        {
            "id": [str(i) for i in range(1, 13)],
            "sensor_x": [100, 98, 102, 97, 101, 99, 10, 11, 9, 8, 12, 10],
            "sensor_y": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "machine": ["A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B"],
            "shift": ["day", "day", "night", "night", "day", "night", "day", "night", "day", "night", "day", "night"],
        }
    )
    selected_ids = ["1", "2", "3", "4", "5", "6"]

    ranking, message = rank_candidate_causes(df, selected_ids, top_n=10)

    assert message is None
    assert not ranking.empty
    assert "sensor_x" in ranking["variable"].tolist()
    assert "machine" in ranking["variable"].tolist()

    sensor_row = ranking[ranking["variable"] == "sensor_x"].iloc[0]
    assert sensor_row["variable_type"] == "numeric"
    assert np.isfinite(sensor_row["effect_size"])


def test_rank_candidate_causes_small_selection() -> None:
    df = pd.DataFrame(
        {
            "id": ["1", "2", "3", "4"],
            "value": [1.0, 2.0, 3.0, 4.0],
            "category": ["a", "a", "b", "b"],
        }
    )

    ranking, message = rank_candidate_causes(df, ["1"])

    assert ranking.empty
    assert message is not None
    assert "選択数を増やしてください" in message
