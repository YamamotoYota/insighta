# Copyright (c) 2026 Yota Yamamoto
# SPDX-License-Identifier: MIT

"""Tests for backend DataTable filtering / sorting / paging helpers."""

from __future__ import annotations

import pandas as pd

from src.table_backend import apply_table_view, slice_table_page


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'id': ['1', '2', '3', '4'],
            'temp': [10.0, 15.0, 12.5, 9.5],
            'machine': ['A', 'B', 'A', 'C'],
        }
    )


def test_apply_table_view_numeric_filter_and_sort() -> None:
    df = _sample_df()
    result = apply_table_view(
        df,
        '{temp} ge 10 && {machine} contains "A"',
        [{'column_id': 'temp', 'direction': 'desc'}],
    )
    assert result['id'].tolist() == ['3', '1']


def test_slice_table_page_returns_expected_count() -> None:
    df = pd.DataFrame({'id': [str(i) for i in range(1, 11)]})
    page_df, page_count = slice_table_page(df, page_current=1, page_size=3)
    assert page_count == 4
    assert page_df['id'].tolist() == ['4', '5', '6']

