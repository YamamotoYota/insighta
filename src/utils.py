# Copyright (c) 2026 Yota Yamamoto
# SPDX-License-Identifier: MIT

"""Shared utility helpers."""

from __future__ import annotations

from typing import Iterable, Sequence

import plotly.graph_objects as go


def pick_column(
    requested: str | None,
    candidates: Sequence[str],
    fallback_index: int = 0,
) -> str | None:
    """Return a valid column from candidates."""
    if requested in candidates:
        return requested
    if not candidates:
        return None
    bounded_index = min(max(fallback_index, 0), len(candidates) - 1)
    return candidates[bounded_index]


def normalize_id_list(values: Iterable[object] | None) -> list[str]:
    """Normalize row IDs into a unique ordered list of strings."""
    if values is None:
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized_value = str(value)
        if normalized_value in seen:
            continue
        seen.add(normalized_value)
        normalized.append(normalized_value)
    return normalized


def format_dataset_meta(source_name: str | None, metadata: dict[str, object]) -> str:
    """Build compact dataset metadata text."""
    row_count = int(metadata.get("row_count", 0))
    column_count = int(metadata.get("column_count", 0))
    numeric_count = len(metadata.get("numeric_cols", []))
    categorical_count = len(metadata.get("categorical_cols", []))
    source_label = source_name or "未読み込み"
    return (
        f"データソース: {source_label} | 行数={row_count} | 列数={column_count} | "
        f"数値列={numeric_count} | カテゴリ列={categorical_count}"
    )


def empty_figure(message: str) -> go.Figure:
    """Return an empty figure with center annotation."""
    fig = go.Figure()
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 14},
            }
        ],
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
    )
    return fig
