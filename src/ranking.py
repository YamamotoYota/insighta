# Copyright (c) 2026 Yota Yamamoto
# SPDX-License-Identifier: MIT

"""Cause candidate ranking based on selected vs non-selected groups."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind

from .utils import normalize_id_list

RANKING_COLUMNS = [
    "variable",
    "variable_type",
    "test",
    "p_value",
    "statistic",
    "effect_size",
    "mean_diff",
]


def cohen_d(sample_a: pd.Series, sample_b: pd.Series) -> float:
    """Compute Cohen's d for two samples."""
    a = sample_a.astype(float).to_numpy()
    b = sample_b.astype(float).to_numpy()
    if len(a) < 2 or len(b) < 2:
        return float("nan")

    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled_denom = len(a) + len(b) - 2
    if pooled_denom <= 0:
        return float("nan")

    pooled = np.sqrt(((len(a) - 1) * var_a + (len(b) - 1) * var_b) / pooled_denom)
    if pooled == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def rank_candidate_causes(
    df: pd.DataFrame,
    selected_ids: Iterable[object] | None,
    *,
    id_col: str = "id",
    top_n: int = 10,
) -> tuple[pd.DataFrame, str | None]:
    """Rank candidate explanatory variables with statistical tests."""
    if id_col not in df.columns:
        raise ValueError(f"id column '{id_col}' does not exist.")

    selected = normalize_id_list(selected_ids)
    if not selected:
        empty = pd.DataFrame(columns=RANKING_COLUMNS)
        return empty, "未選択 (No selection)"

    id_series = df[id_col].astype(str)
    selected_mask = id_series.isin(set(selected))
    n_selected = int(selected_mask.sum())
    n_others = int((~selected_mask).sum())

    if n_selected < 2 or n_others < 2:
        empty = pd.DataFrame(columns=RANKING_COLUMNS)
        return (
            empty,
            f"選択数を増やしてください (選択={n_selected}, 非選択={n_others})",
        )

    selected_df = df[selected_mask]
    others_df = df[~selected_mask]

    numeric_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns if col != id_col
    ]
    categorical_cols = [
        col for col in df.columns if col not in numeric_cols and col != id_col
    ]

    rows: list[dict[str, float | str]] = []

    for col in numeric_cols:
        selected_values = selected_df[col].dropna()
        other_values = others_df[col].dropna()
        if len(selected_values) < 2 or len(other_values) < 2:
            continue

        t_stat, p_value = ttest_ind(
            selected_values.to_numpy(),
            other_values.to_numpy(),
            equal_var=False,
            nan_policy="omit",
        )
        rows.append(
            {
                "variable": col,
                "variable_type": "numeric",
                "test": "Welch t-test",
                "p_value": float(p_value),
                "statistic": float(t_stat),
                "effect_size": cohen_d(selected_values, other_values),
                "mean_diff": float(selected_values.mean() - other_values.mean()),
            }
        )

    for col in categorical_cols:
        table = pd.crosstab(
            df[col].fillna("__MISSING__").astype(str),
            selected_mask,
        )
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue

        chi2_stat, p_value, _, _ = chi2_contingency(table)
        rows.append(
            {
                "variable": col,
                "variable_type": "categorical",
                "test": "Chi-square",
                "p_value": float(p_value),
                "statistic": float(chi2_stat),
                "effect_size": float("nan"),
                "mean_diff": float("nan"),
            }
        )

    if not rows:
        empty = pd.DataFrame(columns=RANKING_COLUMNS)
        return empty, "比較可能な列がありません。"

    ranking = pd.DataFrame(rows)
    ranking["abs_effect_size"] = ranking["effect_size"].abs().fillna(0.0)
    ranking["abs_mean_diff"] = ranking["mean_diff"].abs().fillna(0.0)
    ranking = ranking.sort_values(
        by=["p_value", "abs_effect_size", "abs_mean_diff", "statistic"],
        ascending=[True, False, False, False],
    )
    ranking = ranking.head(top_n)[RANKING_COLUMNS].reset_index(drop=True)
    return ranking, None
