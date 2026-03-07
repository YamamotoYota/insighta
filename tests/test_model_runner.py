# Copyright (c) 2026 Yota Yamamoto
# SPDX-License-Identifier: MIT

"""Tests for model runner utility helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.model_runner import (
    default_hyperparam_grid_text,
    estimate_hyperparam_grid_combinations,
    model_description,
    model_label,
    model_requires_target,
    normalize_cv_search_method,
    parse_param_grid_text,
    parse_param_text,
    recommended_randomized_n_iter,
    run_model,
)


def test_parse_param_text_valid_json() -> None:
    params, err = parse_param_text('{"n_estimators": 300, "max_depth": 8}')
    assert err is None
    assert params["n_estimators"] == 300
    assert params["max_depth"] == 8


def test_parse_param_text_invalid_json() -> None:
    params, err = parse_param_text("{invalid")
    assert params == {}
    assert err is not None


def test_model_requires_target_flags() -> None:
    assert model_requires_target("reg_linear") is True
    assert model_requires_target("cls_rf") is True
    assert model_requires_target("unsup_pca") is False


def test_parse_param_grid_text_valid_json() -> None:
    grid, err = parse_param_grid_text('{"n_estimators": [100, 300], "max_depth": [null, 8]}')
    assert err is None
    assert grid["n_estimators"] == [100, 300]
    assert grid["max_depth"] == [None, 8]


def test_parse_param_grid_text_scalar_is_accepted() -> None:
    grid, err = parse_param_grid_text('{"solver": "lbfgs"}')
    assert err is None
    assert grid["solver"] == ["lbfgs"]


def test_default_grid_text_and_logistic_label() -> None:
    text = default_hyperparam_grid_text("cls_logistic")
    assert '"C"' in text
    assert "ロジスティック回帰" in model_label("cls_logistic")


def test_model_description_contains_t2q_explanation() -> None:
    description = model_description("unsup_pca_t2q")
    assert "T2" in description
    assert "Q" in description


def test_estimate_hyperparam_grid_combinations() -> None:
    grid = {"a": [1, 2, 3], "b": [True, False]}
    assert estimate_hyperparam_grid_combinations(grid) == 6


def test_recommended_randomized_n_iter_caps_by_combination_count() -> None:
    grid = {"a": [1, 2], "b": [10, 20, 30]}
    assert recommended_randomized_n_iter(grid, default_n_iter=20) == 6
    assert recommended_randomized_n_iter(grid, default_n_iter=4) == 4


def test_normalize_cv_search_method() -> None:
    assert normalize_cv_search_method("grid") == "grid"
    assert normalize_cv_search_method("randomized") == "randomized"
    assert normalize_cv_search_method("random") == "randomized"


def test_unsup_pca_t2q_runs_and_returns_monitoring_outputs() -> None:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "id": range(40),
            "x1": rng.normal(0.0, 1.0, 40),
            "x2": rng.normal(1.0, 0.5, 40),
            "x3": rng.normal(-1.0, 0.8, 40),
        }
    )

    result = run_model(
        df,
        model_key="unsup_pca_t2q",
        feature_cols=["x1", "x2", "x3"],
        target_col=None,
        split_method="random",
        train_ratio=0.7,
        random_seed=123,
        split_stratify_col=None,
        split_order_col=None,
        hyperparams={"n_components": 2},
        selected_ids=["0", "1", "2", "3", "4", "5"],
    )

    assert result["task"] == "unsupervised"
    assert result["model_label"] == "教師なし: PCA異常予兆検知 (T2/Q)"
    assert len(result["figures"]) >= 4
    titles = {item["title"] for item in result["importance_tables"]}
    assert "T2寄与度" in titles
    assert "Q寄与度" in titles


def test_unsup_pca_t2q_scales_limits_above_100_and_uses_selected_samples() -> None:
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "id": range(36),
            "x1": rng.normal(0.0, 1.0, 36),
            "x2": rng.normal(1.0, 0.4, 36),
            "x3": rng.normal(-0.5, 0.7, 36),
        }
    )

    selected_ids = ["0", "1", "2", "3", "4"]
    base = run_model(
        df,
        model_key="unsup_pca_t2q",
        feature_cols=["x1", "x2", "x3"],
        target_col=None,
        split_method="random",
        train_ratio=0.7,
        random_seed=99,
        split_stratify_col=None,
        split_order_col=None,
        hyperparams={
            "n_components": 2,
            "t2_warning_limit_percent": 90,
            "t2_alarm_limit_percent": 95,
            "q_warning_limit_percent": 90,
            "q_alarm_limit_percent": 95,
        },
        selected_ids=selected_ids,
    )
    result = run_model(
        df,
        model_key="unsup_pca_t2q",
        feature_cols=["x1", "x2", "x3"],
        target_col=None,
        split_method="random",
        train_ratio=0.7,
        random_seed=99,
        split_stratify_col=None,
        split_order_col=None,
        hyperparams={
            "n_components": 2,
            "t2_warning_limit_percent": 120,
            "t2_alarm_limit_percent": 180,
            "q_warning_limit_percent": 130,
            "q_alarm_limit_percent": 190,
        },
        selected_ids=selected_ids,
    )

    used = result["used_params"]
    assert used["t2_warning_limit_percent"] == 120
    assert used["t2_alarm_limit_percent"] == 180
    assert used["q_warning_limit_percent"] == 130
    assert used["q_alarm_limit_percent"] == 190

    base_metrics = base["metrics"]
    metrics = result["metrics"]
    base_t2_ref = float(base_metrics.loc[base_metrics["dataset"] == "train", "t2_warning_limit"].iloc[0])
    base_q_ref = float(base_metrics.loc[base_metrics["dataset"] == "train", "q_warning_limit"].iloc[0])
    scaled_t2 = metrics.loc[metrics["dataset"] == "train", ["t2_warning_limit", "t2_alarm_limit"]].iloc[0].astype(float)
    scaled_q = metrics.loc[metrics["dataset"] == "train", ["q_warning_limit", "q_alarm_limit"]].iloc[0].astype(float)

    assert np.isclose(float(scaled_t2["t2_warning_limit"]), base_t2_ref * (120.0 / 90.0))
    assert np.isclose(float(scaled_t2["t2_alarm_limit"]), base_t2_ref * 2.0)
    assert np.isclose(float(scaled_q["q_warning_limit"]), base_q_ref * (130.0 / 90.0))
    assert np.isclose(float(scaled_q["q_alarm_limit"]), base_q_ref * (190.0 / 90.0))

    importance_tables = {item["title"]: item["data"] for item in result["importance_tables"]}
    assert "T2寄与度" in importance_tables
    assert "Q寄与度" in importance_tables
    assert set(importance_tables["T2寄与度"]["basis"]) == {importance_tables["T2寄与度"]["basis"].iloc[0]}
    assert "選択中サンプル" in importance_tables["T2寄与度"]["basis"].iloc[0]
    assert "選択中サンプル" in importance_tables["Q寄与度"]["basis"].iloc[0]

