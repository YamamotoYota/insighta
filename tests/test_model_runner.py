# Copyright (c) 2026 Yota Yamamoto
# SPDX-License-Identifier: MIT

"""Tests for model runner utility helpers."""

from __future__ import annotations

from src.model_runner import (
    default_hyperparam_grid_text,
    estimate_hyperparam_grid_combinations,
    model_label,
    model_requires_target,
    normalize_cv_search_method,
    parse_param_grid_text,
    parse_param_text,
    recommended_randomized_n_iter,
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
