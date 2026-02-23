"""Common utilities for statistical modeling preparation."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

SPLIT_METHODS: tuple[str, ...] = ("random", "stratified_random", "sequential")
DEFAULT_SPLIT_METHOD = "random"
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_RANDOM_SEED = 42


def normalize_split_method(value: Any) -> str:
    """Normalize split method string."""
    method = str(value or DEFAULT_SPLIT_METHOD).strip().lower()
    if method not in SPLIT_METHODS:
        return DEFAULT_SPLIT_METHOD
    return method


def normalize_train_ratio(value: Any) -> float:
    """Clamp train ratio into a safe range."""
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return DEFAULT_TRAIN_RATIO
    return min(max(ratio, 0.05), 0.95)


def normalize_random_seed(value: Any) -> int:
    """Normalize random seed."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return DEFAULT_RANDOM_SEED


def parse_lag_config_text(
    text: str | None,
    valid_columns: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse lag configuration text.

    Format:
        column_name: 1,-1,2
    """
    if not text:
        return [], []

    valid_set = set(valid_columns)
    configs: list[dict[str, Any]] = []
    warnings: list[str] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if ":" not in raw:
            warnings.append(f"ラグ設定 {line_no}行目: ':' 区切りがありません。")
            continue

        col_part, lags_part = raw.split(":", 1)
        col = col_part.strip()
        if col not in valid_set:
            warnings.append(f"ラグ設定 {line_no}行目: 列 '{col}' が存在しません。")
            continue

        lags: list[int] = []
        for token in lags_part.split(","):
            t = token.strip()
            if not t:
                continue
            try:
                lag = int(t)
            except ValueError:
                warnings.append(f"ラグ設定 {line_no}行目: '{t}' は整数ではありません。")
                continue
            if lag == 0:
                warnings.append(f"ラグ設定 {line_no}行目: 0 は無視されます。")
                continue
            if lag not in lags:
                lags.append(lag)

        if not lags:
            warnings.append(f"ラグ設定 {line_no}行目: 有効なラグがありません。")
            continue
        configs.append({"column": col, "lags": lags})
    return configs, warnings


def parse_feature_config_text(text: str | None) -> tuple[list[dict[str, str]], list[str]]:
    """Parse feature expression text.

    Format:
        new_col = expression
    """
    if not text:
        return [], []

    features: list[dict[str, str]] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for line_no, line in enumerate(text.splitlines(), start=1):
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if "=" not in raw:
            warnings.append(f"特徴量設定 {line_no}行目: '=' 区切りがありません。")
            continue
        out_col, expr = raw.split("=", 1)
        out_name = out_col.strip()
        expression = expr.strip()
        if not out_name or not expression:
            warnings.append(f"特徴量設定 {line_no}行目: 出力列名または式が空です。")
            continue
        if out_name in seen:
            warnings.append(f"特徴量設定 {line_no}行目: 出力列 '{out_name}' が重複しています。後勝ちで適用します。")
            features = [item for item in features if item["name"] != out_name]
        seen.add(out_name)
        features.append({"name": out_name, "expression": expression})
    return features, warnings


def apply_lag_features(
    df: pd.DataFrame,
    lag_configs: list[dict[str, Any]],
    *,
    order_col: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Add lag/lead columns and return added column names."""
    if not lag_configs:
        return df.copy(), []

    if order_col and order_col in df.columns:
        ordered_index = df.sort_values(order_col, kind="mergesort").index
    else:
        ordered_index = df.index

    ordered = df.loc[ordered_index].copy()
    added_cols: list[str] = []
    for config in lag_configs:
        col = str(config.get("column"))
        if col not in ordered.columns:
            continue
        for lag in config.get("lags", []):
            if not isinstance(lag, int) or lag == 0:
                continue
            new_col = f"{col}_lag{lag}" if lag > 0 else f"{col}_lead{abs(lag)}"
            ordered[new_col] = ordered[col].shift(lag)
            added_cols.append(new_col)

    restored = ordered.loc[df.index].copy()
    return restored, added_cols


def _normalize_expression(expression: str) -> str:
    """Normalize expression text for pd.eval."""
    expr = expression.strip()
    expr = re.sub(r"(?<![\w.])log\(", "np.log(", expr)
    expr = re.sub(r"(?<![\w.])ln\(", "np.log(", expr)
    expr = re.sub(r"(?<![\w.])exp\(", "np.exp(", expr)
    return expr


def _is_expression_safe(expression: str) -> bool:
    """Basic guard for expression safety."""
    lowered = expression.lower()
    blocked_tokens = ("__", "import", "lambda", "exec", "eval", "subprocess", "os.", "sys.")
    return not any(token in lowered for token in blocked_tokens)


def apply_feature_expressions(
    df: pd.DataFrame,
    features: list[dict[str, str]],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Evaluate feature expressions and add columns."""
    if not features:
        return df.copy(), [], []

    working = df.copy()
    added_cols: list[str] = []
    warnings: list[str] = []

    for feature in features:
        out_col = feature["name"]
        expression = _normalize_expression(feature["expression"])
        if not _is_expression_safe(expression):
            warnings.append(f"特徴量 '{out_col}': 式に許可されないトークンが含まれています。")
            continue

        local_vars = {col: working[col] for col in working.columns}
        local_vars["np"] = np
        try:
            result = pd.eval(expression, local_dict=local_vars, engine="python")
        except Exception as exc:
            warnings.append(f"特徴量 '{out_col}': 計算に失敗しました ({exc})")
            continue

        if np.isscalar(result):
            working[out_col] = result
        else:
            series = pd.Series(result, index=working.index)
            if len(series) != len(working):
                warnings.append(f"特徴量 '{out_col}': 結果の長さが一致しません。")
                continue
            working[out_col] = series
        added_cols.append(out_col)

    return working, added_cols, warnings


def split_train_test_indices(
    df: pd.DataFrame,
    *,
    method: str,
    train_ratio: float,
    random_seed: int,
    stratify_col: str | None = None,
    order_col: str | None = None,
) -> tuple[pd.Index, pd.Index, list[str]]:
    """Create train/test indices by selected split method."""
    warnings: list[str] = []
    n_rows = len(df)
    if n_rows == 0:
        return df.index, df.index[:0], warnings
    if n_rows == 1:
        return df.index, df.index[:0], warnings

    train_size = int(round(n_rows * train_ratio))
    train_size = min(max(train_size, 1), n_rows - 1)

    rng = np.random.default_rng(random_seed)
    normalized_method = normalize_split_method(method)

    if normalized_method == "sequential":
        if order_col and order_col in df.columns:
            sorted_index = df.sort_values(order_col, kind="mergesort").index
        else:
            if order_col:
                warnings.append(f"前後分割: 列 '{order_col}' が見つからないため現在順序を使用します。")
            sorted_index = df.index
        train_idx = sorted_index[:train_size]
        test_idx = sorted_index[train_size:]
        return train_idx, test_idx, warnings

    if normalized_method == "stratified_random":
        if not stratify_col or stratify_col not in df.columns:
            warnings.append("層別ランダム分割: 層別列が未設定または不正のためランダム分割にフォールバックしました。")
        else:
            strata = df[stratify_col].fillna("__MISSING__").astype(str)
            train_parts: list[np.ndarray] = []
            test_parts: list[np.ndarray] = []
            for _, group_idx in strata.groupby(strata).groups.items():
                group_index = np.array(list(group_idx))
                perm = rng.permutation(group_index)
                if len(perm) <= 1:
                    train_parts.append(perm)
                    continue
                group_train = int(round(len(perm) * train_ratio))
                group_train = min(max(group_train, 1), len(perm) - 1)
                train_parts.append(perm[:group_train])
                test_parts.append(perm[group_train:])

            train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=object)
            test_idx = np.concatenate(test_parts) if test_parts else np.array([], dtype=object)
            return pd.Index(train_idx), pd.Index(test_idx), warnings

    permuted = rng.permutation(np.array(df.index))
    train_idx = permuted[:train_size]
    test_idx = permuted[train_size:]
    return pd.Index(train_idx), pd.Index(test_idx), warnings


def apply_standardization(
    df: pd.DataFrame,
    *,
    train_idx: pd.Index,
) -> tuple[pd.DataFrame, list[str]]:
    """Standardize numeric columns using train statistics."""
    standardized = df.copy()
    train_df = standardized.loc[train_idx]
    numeric_cols = list(train_df.select_dtypes(include=[np.number]).columns)
    applied_cols: list[str] = []

    for col in numeric_cols:
        mean_val = train_df[col].mean(skipna=True)
        std_val = train_df[col].std(skipna=True, ddof=0)
        if pd.isna(std_val) or std_val == 0:
            std_val = 1.0
        standardized[col] = (standardized[col] - mean_val) / std_val
        applied_cols.append(col)

    return standardized, applied_cols


def apply_modeling_preparation(
    df: pd.DataFrame,
    *,
    split_method: str,
    train_ratio: float,
    random_seed: int,
    stratify_col: str | None,
    order_col: str | None,
    standardize: bool,
    lag_text: str | None,
    feature_text: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply common modeling preparation steps and return metadata."""
    working = df.copy()
    metadata: dict[str, Any] = {
        "split_method": normalize_split_method(split_method),
        "train_ratio": normalize_train_ratio(train_ratio),
        "random_seed": normalize_random_seed(random_seed),
        "stratify_col": stratify_col if stratify_col in working.columns else None,
        "order_col": order_col if order_col in working.columns else None,
        "standardized_cols": [],
        "lag_cols": [],
        "feature_cols": [],
        "warnings": [],
        "train_count": 0,
        "test_count": 0,
    }

    lag_configs, lag_warnings = parse_lag_config_text(lag_text, list(working.columns))
    metadata["warnings"].extend(lag_warnings)
    working, lag_cols = apply_lag_features(working, lag_configs, order_col=metadata["order_col"])
    metadata["lag_cols"] = lag_cols

    feature_configs, feature_warnings = parse_feature_config_text(feature_text)
    metadata["warnings"].extend(feature_warnings)
    working, feature_cols, feature_apply_warnings = apply_feature_expressions(working, feature_configs)
    metadata["feature_cols"] = feature_cols
    metadata["warnings"].extend(feature_apply_warnings)

    train_idx, test_idx, split_warnings = split_train_test_indices(
        working,
        method=metadata["split_method"],
        train_ratio=metadata["train_ratio"],
        random_seed=metadata["random_seed"],
        stratify_col=metadata["stratify_col"],
        order_col=metadata["order_col"],
    )
    metadata["warnings"].extend(split_warnings)
    metadata["train_count"] = int(len(train_idx))
    metadata["test_count"] = int(len(test_idx))

    if standardize:
        working, standardized_cols = apply_standardization(working, train_idx=train_idx)
        metadata["standardized_cols"] = standardized_cols

    return working, metadata

