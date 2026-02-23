"""Model training, hyperparameter suggestion, and evaluation helpers."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .modeling import (
    normalize_random_seed,
    normalize_split_method,
    normalize_train_ratio,
    split_train_test_indices,
)

MODEL_DEFINITIONS: dict[str, dict[str, str]] = {
    "unsup_pca": {"task": "unsupervised", "label": "教師なし: PCA"},
    "unsup_ica": {"task": "unsupervised", "label": "教師なし: ICA"},
    "reg_linear": {"task": "regression", "label": "回帰: 重回帰"},
    "reg_pls": {"task": "regression", "label": "回帰: PLS回帰"},
    "reg_lgbm": {"task": "regression", "label": "回帰: LightGBM"},
    "reg_rf": {"task": "regression", "label": "回帰: ランダムフォレスト"},
    "cls_logistic": {"task": "classification", "label": "分類: ロジスティクス回帰"},
    "cls_lgbm": {"task": "classification", "label": "分類: LightGBM"},
    "cls_tree": {"task": "classification", "label": "分類: 決定木"},
    "cls_rf": {"task": "classification", "label": "分類: ランダムフォレスト"},
}

DEFAULT_MODEL_KEY = "reg_linear"


def model_options() -> list[dict[str, str]]:
    """Return dropdown options for model selection."""
    return [{"label": conf["label"], "value": key} for key, conf in MODEL_DEFINITIONS.items()]


def default_model_key() -> str:
    """Return default model key."""
    return DEFAULT_MODEL_KEY


def model_task(model_key: str) -> str:
    """Return model task type."""
    return MODEL_DEFINITIONS.get(model_key, MODEL_DEFINITIONS[DEFAULT_MODEL_KEY])["task"]


def model_label(model_key: str) -> str:
    """Return display label for model key."""
    return MODEL_DEFINITIONS.get(model_key, MODEL_DEFINITIONS[DEFAULT_MODEL_KEY])["label"]


def model_requires_target(model_key: str) -> bool:
    """Return True if selected model requires target variable."""
    return model_task(model_key) in {"regression", "classification"}


def parse_param_text(text: str | None) -> tuple[dict[str, Any], str | None]:
    """Parse user input JSON text for hyperparameters."""
    raw = (text or "").strip()
    if not raw:
        return {}, None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {}, f"ハイパーパラメータJSONの解析に失敗しました: {exc}"
    if not isinstance(parsed, dict):
        return {}, "ハイパーパラメータは JSON オブジェクトで指定してください。"
    return parsed, None


def format_param_text(params: dict[str, Any]) -> str:
    """Format hyperparameter dictionary for textarea display."""
    normalized = {key: _to_python_scalar(value) for key, value in params.items()}
    return json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True)


def default_hyperparams(model_key: str) -> dict[str, Any]:
    """Return default hyperparameters for each model."""
    defaults: dict[str, dict[str, Any]] = {
        "unsup_pca": {"n_components": 2},
        "unsup_ica": {"n_components": 2, "max_iter": 500, "tol": 1e-4},
        "reg_linear": {"fit_intercept": True},
        "reg_pls": {"n_components": 2, "max_iter": 500},
        "reg_lgbm": {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31},
        "reg_rf": {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 1},
        "cls_logistic": {"C": 1.0, "max_iter": 1000},
        "cls_lgbm": {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31},
        "cls_tree": {"criterion": "entropy", "max_depth": None, "min_samples_leaf": 1},
        "cls_rf": {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 1},
    }
    return dict(defaults.get(model_key, {}))


def hyperparam_grid(model_key: str) -> dict[str, list[Any]]:
    """Return lightweight CV search grids by model."""
    grids: dict[str, dict[str, list[Any]]] = {
        "reg_linear": {"fit_intercept": [True, False]},
        "reg_pls": {"n_components": [1, 2, 3, 4]},
        "reg_lgbm": {
            "n_estimators": [100, 300, 500],
            "learning_rate": [0.03, 0.05, 0.1],
            "num_leaves": [15, 31, 63],
        },
        "reg_rf": {
            "n_estimators": [200, 400],
            "max_depth": [None, 6, 12],
            "min_samples_leaf": [1, 3, 5],
        },
        "cls_logistic": {
            "C": [0.1, 1.0, 10.0],
            "solver": ["lbfgs"],
        },
        "cls_lgbm": {
            "n_estimators": [100, 300, 500],
            "learning_rate": [0.03, 0.05, 0.1],
            "num_leaves": [15, 31, 63],
        },
        "cls_tree": {
            "criterion": ["entropy"],
            "max_depth": [None, 4, 8, 12],
            "min_samples_leaf": [1, 3, 5],
        },
        "cls_rf": {
            "n_estimators": [200, 400],
            "max_depth": [None, 6, 12],
            "min_samples_leaf": [1, 3, 5],
        },
    }
    return dict(grids.get(model_key, {}))


def suggest_hyperparameters(
    df: pd.DataFrame,
    *,
    model_key: str,
    feature_cols: list[str] | None,
    target_col: str | None,
    split_method: str,
    train_ratio: float,
    random_seed: int,
    split_stratify_col: str | None,
    split_order_col: str | None,
    cv_folds: int,
) -> tuple[dict[str, Any], str]:
    """Suggest hyperparameters from training data.

    Supervised models use cross-validation on training split.
    Unsupervised models use train split heuristics.
    """
    task = model_task(model_key)
    safe_cv = int(max(2, min(int(cv_folds), 10)))

    if task == "unsupervised":
        prepared = _prepare_unsupervised_dataset(
            df,
            feature_cols=feature_cols,
            split_method=split_method,
            train_ratio=train_ratio,
            random_seed=random_seed,
            split_stratify_col=split_stratify_col,
            split_order_col=split_order_col,
        )
        params, summary = _suggest_unsupervised_params(
            model_key,
            prepared["x_train"],
            random_seed=prepared["random_seed"],
        )
        return params, summary

    prepared = _prepare_supervised_dataset(
        df,
        model_key=model_key,
        feature_cols=feature_cols,
        target_col=target_col,
        split_method=split_method,
        train_ratio=train_ratio,
        random_seed=random_seed,
        split_stratify_col=split_stratify_col,
        split_order_col=split_order_col,
    )
    x_train: pd.DataFrame = prepared["x_train"]
    y_train = prepared["y_train"]

    from sklearn.compose import TransformedTargetRegressor
    from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    preprocessor = _build_feature_preprocessor(x_train)
    base_estimator = _build_estimator(model_key, {}, random_seed=prepared["random_seed"])
    grid = hyperparam_grid(model_key)
    if not grid:
        params = default_hyperparams(model_key)
        return params, "このモデルには探索グリッドが未定義のため、既定値を提案します。"

    if task == "regression":
        y_train_arr = pd.to_numeric(y_train, errors="coerce").to_numpy()
        wrapped = TransformedTargetRegressor(
            regressor=Pipeline([("prep", preprocessor), ("model", base_estimator)]),
            transformer=StandardScaler(),
        )
        grid_with_prefix = {f"regressor__model__{key}": values for key, values in grid.items()}
        cv = KFold(n_splits=min(safe_cv, len(x_train)), shuffle=True, random_state=prepared["random_seed"])
        search = GridSearchCV(
            estimator=wrapped,
            param_grid=grid_with_prefix,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1,
            error_score="raise",
        )
        search.fit(x_train, y_train_arr)
        best = {
            key.replace("regressor__model__", ""): _to_python_scalar(val)
            for key, val in search.best_params_.items()
        }
        best_rmse = -float(search.best_score_)
        summary = f"CV完了: best RMSE={best_rmse:.6g} (fold={cv.get_n_splits()})"
        return best, summary

    y_train_series = y_train.astype(str)
    class_count = y_train_series.nunique(dropna=True)
    if class_count < 2:
        return default_hyperparams(model_key), "学習データのクラスが1種類のみのためCVを実行できません。"

    cv_splits = min(safe_cv, int(y_train_series.value_counts().min()))
    cv_splits = max(2, cv_splits)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=prepared["random_seed"])
    pipe = Pipeline([("prep", preprocessor), ("model", base_estimator)])
    grid_with_prefix = {f"model__{key}": values for key, values in grid.items()}
    scoring = "roc_auc_ovr" if class_count > 2 else "roc_auc"
    search = GridSearchCV(
        estimator=pipe,
        param_grid=grid_with_prefix,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        error_score="raise",
    )
    search.fit(x_train, y_train_series)
    best = {key.replace("model__", ""): _to_python_scalar(val) for key, val in search.best_params_.items()}
    best_score = float(search.best_score_)
    summary = f"CV完了: best {scoring}={best_score:.6g} (fold={cv.get_n_splits()})"
    return best, summary


def run_model(
    df: pd.DataFrame,
    *,
    model_key: str,
    feature_cols: list[str] | None,
    target_col: str | None,
    split_method: str,
    train_ratio: float,
    random_seed: int,
    split_stratify_col: str | None,
    split_order_col: str | None,
    hyperparams: dict[str, Any] | None,
) -> dict[str, Any]:
    """Train selected model and build evaluation artifacts."""
    task = model_task(model_key)
    merged_params = default_hyperparams(model_key)
    merged_params.update(hyperparams or {})
    merged_params = {key: _to_python_scalar(val) for key, val in merged_params.items()}

    if task == "unsupervised":
        prepared = _prepare_unsupervised_dataset(
            df,
            feature_cols=feature_cols,
            split_method=split_method,
            train_ratio=train_ratio,
            random_seed=random_seed,
            split_stratify_col=split_stratify_col,
            split_order_col=split_order_col,
        )
        return _run_unsupervised_model(model_key, prepared, merged_params)

    prepared = _prepare_supervised_dataset(
        df,
        model_key=model_key,
        feature_cols=feature_cols,
        target_col=target_col,
        split_method=split_method,
        train_ratio=train_ratio,
        random_seed=random_seed,
        split_stratify_col=split_stratify_col,
        split_order_col=split_order_col,
    )
    if task == "regression":
        return _run_regression_model(model_key, prepared, merged_params)
    return _run_classification_model(model_key, prepared, merged_params)


def _prepare_unsupervised_dataset(
    df: pd.DataFrame,
    *,
    feature_cols: list[str] | None,
    split_method: str,
    train_ratio: float,
    random_seed: int,
    split_stratify_col: str | None,
    split_order_col: str | None,
) -> dict[str, Any]:
    """Prepare unsupervised model inputs and train/test split."""
    available = [col for col in df.columns if col != "id"]
    chosen = [col for col in (feature_cols or []) if col in available]
    if not chosen:
        chosen = available[: min(len(available), 8)]
    if not chosen:
        raise ValueError("特徴量列を1つ以上選択してください。")

    working = df[chosen].copy()
    # 全特徴が欠損の行は学習に使えないため除外する。
    working = working.loc[~working[chosen].isna().all(axis=1)]
    if len(working) < 3:
        raise ValueError("モデリングに必要な行数が不足しています。")

    split_frame = working.copy()
    if split_stratify_col and split_stratify_col in df.columns and split_stratify_col not in split_frame.columns:
        split_frame[split_stratify_col] = df.loc[split_frame.index, split_stratify_col]
    if split_order_col and split_order_col in df.columns and split_order_col not in split_frame.columns:
        split_frame[split_order_col] = df.loc[split_frame.index, split_order_col]

    safe_method = normalize_split_method(split_method)
    safe_ratio = normalize_train_ratio(train_ratio)
    safe_seed = normalize_random_seed(random_seed)
    stratify_col = split_stratify_col if split_stratify_col in split_frame.columns else None
    order_col = split_order_col if split_order_col in split_frame.columns else None

    train_idx, test_idx, split_warnings = split_train_test_indices(
        split_frame,
        method=safe_method,
        train_ratio=safe_ratio,
        random_seed=safe_seed,
        stratify_col=stratify_col,
        order_col=order_col,
    )
    if len(train_idx) < 2 or len(test_idx) < 1:
        raise ValueError("学習/テスト分割に失敗しました。分割比率やデータ行数を確認してください。")

    return {
        "x_train": working.loc[train_idx].copy(),
        "x_test": working.loc[test_idx].copy(),
        "train_idx": train_idx,
        "test_idx": test_idx,
        "feature_cols": chosen,
        "split_warnings": split_warnings,
        "split_method": safe_method,
        "train_ratio": safe_ratio,
        "random_seed": safe_seed,
    }


def _prepare_supervised_dataset(
    df: pd.DataFrame,
    *,
    model_key: str,
    feature_cols: list[str] | None,
    target_col: str | None,
    split_method: str,
    train_ratio: float,
    random_seed: int,
    split_stratify_col: str | None,
    split_order_col: str | None,
) -> dict[str, Any]:
    """Prepare supervised model inputs and train/test split."""
    if not target_col or target_col not in df.columns:
        raise ValueError("目的変数を選択してください。")

    available = [col for col in df.columns if col not in {"id", target_col}]
    chosen = [col for col in (feature_cols or []) if col in available]
    if not chosen:
        chosen = available
    if not chosen:
        raise ValueError("説明変数を1つ以上選択してください。")

    task = model_task(model_key)
    working = df[[*chosen, target_col]].copy()
    if task == "regression":
        working[target_col] = pd.to_numeric(working[target_col], errors="coerce")
        working = working.dropna(subset=[target_col])
    else:
        target = working[target_col].astype("string")
        working = working.loc[target.notna()].copy()
        working[target_col] = target.loc[target.notna()].astype(str)

    if len(working) < 5:
        raise ValueError("モデリングに必要な行数が不足しています。")

    split_frame = working.copy()
    if split_stratify_col and split_stratify_col in df.columns and split_stratify_col not in split_frame.columns:
        split_frame[split_stratify_col] = df.loc[split_frame.index, split_stratify_col]
    if split_order_col and split_order_col in df.columns and split_order_col not in split_frame.columns:
        split_frame[split_order_col] = df.loc[split_frame.index, split_order_col]

    safe_method = normalize_split_method(split_method)
    safe_ratio = normalize_train_ratio(train_ratio)
    safe_seed = normalize_random_seed(random_seed)

    stratify_col = split_stratify_col if split_stratify_col in split_frame.columns else None
    if task == "classification" and safe_method == "stratified_random" and not stratify_col:
        stratify_col = target_col
    order_col = split_order_col if split_order_col in split_frame.columns else None

    train_idx, test_idx, split_warnings = split_train_test_indices(
        split_frame,
        method=safe_method,
        train_ratio=safe_ratio,
        random_seed=safe_seed,
        stratify_col=stratify_col,
        order_col=order_col,
    )
    if len(train_idx) < 2 or len(test_idx) < 1:
        raise ValueError("学習/テスト分割に失敗しました。分割比率やデータ行数を確認してください。")

    x_train = working.loc[train_idx, chosen].copy()
    x_test = working.loc[test_idx, chosen].copy()
    y_train = working.loc[train_idx, target_col].copy()
    y_test = working.loc[test_idx, target_col].copy()

    return {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "target_col": target_col,
        "feature_cols": chosen,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "split_warnings": split_warnings,
        "split_method": safe_method,
        "train_ratio": safe_ratio,
        "random_seed": safe_seed,
    }


def _build_feature_preprocessor(x_train: pd.DataFrame) -> Any:
    """Build column transformer with standardization on numeric features."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    numeric_cols = [
        col
        for col in x_train.columns
        if pd.api.types.is_numeric_dtype(x_train[col]) or pd.api.types.is_bool_dtype(x_train[col])
    ]
    categorical_cols = [col for col in x_train.columns if col not in numeric_cols]

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_cols:
        num_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", num_pipe, numeric_cols))

    if categorical_cols:
        cat_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", _build_onehot_encoder()),
            ]
        )
        transformers.append(("cat", cat_pipe, categorical_cols))

    if not transformers:
        raise ValueError("有効な説明変数がありません。")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _build_onehot_encoder() -> Any:
    """Build OneHotEncoder with compatibility for sklearn versions."""
    from sklearn.preprocessing import OneHotEncoder

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _build_estimator(model_key: str, params: dict[str, Any], *, random_seed: int) -> Any:
    """Instantiate estimator from model key."""
    merged = default_hyperparams(model_key)
    merged.update(params)
    if model_key in {"reg_rf", "cls_rf", "cls_tree", "reg_lgbm", "cls_lgbm", "unsup_pca", "unsup_ica"}:
        merged.setdefault("random_state", random_seed)

    try:
        if model_key == "reg_linear":
            from sklearn.linear_model import LinearRegression

            return LinearRegression(**merged)
        if model_key == "reg_pls":
            from sklearn.cross_decomposition import PLSRegression

            return PLSRegression(**merged)
        if model_key == "reg_rf":
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(**merged)
        if model_key == "cls_logistic":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(**merged)
        if model_key == "cls_tree":
            from sklearn.tree import DecisionTreeClassifier

            return DecisionTreeClassifier(**merged)
        if model_key == "cls_rf":
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(**merged)
        if model_key == "unsup_pca":
            from sklearn.decomposition import PCA

            return PCA(**merged)
        if model_key == "unsup_ica":
            from sklearn.decomposition import FastICA

            return FastICA(**merged)
        if model_key == "reg_lgbm":
            from lightgbm import LGBMRegressor

            return LGBMRegressor(**merged)
        if model_key == "cls_lgbm":
            from lightgbm import LGBMClassifier

            return LGBMClassifier(**merged)
    except ImportError as exc:
        if model_key in {"reg_lgbm", "cls_lgbm"}:
            raise ValueError(
                "LightGBM が見つかりません。`pip install lightgbm` を実行してください。"
            ) from exc
        raise ValueError(
            "scikit-learn が見つかりません。`pip install scikit-learn` を実行してください。"
        ) from exc
    except TypeError as exc:
        raise ValueError(f"ハイパーパラメータが不正です: {exc}") from exc
    raise ValueError(f"未対応のモデルキーです: {model_key}")


def _transformed_feature_names(preprocessor: Any) -> list[str]:
    """Return transformed feature names from a fitted preprocessor."""
    try:
        names = list(preprocessor.get_feature_names_out())
        return [str(name) for name in names]
    except Exception:
        return [f"feature_{i}" for i in range(int(getattr(preprocessor, "n_features_in_", 0)))]


def _importance_bar_figure(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str = "feature",
    title: str,
    top_n: int = 20,
) -> go.Figure:
    """Build horizontal bar figure for importance table."""
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    working = df.copy()
    if pd.api.types.is_numeric_dtype(working[x_col]):
        working = working.sort_values(x_col, ascending=False)
    working = working.head(top_n)
    fig = go.Figure(
        go.Bar(
            x=working[x_col],
            y=working[y_col],
            orientation="h",
            marker_color="#4C78A8",
        )
    )
    fig.update_layout(
        title=title,
        margin={"l": 120, "r": 24, "t": 52, "b": 44},
        yaxis={"autorange": "reversed"},
        xaxis_title=x_col,
        yaxis_title="feature",
    )
    return fig


def _coef_importance_table(feature_names: list[str], coef: np.ndarray, label: str) -> tuple[pd.DataFrame, go.Figure]:
    """Create standardized coefficient importance table/figure."""
    coef_1d = np.asarray(coef, dtype=float).reshape(-1)
    df = pd.DataFrame({"feature": feature_names, "beta": coef_1d})
    df["abs_beta"] = df["beta"].abs()
    df = df.sort_values("abs_beta", ascending=False)
    fig = _importance_bar_figure(df, x_col="beta", title=label, top_n=20)
    return df, fig


def _pls_vip_scores(pls_model: Any, feature_names: list[str]) -> pd.DataFrame:
    """Calculate VIP scores for fitted PLSRegression."""
    t = np.asarray(pls_model.x_scores_, dtype=float)
    w = np.asarray(pls_model.x_weights_, dtype=float)
    q = np.asarray(pls_model.y_loadings_, dtype=float)
    if q.ndim == 1:
        q = q.reshape(1, -1)

    p, h = w.shape
    if p == 0 or h == 0:
        return pd.DataFrame(columns=["feature", "vip"])
    s = np.diag(t.T @ t @ q.T @ q).reshape(-1)
    total_s = float(np.sum(s))
    if total_s <= 0:
        total_s = 1.0
    vip = np.zeros(p, dtype=float)
    for i in range(p):
        weight = np.array([(w[i, j] ** 2) * s[j] / np.sum(w[:, j] ** 2) for j in range(h)], dtype=float)
        vip[i] = np.sqrt(p * np.sum(weight) / total_s)
    df = pd.DataFrame({"feature": feature_names, "vip": vip})
    df = df.sort_values("vip", ascending=False)
    return df


def _permutation_importance_table(
    fitted_model: Any,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    *,
    scoring: str,
    random_seed: int,
) -> pd.DataFrame:
    """Compute permutation importance on original feature columns."""
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        fitted_model,
        x_test,
        y_test,
        n_repeats=10,
        random_state=random_seed,
        scoring=scoring,
    )
    df = pd.DataFrame(
        {
            "feature": list(x_test.columns),
            "permutation_importance_mean": np.asarray(result.importances_mean, dtype=float),
            "permutation_importance_std": np.asarray(result.importances_std, dtype=float),
        }
    )
    return df.sort_values("permutation_importance_mean", ascending=False)


def _tree_shap_importance_table(
    estimator: Any,
    x_transformed: np.ndarray,
    feature_names: list[str],
) -> tuple[pd.DataFrame | None, str | None]:
    """Compute SHAP mean absolute importance for tree models (optional)."""
    try:
        import shap  # type: ignore
    except ImportError:
        return None, "SHAP値の計算には `shap` ライブラリが必要です。"

    try:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(x_transformed)
    except Exception as exc:
        return None, f"SHAP値の計算に失敗しました: {exc}"

    values = np.asarray(shap_values, dtype=float)
    if isinstance(shap_values, list):
        # multiclass: list[n_classes] of (n_samples, n_features)
        stacked = np.stack([np.asarray(v, dtype=float) for v in shap_values], axis=0)
        mean_abs = np.mean(np.abs(stacked), axis=(0, 1))
    else:
        if values.ndim == 3:
            mean_abs = np.mean(np.abs(values), axis=(0, 1))
        else:
            mean_abs = np.mean(np.abs(values), axis=0)
    df = pd.DataFrame({"feature": feature_names, "shap_mean_abs": np.asarray(mean_abs, dtype=float)})
    return df.sort_values("shap_mean_abs", ascending=False), None


def _decision_tree_structure_figure(
    estimator: Any,
    feature_names: list[str],
    *,
    max_depth: int = 3,
) -> go.Figure:
    """Build a simple node-link diagram for decision tree."""
    tree = estimator.tree_
    left = tree.children_left
    right = tree.children_right
    feat_idx = tree.feature
    thr = tree.threshold
    values = tree.value

    nodes: list[dict[str, Any]] = []
    edges_x: list[float] = []
    edges_y: list[float] = []
    leaf_counter = 0

    def walk(node_id: int, depth: int) -> tuple[float, float]:
        nonlocal leaf_counter
        if depth > max_depth:
            x = float(leaf_counter)
            y = -float(depth)
            leaf_counter += 1
            nodes.append({"x": x, "y": y, "text": f"… (node={node_id})", "leaf": True})
            return x, y

        is_leaf = left[node_id] == right[node_id]
        if is_leaf:
            x = float(leaf_counter)
            y = -float(depth)
            leaf_counter += 1
            class_counts = np.asarray(values[node_id]).reshape(-1)
            pred = int(np.argmax(class_counts)) if class_counts.size else -1
            txt = f"leaf {node_id}<br>pred={pred}<br>n={int(np.sum(class_counts))}"
            nodes.append({"x": x, "y": y, "text": txt, "leaf": True})
            return x, y

        lx, ly = walk(int(left[node_id]), depth + 1)
        rx, ry = walk(int(right[node_id]), depth + 1)
        x = (lx + rx) / 2.0
        y = -float(depth)
        f_idx = int(feat_idx[node_id])
        f_name = feature_names[f_idx] if 0 <= f_idx < len(feature_names) else f"f{f_idx}"
        txt = f"node {node_id}<br>{f_name} <= {thr[node_id]:.3g}<br>n={int(np.sum(values[node_id]))}"
        nodes.append({"x": x, "y": y, "text": txt, "leaf": False})

        edges_x.extend([x, lx, None, x, rx, None])
        edges_y.extend([y, ly, None, y, ry, None])
        return x, y

    walk(0, 0)
    node_df = pd.DataFrame(nodes)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edges_x,
            y=edges_y,
            mode="lines",
            line={"color": "#9e9e9e"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=node_df["x"],
            y=node_df["y"],
            mode="markers+text",
            text=node_df["text"],
            textposition="top center",
            marker={
                "size": 14,
                "color": ["#F58518" if bool(v) else "#4C78A8" for v in node_df["leaf"]],
                "line": {"color": "#333", "width": 1},
            },
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        margin={"l": 24, "r": 24, "t": 48, "b": 24},
        xaxis={"visible": False},
        yaxis={"visible": False},
        title=f"決定木の図式化 (depth<= {max_depth})",
    )
    return fig


def _run_regression_model(
    model_key: str,
    prepared: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Train/evaluate regression model and create figures."""
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    x_train: pd.DataFrame = prepared["x_train"]
    x_test: pd.DataFrame = prepared["x_test"]
    y_train = pd.to_numeric(prepared["y_train"], errors="coerce").to_numpy()
    y_test = pd.to_numeric(prepared["y_test"], errors="coerce").to_numpy()

    preprocessor = _build_feature_preprocessor(x_train)
    estimator = _build_estimator(model_key, params, random_seed=prepared["random_seed"])

    model = TransformedTargetRegressor(
        regressor=Pipeline([("prep", preprocessor), ("model", estimator)]),
        transformer=StandardScaler(),
    )
    model.fit(x_train, y_train)

    fitted_pipe = model.regressor_
    fitted_prep = fitted_pipe.named_steps["prep"]
    fitted_estimator = fitted_pipe.named_steps["model"]
    transformed_names = _transformed_feature_names(fitted_prep)
    x_test_transformed = np.asarray(fitted_prep.transform(x_test))

    pred_train = np.asarray(model.predict(x_train), dtype=float)
    pred_test = np.asarray(model.predict(x_test), dtype=float)

    r2_train = float(r2_score(y_train, pred_train))
    r2_test = float(r2_score(y_test, pred_test))
    rmse_train = float(np.sqrt(mean_squared_error(y_train, pred_train)))
    rmse_test = float(np.sqrt(mean_squared_error(y_test, pred_test)))
    mae_train = float(mean_absolute_error(y_train, pred_train))
    mae_test = float(mean_absolute_error(y_test, pred_test))

    metrics = pd.DataFrame(
        [
            {"dataset": "train", "R2": r2_train, "RMSE": rmse_train, "MAE": mae_train},
            {"dataset": "test", "R2": r2_test, "RMSE": rmse_test, "MAE": mae_test},
        ]
    )

    overlay_fig = _build_regression_overlay_figure(
        y_train=y_train,
        y_test=y_test,
        pred_train=pred_train,
        pred_test=pred_test,
        train_idx=prepared["train_idx"],
        test_idx=prepared["test_idx"],
        target_col=str(prepared["target_col"]),
    )
    yy_fig = _build_regression_yy_figure(
        y_train=y_train,
        y_test=y_test,
        pred_train=pred_train,
        pred_test=pred_test,
        target_col=str(prepared["target_col"]),
    )

    notes: list[str] = list(prepared["split_warnings"])
    notes.append("回帰モデルは標準化済み学習データで学習し、予測値は元スケールに戻して評価しています。")

    importance_tables: list[dict[str, Any]] = []
    importance_figures: list[dict[str, Any]] = []
    extra_text_blocks: list[dict[str, str]] = []

    if model_key == "reg_linear":
        coef = np.asarray(getattr(fitted_estimator, "coef_", []), dtype=float).reshape(-1)
        coef_df, coef_fig = _coef_importance_table(transformed_names[: len(coef)], coef, "標準化回帰係数 (β)")
        importance_tables.append({"title": "標準化回帰係数 (β)", "data": coef_df})
        importance_figures.append({"title": "標準化回帰係数 (β)", "figure": coef_fig})
    elif model_key == "reg_pls":
        vip_df = _pls_vip_scores(fitted_estimator, transformed_names)
        importance_tables.append({"title": "VIP", "data": vip_df})
        importance_figures.append({"title": "VIP", "figure": _importance_bar_figure(vip_df, x_col="vip", title="PLS VIP")})
    elif model_key == "reg_lgbm":
        booster = getattr(fitted_estimator, "booster_", None)
        if booster is not None:
            gain = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
            gain_df = pd.DataFrame({"feature": transformed_names[: len(gain)], "gain_importance": gain}).sort_values(
                "gain_importance", ascending=False
            )
            importance_tables.append({"title": "Gain importance", "data": gain_df})
            importance_figures.append(
                {"title": "Gain importance", "figure": _importance_bar_figure(gain_df, x_col="gain_importance", title="LightGBM Gain importance")}
            )
        shap_df, shap_msg = _tree_shap_importance_table(fitted_estimator, x_test_transformed, transformed_names)
        if shap_df is not None:
            importance_tables.append({"title": "SHAP値 (mean |SHAP|)", "data": shap_df})
            importance_figures.append(
                {"title": "SHAP値 (mean |SHAP|)", "figure": _importance_bar_figure(shap_df, x_col="shap_mean_abs", title="LightGBM SHAP")}
            )
        elif shap_msg:
            extra_text_blocks.append({"title": "SHAP値", "text": shap_msg})
    elif model_key == "reg_rf":
        perm_df = _permutation_importance_table(
            model,
            x_test,
            y_test,
            scoring="neg_root_mean_squared_error",
            random_seed=int(prepared["random_seed"]),
        )
        importance_tables.append({"title": "Permutation importance", "data": perm_df})
        importance_figures.append(
            {
                "title": "Permutation importance",
                "figure": _importance_bar_figure(perm_df, x_col="permutation_importance_mean", title="RandomForest Permutation importance"),
            }
        )
        shap_df, shap_msg = _tree_shap_importance_table(fitted_estimator, x_test_transformed, transformed_names)
        if shap_df is not None:
            importance_tables.append({"title": "SHAP値 (mean |SHAP|)", "data": shap_df})
            importance_figures.append(
                {"title": "SHAP値 (mean |SHAP|)", "figure": _importance_bar_figure(shap_df, x_col="shap_mean_abs", title="RandomForest SHAP")}
            )
        elif shap_msg:
            extra_text_blocks.append({"title": "SHAP値", "text": shap_msg})

    return {
        "task": "regression",
        "model_key": model_key,
        "model_label": model_label(model_key),
        "used_params": params,
        "metrics": metrics,
        "figures": [overlay_fig, yy_fig],
        "notes": notes,
        "importance_tables": importance_tables,
        "importance_figures": importance_figures,
        "extra_text_blocks": extra_text_blocks,
        "artifact_bundle": {
            "meta": {
                "model_key": model_key,
                "model_label": model_label(model_key),
                "task": "regression",
                "target_col": prepared.get("target_col"),
                "feature_count": len(prepared.get("feature_cols", [])),
            },
            "trained_model": model,
            "prepared_info": {
                "feature_cols": list(prepared.get("feature_cols", [])),
                "target_col": prepared.get("target_col"),
                "params": params,
            },
        },
    }


def _run_classification_model(
    model_key: str,
    prepared: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Train/evaluate classification model and create figures."""
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, label_binarize

    x_train: pd.DataFrame = prepared["x_train"]
    x_test: pd.DataFrame = prepared["x_test"]
    y_train_raw = prepared["y_train"].astype(str).to_numpy()
    y_test_raw = prepared["y_test"].astype(str).to_numpy()

    label_encoder = LabelEncoder()
    label_encoder.fit(np.concatenate([y_train_raw, y_test_raw]))
    y_train = label_encoder.transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)
    class_names = list(label_encoder.classes_)

    if len(class_names) < 2:
        raise ValueError("分類モデルでは2クラス以上が必要です。")

    preprocessor = _build_feature_preprocessor(x_train)
    estimator = _build_estimator(model_key, params, random_seed=prepared["random_seed"])
    model = Pipeline([("prep", preprocessor), ("model", estimator)])
    model.fit(x_train, y_train)

    fitted_prep = model.named_steps["prep"]
    fitted_estimator = model.named_steps["model"]
    transformed_names = _transformed_feature_names(fitted_prep)
    x_test_transformed = np.asarray(fitted_prep.transform(x_test))

    pred_train = np.asarray(model.predict(x_train), dtype=int)
    pred_test = np.asarray(model.predict(x_test), dtype=int)
    acc_train = float(accuracy_score(y_train, pred_train))
    acc_test = float(accuracy_score(y_test, pred_test))

    proba_train = model.predict_proba(x_train) if hasattr(model, "predict_proba") else None
    proba_test = model.predict_proba(x_test) if hasattr(model, "predict_proba") else None

    auc_train = np.nan
    auc_test = np.nan
    roc_fig = go.Figure()
    roc_fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"dash": "dash", "color": "#9e9e9e"},
            name="ランダム",
        )
    )

    if proba_train is not None and proba_test is not None:
        if len(class_names) == 2:
            train_scores = proba_train[:, 1]
            test_scores = proba_test[:, 1]
            fpr_tr, tpr_tr, _ = roc_curve(y_train, train_scores)
            fpr_te, tpr_te, _ = roc_curve(y_test, test_scores)
            auc_train = float(roc_auc_score(y_train, train_scores))
            auc_test = float(roc_auc_score(y_test, test_scores))
            roc_fig.add_trace(
                go.Scatter(
                    x=fpr_tr,
                    y=tpr_tr,
                    mode="lines",
                    name=f"学習 ROC (AUC={auc_train:.3f})",
                    line={"color": "#4C78A8"},
                )
            )
            roc_fig.add_trace(
                go.Scatter(
                    x=fpr_te,
                    y=tpr_te,
                    mode="lines",
                    name=f"テスト ROC (AUC={auc_test:.3f})",
                    line={"color": "#F58518"},
                )
            )
        else:
            y_train_bin = label_binarize(y_train, classes=np.arange(len(class_names)))
            y_test_bin = label_binarize(y_test, classes=np.arange(len(class_names)))
            auc_train = float(roc_auc_score(y_train_bin, proba_train, average="macro", multi_class="ovr"))
            auc_test = float(roc_auc_score(y_test_bin, proba_test, average="macro", multi_class="ovr"))

            for class_idx, name in enumerate(class_names):
                fpr_tr, tpr_tr, _ = roc_curve(y_train_bin[:, class_idx], proba_train[:, class_idx])
                roc_fig.add_trace(
                    go.Scatter(
                        x=fpr_tr,
                        y=tpr_tr,
                        mode="lines",
                        name=f"学習 {name}",
                        line={"width": 2},
                    )
                )
            for class_idx, name in enumerate(class_names):
                fpr_te, tpr_te, _ = roc_curve(y_test_bin[:, class_idx], proba_test[:, class_idx])
                roc_fig.add_trace(
                    go.Scatter(
                        x=fpr_te,
                        y=tpr_te,
                        mode="lines",
                        name=f"テスト {name}",
                        line={"dash": "dot", "width": 2},
                    )
                )
            roc_fig.update_layout(title=f"ROC曲線 (macro AUC: train={auc_train:.3f}, test={auc_test:.3f})")

    roc_fig.update_layout(
        margin={"l": 44, "r": 24, "t": 52, "b": 52},
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )

    labels = list(range(len(class_names)))
    cm_train = confusion_matrix(y_train, pred_train, labels=labels)
    cm_test = confusion_matrix(y_test, pred_test, labels=labels)
    cm_fig = make_subplots(rows=1, cols=2, subplot_titles=["学習データ", "テストデータ"])
    cm_fig.add_trace(
        go.Heatmap(
            z=cm_train,
            x=class_names,
            y=class_names,
            colorscale="Blues",
            showscale=False,
            text=cm_train,
            texttemplate="%{text}",
        ),
        row=1,
        col=1,
    )
    cm_fig.add_trace(
        go.Heatmap(
            z=cm_test,
            x=class_names,
            y=class_names,
            colorscale="Oranges",
            showscale=False,
            text=cm_test,
            texttemplate="%{text}",
        ),
        row=1,
        col=2,
    )
    cm_fig.update_xaxes(title_text="予測クラス", row=1, col=1)
    cm_fig.update_xaxes(title_text="予測クラス", row=1, col=2)
    cm_fig.update_yaxes(title_text="実クラス", autorange="reversed", row=1, col=1)
    cm_fig.update_yaxes(title_text="実クラス", autorange="reversed", row=1, col=2)
    cm_fig.update_layout(margin={"l": 44, "r": 24, "t": 52, "b": 52})

    metrics = pd.DataFrame(
        [
            {"dataset": "train", "accuracy": acc_train, "auc": _to_python_scalar(auc_train)},
            {"dataset": "test", "accuracy": acc_test, "auc": _to_python_scalar(auc_test)},
        ]
    )
    notes: list[str] = list(prepared["split_warnings"])
    notes.append("分類モデルは標準化済み学習データで学習し、標準化済みテストデータで評価しています。")

    importance_tables: list[dict[str, Any]] = []
    importance_figures: list[dict[str, Any]] = []
    extra_text_blocks: list[dict[str, str]] = []

    if model_key == "cls_logistic":
        coef_arr = np.asarray(getattr(fitted_estimator, "coef_", []), dtype=float)
        if coef_arr.ndim == 1:
            coef_df, coef_fig = _coef_importance_table(transformed_names[: len(coef_arr)], coef_arr, "標準化回帰係数 (β)")
            importance_tables.append({"title": "標準化回帰係数 (β)", "data": coef_df})
            importance_figures.append({"title": "標準化回帰係数 (β)", "figure": coef_fig})
        elif coef_arr.ndim == 2 and coef_arr.size:
            mean_abs = np.mean(np.abs(coef_arr), axis=0)
            beta_df = pd.DataFrame(
                {
                    "feature": transformed_names[: coef_arr.shape[1]],
                    "beta_abs_mean": mean_abs,
                }
            ).sort_values("beta_abs_mean", ascending=False)
            importance_tables.append({"title": "標準化回帰係数 (β, クラス平均絶対値)", "data": beta_df})
            importance_figures.append(
                {
                    "title": "標準化回帰係数 (β, クラス平均絶対値)",
                    "figure": _importance_bar_figure(beta_df, x_col="beta_abs_mean", title="Logistic β (mean abs)"),
                }
            )
    elif model_key == "cls_lgbm":
        booster = getattr(fitted_estimator, "booster_", None)
        if booster is not None:
            gain = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
            gain_df = pd.DataFrame({"feature": transformed_names[: len(gain)], "gain_importance": gain}).sort_values(
                "gain_importance", ascending=False
            )
            importance_tables.append({"title": "Gain importance", "data": gain_df})
            importance_figures.append(
                {"title": "Gain importance", "figure": _importance_bar_figure(gain_df, x_col="gain_importance", title="LightGBM Gain importance")}
            )
        shap_df, shap_msg = _tree_shap_importance_table(fitted_estimator, x_test_transformed, transformed_names)
        if shap_df is not None:
            importance_tables.append({"title": "SHAP値 (mean |SHAP|)", "data": shap_df})
            importance_figures.append(
                {"title": "SHAP値 (mean |SHAP|)", "figure": _importance_bar_figure(shap_df, x_col="shap_mean_abs", title="LightGBM SHAP")}
            )
        elif shap_msg:
            extra_text_blocks.append({"title": "SHAP値", "text": shap_msg})
    elif model_key == "cls_tree":
        info_gain = np.asarray(getattr(fitted_estimator, "feature_importances_", []), dtype=float)
        info_df = pd.DataFrame({"feature": transformed_names[: len(info_gain)], "information_gain": info_gain}).sort_values(
            "information_gain", ascending=False
        )
        importance_tables.append({"title": "Information Gain", "data": info_df})
        importance_figures.append(
            {"title": "Information Gain", "figure": _importance_bar_figure(info_df, x_col="information_gain", title="DecisionTree Information Gain")}
        )
        shap_df, shap_msg = _tree_shap_importance_table(fitted_estimator, x_test_transformed, transformed_names)
        if shap_df is not None:
            importance_tables.append({"title": "SHAP値 (mean |SHAP|)", "data": shap_df})
            importance_figures.append(
                {"title": "SHAP値 (mean |SHAP|)", "figure": _importance_bar_figure(shap_df, x_col="shap_mean_abs", title="DecisionTree SHAP")}
            )
        elif shap_msg:
            extra_text_blocks.append({"title": "SHAP値", "text": shap_msg})
        try:
            importance_figures.append(
                {
                    "title": "決定木の図式化",
                    "figure": _decision_tree_structure_figure(fitted_estimator, transformed_names, max_depth=3),
                }
            )
        except Exception as exc:
            extra_text_blocks.append({"title": "決定木の図式化", "text": f"図式化に失敗しました: {exc}"})
    elif model_key == "cls_rf":
        scoring = "roc_auc_ovr" if len(class_names) > 2 else "roc_auc"
        try:
            perm_df = _permutation_importance_table(
                model,
                x_test,
                y_test,
                scoring=scoring,
                random_seed=int(prepared["random_seed"]),
            )
        except Exception:
            perm_df = _permutation_importance_table(
                model,
                x_test,
                y_test,
                scoring="accuracy",
                random_seed=int(prepared["random_seed"]),
            )
        importance_tables.append({"title": "Permutation importance", "data": perm_df})
        importance_figures.append(
            {
                "title": "Permutation importance",
                "figure": _importance_bar_figure(perm_df, x_col="permutation_importance_mean", title="RandomForest Permutation importance"),
            }
        )
        shap_df, shap_msg = _tree_shap_importance_table(fitted_estimator, x_test_transformed, transformed_names)
        if shap_df is not None:
            importance_tables.append({"title": "SHAP値 (mean |SHAP|)", "data": shap_df})
            importance_figures.append(
                {"title": "SHAP値 (mean |SHAP|)", "figure": _importance_bar_figure(shap_df, x_col="shap_mean_abs", title="RandomForest SHAP")}
            )
        elif shap_msg:
            extra_text_blocks.append({"title": "SHAP値", "text": shap_msg})

    return {
        "task": "classification",
        "model_key": model_key,
        "model_label": model_label(model_key),
        "used_params": params,
        "metrics": metrics,
        "figures": [cm_fig, roc_fig],
        "notes": notes,
        "importance_tables": importance_tables,
        "importance_figures": importance_figures,
        "extra_text_blocks": extra_text_blocks,
        "artifact_bundle": {
            "meta": {
                "model_key": model_key,
                "model_label": model_label(model_key),
                "task": "classification",
                "target_col": prepared.get("target_col"),
                "feature_count": len(prepared.get("feature_cols", [])),
            },
            "trained_model": model,
            "label_encoder": label_encoder,
            "prepared_info": {
                "feature_cols": list(prepared.get("feature_cols", [])),
                "target_col": prepared.get("target_col"),
                "params": params,
                "class_names": class_names,
            },
        },
    }


def _run_unsupervised_model(
    model_key: str,
    prepared: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Train/evaluate unsupervised model and create figures."""
    x_train: pd.DataFrame = prepared["x_train"]
    x_test: pd.DataFrame = prepared["x_test"]

    preprocessor = _build_feature_preprocessor(x_train)
    x_train_scaled = preprocessor.fit_transform(x_train)
    x_test_scaled = preprocessor.transform(x_test)
    transformed_names = _transformed_feature_names(preprocessor)

    feature_count = int(x_train_scaled.shape[1])
    if feature_count < 2:
        raise ValueError("教師なしモデルには2次元以上の特徴量が必要です。")

    user_n = int(params.get("n_components", 2))
    n_components = min(max(2, user_n), feature_count)
    tuned_params = dict(params)
    tuned_params["n_components"] = n_components
    estimator = _build_estimator(model_key, tuned_params, random_seed=prepared["random_seed"])
    estimator.fit(x_train_scaled)

    score_train = np.asarray(estimator.transform(x_train_scaled), dtype=float)
    score_test = np.asarray(estimator.transform(x_test_scaled), dtype=float)

    recon_train = np.asarray(estimator.inverse_transform(score_train), dtype=float)
    recon_test = np.asarray(estimator.inverse_transform(score_test), dtype=float)
    rmse_train = float(np.sqrt(np.mean((x_train_scaled - recon_train) ** 2)))
    rmse_test = float(np.sqrt(np.mean((x_test_scaled - recon_test) ** 2)))

    score_fig = _build_unsupervised_score_figure(
        score_train=score_train,
        score_test=score_test,
        train_idx=prepared["train_idx"],
        test_idx=prepared["test_idx"],
        model_key=model_key,
    )
    detail_fig = _build_unsupervised_detail_figure(
        model_key=model_key,
        estimator=estimator,
        score_train=score_train,
        score_test=score_test,
    )

    metrics = pd.DataFrame(
        [
            {"dataset": "train", "reconstruction_rmse": rmse_train},
            {"dataset": "test", "reconstruction_rmse": rmse_test},
        ]
    )
    notes: list[str] = list(prepared["split_warnings"])
    notes.append("教師なしモデルは標準化済み学習データで学習し、学習/テスト両方の変換結果を表示しています。")

    importance_tables: list[dict[str, Any]] = []
    importance_figures: list[dict[str, Any]] = []
    extra_text_blocks: list[dict[str, str]] = []

    if model_key == "unsup_pca":
        components = np.asarray(getattr(estimator, "components_", []), dtype=float)
        explained = np.asarray(getattr(estimator, "explained_variance_", []), dtype=float)
        if components.ndim == 2 and components.size and explained.size:
            loadings = components.T * np.sqrt(explained.reshape(1, -1))
            loading_sq = loadings**2
            comp_labels = [f"PC{i+1}" for i in range(loading_sq.shape[1])]
            load_sq_df = pd.DataFrame(loading_sq, index=transformed_names[: loading_sq.shape[0]], columns=comp_labels)
            sum_df = (
                pd.DataFrame({"feature": load_sq_df.index, "loading_sq_sum": load_sq_df.sum(axis=1).to_numpy()})
                .sort_values("loading_sq_sum", ascending=False)
            )
            importance_tables.append({"title": "Loadingの2乗（成分別）", "data": load_sq_df.reset_index().rename(columns={"index": "feature"})})
            importance_tables.append({"title": "Loadingの2乗（合計）", "data": sum_df})
            importance_figures.append(
                {
                    "title": "Loadingの2乗（合計）",
                    "figure": _importance_bar_figure(sum_df, x_col="loading_sq_sum", title="PCA Loading^2 (sum)"),
                }
            )
            heat = go.Figure(
                go.Heatmap(
                    z=load_sq_df.to_numpy(),
                    x=comp_labels,
                    y=list(load_sq_df.index),
                    colorscale="Blues",
                )
            )
            heat.update_layout(
                title="PCA Loadingの2乗 ヒートマップ",
                margin={"l": 120, "r": 24, "t": 52, "b": 44},
            )
            importance_figures.append({"title": "Loadingの2乗 ヒートマップ", "figure": heat})
    elif model_key == "unsup_ica":
        mixing = np.asarray(getattr(estimator, "mixing_", []), dtype=float)
        if mixing.ndim == 2 and mixing.size:
            comp_labels = [f"IC{i+1}" for i in range(mixing.shape[1])]
            mixing_df = pd.DataFrame(mixing, index=transformed_names[: mixing.shape[0]], columns=comp_labels)
            importance_tables.append({"title": "Mixing matrix", "data": mixing_df.reset_index().rename(columns={"index": "feature"})})
            heat = go.Figure(
                go.Heatmap(
                    z=mixing_df.to_numpy(),
                    x=comp_labels,
                    y=list(mixing_df.index),
                    colorscale="RdBu",
                    zmid=0.0,
                )
            )
            heat.update_layout(
                title="ICA Mixing matrix",
                margin={"l": 120, "r": 24, "t": 52, "b": 44},
            )
            importance_figures.append({"title": "Mixing matrix", "figure": heat})

    return {
        "task": "unsupervised",
        "model_key": model_key,
        "model_label": model_label(model_key),
        "used_params": tuned_params,
        "metrics": metrics,
        "figures": [score_fig, detail_fig],
        "notes": notes,
        "importance_tables": importance_tables,
        "importance_figures": importance_figures,
        "extra_text_blocks": extra_text_blocks,
        "artifact_bundle": {
            "meta": {
                "model_key": model_key,
                "model_label": model_label(model_key),
                "task": "unsupervised",
                "target_col": None,
                "feature_count": len(prepared.get("feature_cols", [])),
            },
            "trained_model": estimator,
            "preprocessor": preprocessor,
            "prepared_info": {
                "feature_cols": list(prepared.get("feature_cols", [])),
                "params": tuned_params,
            },
        },
    }


def _suggest_unsupervised_params(
    model_key: str,
    x_train: pd.DataFrame,
    *,
    random_seed: int,
) -> tuple[dict[str, Any], str]:
    """Suggest unsupervised hyperparameters by train split heuristics."""
    preprocessor = _build_feature_preprocessor(x_train)
    x_scaled = preprocessor.fit_transform(x_train)
    feature_count = int(x_scaled.shape[1])
    max_n = max(2, min(10, feature_count))

    if model_key == "unsup_pca":
        from sklearn.decomposition import PCA

        probe = PCA(n_components=min(feature_count, max_n), random_state=random_seed)
        probe.fit(x_scaled)
        cum = np.cumsum(np.asarray(probe.explained_variance_ratio_, dtype=float))
        target_n = int(np.searchsorted(cum, 0.90) + 1)
        target_n = min(max(2, target_n), max_n)
        params = {"n_components": target_n}
        summary = f"PCA提案: 累積寄与率90%到達で n_components={target_n}"
        return params, summary

    from sklearn.decomposition import FastICA

    best_n = 2
    best_score = -np.inf
    for n_comp in range(2, max_n + 1):
        try:
            ica = FastICA(n_components=n_comp, random_state=random_seed, max_iter=500, tol=1e-4)
            transformed = np.asarray(ica.fit_transform(x_scaled), dtype=float)
            kurt = np.abs(pd.DataFrame(transformed).kurtosis(axis=0, fisher=False)).mean()
            score = float(kurt)
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_n = n_comp
    params = {"n_components": best_n, "max_iter": 500, "tol": 1e-4}
    summary = f"ICA提案: 非ガウス性スコア最大で n_components={best_n}"
    return params, summary


def _build_regression_overlay_figure(
    *,
    y_train: np.ndarray,
    y_test: np.ndarray,
    pred_train: np.ndarray,
    pred_test: np.ndarray,
    train_idx: pd.Index,
    test_idx: pd.Index,
    target_col: str,
) -> go.Figure:
    """Build regression actual-vs-prediction overlay figure."""
    train_frame = pd.DataFrame(
        {
            "x": [str(idx) for idx in train_idx],
            "actual": y_train,
            "pred": pred_train,
            "split": "train",
        }
    )
    test_frame = pd.DataFrame(
        {
            "x": [str(idx) for idx in test_idx],
            "actual": y_test,
            "pred": pred_test,
            "split": "test",
        }
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=train_frame["x"],
            y=train_frame["actual"],
            mode="lines+markers",
            name="学習 実測",
            line={"color": "#4C78A8"},
            marker={"size": 5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=train_frame["x"],
            y=train_frame["pred"],
            mode="lines+markers",
            name="学習 予測",
            line={"color": "#4C78A8", "dash": "dot"},
            marker={"size": 5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test_frame["x"],
            y=test_frame["actual"],
            mode="lines+markers",
            name="テスト 実測",
            line={"color": "#F58518"},
            marker={"size": 5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test_frame["x"],
            y=test_frame["pred"],
            mode="lines+markers",
            name="テスト 予測",
            line={"color": "#F58518", "dash": "dot"},
            marker={"size": 5},
        )
    )
    fig.update_layout(
        title=f"実測値と予測値の重ね合わせ: {target_col}",
        xaxis_title="サンプル",
        yaxis_title=target_col,
        margin={"l": 44, "r": 24, "t": 52, "b": 72},
        legend={"orientation": "h"},
    )
    return fig


def _build_regression_yy_figure(
    *,
    y_train: np.ndarray,
    y_test: np.ndarray,
    pred_train: np.ndarray,
    pred_test: np.ndarray,
    target_col: str,
) -> go.Figure:
    """Build yy-plot figure for regression."""
    all_vals = np.concatenate([y_train, y_test, pred_train, pred_test])
    min_val = float(np.nanmin(all_vals))
    max_val = float(np.nanmax(all_vals))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_train,
            y=pred_train,
            mode="markers",
            name="学習",
            marker={"color": "#4C78A8", "size": 6, "opacity": 0.8},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=y_test,
            y=pred_test,
            mode="markers",
            name="テスト",
            marker={"color": "#F58518", "size": 6, "opacity": 0.8},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="y=x",
            line={"color": "#111111", "dash": "dash"},
        )
    )
    fig.update_layout(
        title=f"yyプロット: {target_col}",
        xaxis_title="実測値",
        yaxis_title="予測値",
        margin={"l": 44, "r": 24, "t": 52, "b": 52},
    )
    return fig


def _build_unsupervised_score_figure(
    *,
    score_train: np.ndarray,
    score_test: np.ndarray,
    train_idx: pd.Index,
    test_idx: pd.Index,
    model_key: str,
) -> go.Figure:
    """Build score scatter figure for unsupervised models."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=score_train[:, 0],
            y=score_train[:, 1],
            mode="markers",
            name="学習",
            marker={"color": "#4C78A8", "size": 6, "opacity": 0.75},
            customdata=[str(idx) for idx in train_idx],
            hovertemplate="comp1=%{x:.4g}<br>comp2=%{y:.4g}<br>index=%{customdata}<extra>学習</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=score_test[:, 0],
            y=score_test[:, 1],
            mode="markers",
            name="テスト",
            marker={"color": "#F58518", "size": 6, "opacity": 0.75},
            customdata=[str(idx) for idx in test_idx],
            hovertemplate="comp1=%{x:.4g}<br>comp2=%{y:.4g}<br>index=%{customdata}<extra>テスト</extra>",
        )
    )
    title = "PCAスコア散布図" if model_key == "unsup_pca" else "ICAスコア散布図"
    fig.update_layout(
        title=title,
        xaxis_title="第1成分",
        yaxis_title="第2成分",
        margin={"l": 44, "r": 24, "t": 52, "b": 52},
        legend={"orientation": "h"},
    )
    return fig


def _build_unsupervised_detail_figure(
    *,
    model_key: str,
    estimator: Any,
    score_train: np.ndarray,
    score_test: np.ndarray,
) -> go.Figure:
    """Build unsupervised detail figure."""
    if model_key == "unsup_pca":
        variance = np.asarray(getattr(estimator, "explained_variance_ratio_", []), dtype=float)
        cumulative = np.cumsum(variance)
        x_vals = np.arange(1, len(variance) + 1)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=x_vals, y=variance, name="寄与率", marker_color="#4C78A8"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=cumulative,
                mode="lines+markers",
                name="累積寄与率",
                line={"color": "#F58518"},
            ),
            secondary_y=True,
        )
        fig.update_yaxes(title_text="寄与率", secondary_y=False)
        fig.update_yaxes(title_text="累積寄与率", range=[0, 1], secondary_y=True)
        fig.update_layout(
            title="PCA 寄与率",
            xaxis_title="主成分番号",
            margin={"l": 44, "r": 24, "t": 52, "b": 52},
        )
        return fig

    fig = make_subplots(rows=1, cols=2, subplot_titles=["学習データ成分分布", "テストデータ成分分布"])
    fig.add_trace(
        go.Histogram(x=score_train[:, 0], nbinsx=40, marker_color="#4C78A8", name="学習 comp1"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(x=score_test[:, 0], nbinsx=40, marker_color="#F58518", name="テスト comp1"),
        row=1,
        col=2,
    )
    fig.update_layout(
        title="ICA 成分分布",
        margin={"l": 44, "r": 24, "t": 52, "b": 52},
        showlegend=False,
    )
    return fig


def _to_python_scalar(value: Any) -> Any:
    """Convert numpy scalars/arrays to JSON-safe Python scalars."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    return value
