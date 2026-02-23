"""Dash callback registration."""

from __future__ import annotations

import base64
import io
import os
import threading
import time
from typing import Any
from urllib.parse import parse_qs
from uuid import uuid4

import pandas as pd
from dash import Dash, Input, Output, State, callback_context, dash_table, dcc, html, no_update
from flask import request

from .data_io import DataLoadError, load_dataset_from_upload, prepare_dataframe
from .figures import create_distribution_figure, create_scatter_figure, create_scatter_matrix_figure
from .modeling import (
    apply_modeling_preparation,
    normalize_random_seed,
    normalize_split_method,
    normalize_train_ratio,
)
from .preprocess import (
    SUPPORTED_CAST_TYPES,
    apply_analysis_filters,
    apply_type_overrides,
    build_runtime_metadata,
    infer_series_kind_label,
    normalize_type_overrides,
)
from .model_runner import (
    default_model_key,
    format_param_text,
    model_label,
    model_requires_target,
    model_task,
    parse_param_text,
    run_model,
    suggest_hyperparameters,
)
from .db_connectors import (
    DatabaseConfig,
    DatabaseError,
    build_select_sample_query,
    dbms_label,
    execute_query,
    list_tables,
    normalize_dbms,
    normalize_port,
)
from .ranking import RANKING_COLUMNS, rank_candidate_causes
from .state import (
    build_current_data_state,
    build_default_ui_config,
    build_default_view_config,
    dataframe_from_json,
)
from .utils import empty_figure, format_dataset_meta, normalize_id_list, pick_column

VISIBLE_GRAPH_KEYS = ["scatter", "hist", "box", "matrix"]
MODEL_ARTIFACT_CACHE: dict[str, dict[str, Any]] = {}


def _schedule_server_shutdown(shutdown_callable: Any) -> None:
    """Stop the local Dash server shortly after returning callback response."""

    def _worker() -> None:
        # Let the callback response reach the browser first.
        time.sleep(0.8)
        try:
            if callable(shutdown_callable):
                shutdown_callable()
                return
        except Exception:
            pass
        # Fallback for environments where Werkzeug shutdown hook is unavailable.
        os._exit(0)

    threading.Thread(target=_worker, daemon=True).start()


def _normalize_visible_graphs(values: list[str] | None) -> list[str]:
    """Normalize visible graph key list."""
    if values is None:
        return VISIBLE_GRAPH_KEYS.copy()
    return [key for key in values if key in VISIBLE_GRAPH_KEYS]


def _model_store_payload(cache_key: str | None) -> dict[str, Any]:
    """Build dcc.Store payload for current model artifact."""
    key = str(cache_key or "")
    if not key:
        return {"cache_key": None, "meta": {}}
    artifact = MODEL_ARTIFACT_CACHE.get(key) or {}
    meta = artifact.get("meta", {})
    return {"cache_key": key, "meta": meta if isinstance(meta, dict) else {}}


def _cache_model_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    """Store model artifact in server memory and return store payload."""
    cache_key = str(uuid4())
    MODEL_ARTIFACT_CACHE[cache_key] = artifact
    return _model_store_payload(cache_key)


def _get_model_artifact_from_store(store_data: dict[str, Any] | None) -> dict[str, Any] | None:
    """Get cached model artifact from dcc.Store payload."""
    key = str((store_data or {}).get("cache_key") or "")
    if not key:
        return None
    artifact = MODEL_ARTIFACT_CACHE.get(key)
    return artifact if isinstance(artifact, dict) else None


def _decode_base64_upload(contents: str) -> bytes:
    """Decode Dash upload payload."""
    try:
        _header, encoded = contents.split(",", 1)
    except ValueError as exc:
        raise ValueError("アップロードデータの形式が不正です。") from exc
    try:
        return base64.b64decode(encoded)
    except Exception as exc:
        raise ValueError("アップロードデータのデコードに失敗しました。") from exc


def _loaded_model_summary_text(store_data: dict[str, Any] | None) -> str:
    """Build short summary text for loaded model store."""
    meta = (store_data or {}).get("meta") or {}
    if not isinstance(meta, dict) or not meta:
        return ""
    label = str(meta.get("model_label") or meta.get("model_key") or "モデル")
    task = str(meta.get("task") or "")
    target = meta.get("target_col")
    target_text = f" | 目的変数={target}" if target else ""
    feature_count = meta.get("feature_count")
    feat_text = f" | 説明変数数={feature_count}" if feature_count is not None else ""
    task_text = f" | タスク={task}" if task else ""
    return f"現在のモデル: {label}{task_text}{target_text}{feat_text}"


def _sql_connection_state_payload(
    *,
    dbms: str,
    server: str = "",
    port: int | None = None,
    database: str = "",
    schema: str = "",
    sqlite_path: str = "",
    username: str = "",
    password: str = "",
    connected: bool = False,
    tables: list[str] | None = None,
) -> dict[str, Any]:
    """Build normalized SQL connection store payload."""
    return {
        "connected": bool(connected),
        "dbms": normalize_dbms(dbms),
        "server": str(server or "").strip(),
        "port": normalize_port(port),
        "database": str(database or "").strip(),
        "schema": str(schema or "").strip(),
        "sqlite_path": str(sqlite_path or "").strip(),
        "username": str(username or "").strip(),
        "password": str(password or ""),
        "tables": list(tables or []),
    }


def _db_config_from_sql_state(sql_state: dict[str, Any] | None) -> DatabaseConfig:
    """Convert sql-connection-store payload into DatabaseConfig."""
    state = sql_state or {}
    return DatabaseConfig(
        dbms=normalize_dbms(state.get("dbms")),
        host=str(state.get("server", "")),
        port=normalize_port(state.get("port")),
        database=str(state.get("database", "")),
        schema=str(state.get("schema", "")),
        sqlite_path=str(state.get("sqlite_path", "")),
        username=str(state.get("username", "")),
        password=str(state.get("password", "")),
    )


def _pick_multi_columns(
    requested: list[str] | None,
    candidates: list[str],
    fallback_count: int,
) -> list[str]:
    """Pick valid columns from a multi-select list."""
    selected = [col for col in (requested or []) if col in candidates]
    if selected:
        return selected
    return candidates[:fallback_count]


def _as_id_list(value: Any) -> list[str]:
    """Normalize selectedData customdata payload into id list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        if len(value) == 1 and not isinstance(value[0], (list, tuple, dict)):
            return [str(value[0])]
        if all(not isinstance(item, (list, tuple, dict)) for item in value):
            return [str(item) for item in value]
        flattened: list[str] = []
        for item in value:
            flattened.extend(_as_id_list(item))
        return flattened
    if isinstance(value, dict):
        return []
    return [str(value)]


def _extract_ids_from_selection(selection_data: dict[str, Any] | None) -> list[str]:
    """Extract row IDs from selectedData where points carry customdata."""
    if not selection_data:
        return []

    selected_ids: list[str] = []
    for point in selection_data.get("points", []):
        selected_ids.extend(_as_id_list(point.get("customdata")))
    return normalize_id_list(selected_ids)


def _extract_ids_from_distribution_selection(
    selection_data: dict[str, Any] | None,
    figure: dict[str, Any] | None,
) -> list[str]:
    """Extract IDs from histogram/box selectedData using trace customdata fallback."""
    if not selection_data:
        return []

    figure_data = (figure or {}).get("data", [])
    selected_ids: list[str] = []

    for point in selection_data.get("points", []):
        selected_ids.extend(_as_id_list(point.get("customdata")))

        curve_number = point.get("curveNumber")
        if not isinstance(curve_number, int) or curve_number < 0 or curve_number >= len(figure_data):
            continue

        trace_customdata = figure_data[curve_number].get("customdata")
        if trace_customdata is None:
            continue

        point_indexes: list[int] = []
        point_numbers = point.get("pointNumbers")
        if isinstance(point_numbers, list):
            point_indexes.extend([idx for idx in point_numbers if isinstance(idx, int)])

        point_number = point.get("pointNumber")
        if isinstance(point_number, int):
            point_indexes.append(point_number)

        for idx in point_indexes:
            if idx < 0:
                continue
            if isinstance(trace_customdata, (list, tuple)) and idx < len(trace_customdata):
                selected_ids.extend(_as_id_list(trace_customdata[idx]))

    return normalize_id_list(selected_ids)


def _query_requests_graphs(search: str | None) -> bool:
    """Return True if query string asks to show graph setting area."""
    if not search:
        return False
    parsed = parse_qs(search.lstrip("?"))
    flag = str(parsed.get("show_graphs", ["0"])[0]).lower()
    return flag in {"1", "true", "yes", "on"}


def _query_window_graph(search: str | None) -> str | None:
    """Extract graph key from query string."""
    if not search:
        return None
    parsed = parse_qs(search.lstrip("?"))
    graph = str(parsed.get("graph", [""])[0]).strip().lower()
    if graph == "dist":
        graph = "hist"
    return graph if graph in VISIBLE_GRAPH_KEYS else None


def _graph_section_style(show_graphs: bool) -> dict[str, str]:
    """Return graph section display style."""
    if show_graphs:
        return {"display": "block", "marginTop": "12px"}
    return {"display": "none", "marginTop": "12px"}


def _graph_card_style(visible: bool, height: str = "420px") -> dict[str, str]:
    """Return graph card style with visibility state."""
    if not visible:
        return {"display": "none"}
    return {
        "display": "block",
        "height": height,
        "minHeight": "260px",
        "minWidth": "320px",
        "resize": "both",
        "overflow": "hidden",
        "border": "1px solid #d9d9d9",
        "borderRadius": "8px",
        "padding": "8px",
        "backgroundColor": "#fff",
    }


def _window_link_style(enabled: bool) -> dict[str, str]:
    """Return style for graph window links."""
    if enabled:
        return {"marginRight": "10px"}
    return {
        "marginRight": "10px",
        "color": "#999",
        "pointerEvents": "none",
        "textDecoration": "none",
    }


def _ranking_columns() -> list[dict[str, str]]:
    """Return ranking table schema."""
    labels = {
        "variable": "変数",
        "variable_type": "型",
        "test": "検定",
        "p_value": "p値",
        "statistic": "統計量",
        "effect_size": "効果量d",
        "mean_diff": "平均差",
    }
    return [{"name": labels[col], "id": col} for col in RANKING_COLUMNS]


def _plot_columns(columns: list[str], *, id_col: str = "id") -> list[str]:
    """Build graph-selectable column order (non-id first, id last)."""
    non_id = [col for col in columns if col != id_col]
    if id_col in columns:
        return [*non_id, id_col]
    return non_id


def _extract_type_overrides(
    dtype_rows: list[dict[str, Any]] | None,
    valid_columns: list[str],
) -> dict[str, str]:
    """Extract `{column: cast_type}` map from dtype config table rows."""
    valid_set = set(valid_columns)
    overrides: dict[str, str] = {}
    for row in dtype_rows or []:
        col = str((row or {}).get("column") or "")
        if col not in valid_set:
            continue
        cast_type = str((row or {}).get("target_type") or "auto").lower()
        if cast_type not in SUPPORTED_CAST_TYPES:
            cast_type = "auto"
        overrides[col] = cast_type
    return overrides


def _analysis_option_flags(values: list[str] | None) -> tuple[bool, bool]:
    """Normalize analysis option checklist into boolean flags."""
    selected = set(values or [])
    return (
        "exclude_missing_rows" in selected,
        "treat_selected_as_missing" in selected,
    )


def _standardize_enabled(values: list[str] | None) -> bool:
    """Return whether standardization is enabled."""
    return "on" in set(values or [])


def _split_method_label(method: str) -> str:
    """Convert split method key to Japanese label."""
    labels = {
        "random": "ランダム",
        "stratified_random": "層別ランダム",
        "sequential": "前後",
    }
    return labels.get(method, method)


def _build_modeling_summary(metadata: dict[str, Any]) -> html.Div:
    """Render modeling preparation summary panel."""
    if not metadata:
        return html.Div()

    items: list[str] = [
        f"分割: {_split_method_label(str(metadata.get('split_method', '')))}",
        f"学習/テスト: {metadata.get('train_count', 0)} / {metadata.get('test_count', 0)}",
    ]

    standardized_cols = metadata.get("standardized_cols", [])
    lag_cols = metadata.get("lag_cols", [])
    feature_cols = metadata.get("feature_cols", [])

    if standardized_cols:
        items.append(f"標準化列: {len(standardized_cols)}")
    if lag_cols:
        items.append(f"追加ラグ列: {len(lag_cols)}")
    if feature_cols:
        items.append(f"追加特徴量: {len(feature_cols)}")

    warnings: list[str] = list(metadata.get("warnings", []))
    warning_lines = [html.Li(msg) for msg in warnings[:8]]

    children: list[Any] = [html.Div(" | ".join(items))]
    if warning_lines:
        children.append(html.Ul(warning_lines, style={"marginTop": "6px"}))
    return html.Div(children)


def _prepare_modeling_runtime_dataframe(
    current_data: dict[str, Any],
    ui_config: dict[str, Any] | None,
    selected_ids: list[str] | None,
) -> pd.DataFrame:
    """Build dataframe for model training from current runtime settings.

    Notes:
        - lag/feature generation is applied from shared config.
        - global analysis-only standardization is disabled for training input;
          model training module performs train-based standardization internally.
    """
    cfg = dict(ui_config or {})
    typed_df = _prepare_typed_dataframe(current_data, cfg)
    modeling_cfg = dict(cfg)
    modeling_cfg["apply_standardize"] = False
    modeled_df, _ = _apply_modeling_config(typed_df, modeling_cfg)
    filtered_df = apply_analysis_filters(
        modeled_df,
        exclude_missing_rows=bool(cfg.get("exclude_missing_rows", False)),
        selected_ids=normalize_id_list(selected_ids),
        treat_selected_as_missing=bool(cfg.get("treat_selected_as_missing", False)),
        id_col="id",
    )
    return filtered_df


def _render_modeling_result_panel(result: dict[str, Any]) -> html.Div:
    """Render model run result into Dash components."""
    metrics_df: pd.DataFrame = result.get("metrics", pd.DataFrame()).copy()
    metric_records: list[dict[str, Any]] = []
    metric_columns: list[dict[str, str]] = []
    if not metrics_df.empty:
        for col in metrics_df.columns:
            if pd.api.types.is_numeric_dtype(metrics_df[col]):
                metrics_df[col] = metrics_df[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.6g}")
        metric_records = metrics_df.to_dict("records")
        metric_columns = [{"name": str(col), "id": str(col)} for col in metrics_df.columns]

    figures = list(result.get("figures", []))
    graph_nodes = [
        dcc.Graph(
            figure=fig,
            config={"displaylogo": False, "responsive": True},
            style={"height": "420px", "width": "100%"},
        )
        for fig in figures
    ]

    importance_sections: list[Any] = []
    for item in result.get("importance_tables", []) or []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "変数重要度")
        table_df = item.get("data")
        if not isinstance(table_df, pd.DataFrame) or table_df.empty:
            continue
        display_df = table_df.copy()
        for col in display_df.columns:
            if pd.api.types.is_numeric_dtype(display_df[col]):
                display_df[col] = display_df[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.6g}")
        importance_sections.append(
            html.Div(
                [
                    html.H5(title, style={"margin": "10px 0 6px 0"}),
                    dash_table.DataTable(
                        columns=[{"name": str(col), "id": str(col)} for col in display_df.columns],
                        data=display_df.to_dict("records"),
                        style_table={"overflowX": "auto"},
                        style_cell={"fontSize": 12, "textAlign": "left"},
                        page_size=15,
                    ),
                ]
            )
        )

    for item in result.get("importance_figures", []) or []:
        if not isinstance(item, dict):
            continue
        fig = item.get("figure")
        if fig is None:
            continue
        title = str(item.get("title") or "重要度可視化")
        importance_sections.append(
            html.Div(
                [
                    html.H5(title, style={"margin": "10px 0 6px 0"}),
                    dcc.Graph(
                        figure=fig,
                        config={"displaylogo": False, "responsive": True},
                        style={"height": "420px", "width": "100%"},
                    ),
                ]
            )
        )
    for item in result.get("extra_text_blocks", []) or []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "")
        text = str(item.get("text") or "")
        if not text:
            continue
        importance_sections.append(
            html.Div(
                [
                    html.H5(title, style={"margin": "10px 0 6px 0"}) if title else html.Div(),
                    html.Pre(text, style={"backgroundColor": "#f7f7f7", "padding": "8px", "fontSize": 12, "whiteSpace": "pre-wrap"}),
                ]
            )
        )

    notes = [str(item) for item in result.get("notes", []) if str(item).strip()]
    used_params = result.get("used_params", {})
    params_text = format_param_text(used_params) if isinstance(used_params, dict) else "{}"

    return html.Div(
        [
            html.H4(result.get("model_label", "モデリング結果")),
            html.Pre(params_text, style={"backgroundColor": "#f7f7f7", "padding": "8px", "fontSize": 12}),
            dash_table.DataTable(
                columns=metric_columns,
                data=metric_records,
                style_table={"overflowX": "auto"},
                style_cell={"fontSize": 12, "textAlign": "left"},
            ),
            html.Div(
                graph_nodes,
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(420px, 1fr))",
                    "gap": "10px",
                    "marginTop": "8px",
                },
            ),
            html.Div(importance_sections, style={"marginTop": "8px"}) if importance_sections else html.Div(),
            html.Ul([html.Li(note) for note in notes], style={"marginTop": "8px"}) if notes else html.Div(),
        ]
    )


def _prepare_typed_dataframe(
    current_data: dict[str, Any],
    ui_config: dict[str, Any] | None,
) -> pd.DataFrame:
    """Deserialize dataframe and apply dtype overrides."""
    base_df = prepare_dataframe(dataframe_from_json(current_data["df_json"]))
    overrides = normalize_type_overrides((ui_config or {}).get("type_overrides"), base_df.columns, id_col="id")
    return apply_type_overrides(base_df, overrides, id_col="id")


def _apply_modeling_config(
    df: pd.DataFrame,
    ui_config: dict[str, Any] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply modeling preparation settings to dataframe."""
    cfg = ui_config or {}
    return apply_modeling_preparation(
        df,
        split_method=normalize_split_method(cfg.get("split_method")),
        train_ratio=normalize_train_ratio(cfg.get("train_ratio")),
        random_seed=normalize_random_seed(cfg.get("split_seed")),
        stratify_col=(cfg.get("split_stratify_col") or None),
        order_col=(cfg.get("split_order_col") or None),
        standardize=bool(cfg.get("apply_standardize", False)),
        lag_text=str(cfg.get("lag_config_text") or ""),
        feature_text=str(cfg.get("feature_config_text") or ""),
    )


def _to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert dataframe to JSON-safe records with missing values as None."""
    safe_df = df.copy()
    for col in safe_df.columns:
        if pd.api.types.is_datetime64_any_dtype(safe_df[col]):
            safe_df[col] = safe_df[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    safe_df = safe_df.astype(object).where(pd.notna(safe_df), None)
    return safe_df.to_dict("records")


def _is_additive_selection(key_state: dict[str, Any] | None) -> bool:
    """Return whether Ctrl(Command) is currently pressed."""
    state = key_state or {}
    return bool(state.get("ctrl") or state.get("meta"))


def _build_summary_panel(
    df: pd.DataFrame,
    selected_ids: list[str],
    numeric_cols: list[str],
) -> html.Div:
    """Build selected vs non-selected summary section."""
    if not selected_ids:
        return html.Div("未選択 (No selection)")

    selected_mask = df["id"].astype(str).isin(set(selected_ids))
    n_selected = int(selected_mask.sum())
    n_others = int((~selected_mask).sum())

    if n_selected == 0 or n_others == 0:
        return html.Div(
            [
                html.P(f"選択: {n_selected}"),
                html.P(f"非選択: {n_others}"),
                html.P("比較には選択群と非選択群の両方にデータが必要です。"),
            ]
        )

    lines: list[Any] = [
        html.P(f"選択: {n_selected}"),
        html.P(f"非選択: {n_others}"),
    ]
    if not numeric_cols:
        lines.append(html.P("数値列がないため平均差サマリを表示できません。"))
        return html.Div(lines)

    selected_mean = df.loc[selected_mask, numeric_cols].mean(numeric_only=True)
    other_mean = df.loc[~selected_mask, numeric_cols].mean(numeric_only=True)
    diff = (selected_mean - other_mean).sort_values(key=lambda series: series.abs(), ascending=False).head(5)

    header = html.Thead(
        html.Tr([html.Th("列"), html.Th("選択群平均"), html.Th("非選択群平均"), html.Th("差分")])
    )
    rows = html.Tbody(
        [
            html.Tr(
                [
                    html.Td(col),
                    html.Td(f"{selected_mean[col]:.4g}"),
                    html.Td(f"{other_mean[col]:.4g}"),
                    html.Td(f"{diff[col]:.4g}"),
                ]
            )
            for col in diff.index
        ]
    )
    lines.append(html.Table([header, rows], style={"width": "100%", "fontSize": 12}))
    return html.Div(lines)


def _is_current_run_data(
    current_data: dict[str, Any] | None,
    app_run_data: dict[str, Any] | None,
) -> bool:
    """Return True when dataset belongs to current app process run."""
    if not current_data:
        return False

    has_dataset = bool(current_data.get("df_json"))
    if not has_dataset:
        return True

    app_run_id = str((app_run_data or {}).get("app_run_id") or "")
    stored_run_id = str(current_data.get("app_run_id") or "")
    return bool(app_run_id and stored_run_id == app_run_id)


def register_callbacks(app: Dash) -> None:
    """Register Dash callbacks."""

    app.clientside_callback(
        """
        function(n) {
            if (window.insightaKeyboardState) {
                return {
                    ctrl: !!window.insightaKeyboardState.ctrl,
                    meta: !!window.insightaKeyboardState.meta
                };
            }
            return {ctrl: false, meta: false};
        }
        """,
        Output("keyboard-state-store", "data"),
        Input("keyboard-poll-interval", "n_intervals"),
    )

    @app.callback(
        Output("current-data-store", "clear_data"),
        Output("selected-ids-store", "clear_data"),
        Output("ui-config-store", "clear_data"),
        Output("view-config-store", "clear_data"),
        Input("app-run-store", "data"),
        State("current-data-store", "data"),
    )
    def clear_stale_local_session(
        app_run_data: dict[str, Any] | None,
        current_data: dict[str, Any] | None,
    ) -> tuple[bool, bool, bool, bool]:
        """Clear persisted local stores when app process has restarted."""
        app_run_id = str((app_run_data or {}).get("app_run_id") or "")
        if not app_run_id:
            return False, False, False, False

        current = current_data or {}
        stored_run_id = str(current.get("app_run_id") or "")
        has_dataset = bool(current.get("df_json"))

        if has_dataset and stored_run_id and stored_run_id != app_run_id:
            return True, True, True, True
        if has_dataset and not stored_run_id:
            return True, True, True, True
        return False, False, False, False

    @app.callback(
        Output("shutdown-status", "children"),
        Output("shutdown-button", "disabled"),
        Input("shutdown-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def shutdown_application(n_clicks: int | None) -> tuple[str, bool]:
        """Shutdown local app server from the UI button."""
        if not n_clicks:
            return no_update, no_update

        shutdown_callable = None
        try:
            shutdown_callable = request.environ.get("werkzeug.server.shutdown")
        except Exception:
            shutdown_callable = None

        _schedule_server_shutdown(shutdown_callable)
        return "アプリを終了しています... ブラウザを閉じてください。", True

    @app.callback(
        Output("sql-connection-store", "data"),
        Output("sql-table-dropdown", "options"),
        Output("sql-table-dropdown", "value"),
        Output("sql-connect-status", "children"),
        Input("sql-connect-button", "n_clicks"),
        State("sql-dbms-dropdown", "value"),
        State("sql-server-input", "value"),
        State("sql-port-input", "value"),
        State("sql-database-input", "value"),
        State("sql-schema-input", "value"),
        State("sql-sqlite-path-input", "value"),
        State("sql-username-input", "value"),
        State("sql-password-input", "value"),
        prevent_initial_call=True,
    )
    def connect_sql_database(
        _n_clicks: int | None,
        dbms: str | None,
        server: str | None,
        port: int | float | None,
        database: str | None,
        schema: str | None,
        sqlite_path: str | None,
        username: str | None,
        password: str | None,
    ) -> tuple[dict[str, Any], list[dict[str, str]], str | None, str]:
        dbms_key = normalize_dbms(dbms)
        base_state = _sql_connection_state_payload(
            dbms=dbms_key,
            server=server or "",
            port=port,
            database=database or "",
            schema=schema or "",
            sqlite_path=sqlite_path or "",
            username=username or "",
            password=password or "",
            connected=False,
            tables=[],
        )
        try:
            config = _db_config_from_sql_state(base_state)
            tables = list_tables(config)
        except DatabaseError as exc:
            return base_state, [], None, f"{dbms_label(dbms_key)} 接続に失敗しました: {exc}"

        options = [{"label": name, "value": name} for name in tables]
        connected_state = dict(base_state)
        connected_state["connected"] = True
        connected_state["tables"] = tables
        default_value = tables[0] if tables else None
        return connected_state, options, default_value, f"接続成功: {len(tables)} テーブルを取得しました。"

    @app.callback(
        Output("sql-query-text", "value"),
        Input("sql-build-query-button", "n_clicks"),
        State("sql-table-dropdown", "value"),
        State("sql-topn-input", "value"),
        State("sql-query-text", "value"),
        State("sql-connection-store", "data"),
        prevent_initial_call=True,
    )
    def build_sql_query(
        _n_clicks: int | None,
        table_name: str | None,
        top_n: int | float | None,
        current_query: str | None,
        sql_state: dict[str, Any] | None,
    ) -> str | Any:
        if not table_name:
            return current_query or ""
        safe_top_n = int(top_n) if isinstance(top_n, (int, float)) else 1000
        config = _db_config_from_sql_state(sql_state)
        return build_select_sample_query(config, table_name, top_n=safe_top_n)

    @app.callback(
        Output("current-data-store", "data"),
        Output("upload-status", "children"),
        Output("selected-ids-store", "data", allow_duplicate=True),
        Input("upload-data", "contents"),
        Input("sql-run-query-button", "n_clicks"),
        State("upload-data", "filename"),
        State("app-run-store", "data"),
        State("sql-connection-store", "data"),
        State("sql-query-text", "value"),
        prevent_initial_call=True,
    )
    def load_dataset(
        upload_contents: str | None,
        _sql_run_clicks: int | None,
        upload_filename: str | None,
        app_run_data: dict[str, Any] | None,
        sql_state: dict[str, Any] | None,
        sql_query: str | None,
    ) -> tuple[dict[str, Any] | Any, str, list[str] | Any]:
        trigger = callback_context.triggered_id
        app_run_id = str((app_run_data or {}).get("app_run_id") or "")

        if trigger == "upload-data":
            if not upload_contents:
                return no_update, "アップロードデータが空です。", no_update
            try:
                uploaded_df = prepare_dataframe(load_dataset_from_upload(upload_contents, upload_filename))
                source_name = upload_filename or "uploaded"
                current_data = build_current_data_state(uploaded_df, source_name, app_run_id=app_run_id)
                return current_data, f"ファイルを読み込みました: {source_name}", []
            except DataLoadError as exc:
                return no_update, f"ファイル読み込みに失敗しました: {exc}", no_update

        if trigger == "sql-run-query-button":
            state = sql_state or {}
            if not state.get("connected"):
                return no_update, "先に DB 接続を実行してください。", no_update

            query_text = (sql_query or "").strip()
            if not query_text:
                return no_update, "SQLクエリを入力してください。", no_update

            config = _db_config_from_sql_state(state)
            try:
                df = prepare_dataframe(execute_query(config, query_text))
                db_label = dbms_label(config.dbms)
                if normalize_dbms(config.dbms) == "sqlite":
                    source_name = f"{db_label}: {config.sqlite_path}"
                else:
                    source_name = f"{db_label}: {config.host}/{config.database}"
                current_data = build_current_data_state(df, source_name, app_run_id=app_run_id)
                return current_data, f"SQL結果を読み込みました: {len(df)} 行", []
            except (DatabaseError, DataLoadError) as exc:
                return no_update, f"SQL実行に失敗しました: {exc}", no_update

        return no_update, "データ読み込み操作が実行されませんでした。", no_update

    @app.callback(
        Output("dtype-config-table", "data"),
        Input("current-data-store", "data"),
        Input("app-run-store", "data"),
        State("ui-config-store", "data"),
    )
    def refresh_dtype_config_table(
        current_data: dict[str, Any] | None,
        app_run_data: dict[str, Any] | None,
        ui_config: dict[str, Any] | None,
    ) -> list[dict[str, str]]:
        """Populate dtype override table from current dataframe."""
        if not _is_current_run_data(current_data, app_run_data) or not current_data or not current_data.get("df_json"):
            return []

        df = prepare_dataframe(dataframe_from_json(current_data["df_json"]))
        existing_overrides = normalize_type_overrides((ui_config or {}).get("type_overrides"), df.columns, id_col="id")
        rows: list[dict[str, str]] = []
        for col in df.columns:
            if col == "id":
                continue
            rows.append(
                {
                    "column": col,
                    "current_type": infer_series_kind_label(df[col]),
                    "target_type": existing_overrides.get(col, "auto"),
                }
            )
        return rows

    @app.callback(
        Output("ui-config-store", "data"),
        Input("current-data-store", "data"),
        Input("visible-graphs-checklist", "value"),
        Input("analysis-options-check", "value"),
        Input("dtype-config-table", "data"),
        Input("split-method-dropdown", "value"),
        Input("split-ratio-input", "value"),
        Input("split-seed-input", "value"),
        Input("split-stratify-column-dropdown", "value"),
        Input("split-order-column-dropdown", "value"),
        Input("standardize-check", "value"),
        Input("lag-config-text", "value"),
        Input("feature-config-text", "value"),
        Input("show-graphs-button", "n_clicks"),
        Input("url-location", "search"),
        State("ui-config-store", "data"),
    )
    def sync_shared_ui_config(
        current_data: dict[str, Any] | None,
        visible_graphs: list[str] | None,
        analysis_options: list[str] | None,
        dtype_table_rows: list[dict[str, Any]] | None,
        split_method_value: str | None,
        split_ratio_value: float | int | None,
        split_seed_value: float | int | None,
        split_stratify_col: str | None,
        split_order_col: str | None,
        standardize_check: list[str] | None,
        lag_config_text: str | None,
        feature_config_text: str | None,
        _show_graphs_clicks: int | None,
        url_search: str | None,
        existing: dict[str, Any] | None,
    ) -> dict[str, Any]:
        metadata = (current_data or {}).get("metadata", {})
        all_cols: list[str] = _plot_columns(list(metadata.get("columns", [])))
        non_id_cols: list[str] = [col for col in all_cols if col != "id"]
        exclude_missing_rows, treat_selected_as_missing = _analysis_option_flags(analysis_options)
        standardize_enabled = _standardize_enabled(standardize_check)

        trigger = callback_context.triggered_id
        if trigger == "current-data-store" or not existing:
            config = build_default_ui_config(metadata)
            if existing:
                config["show_graphs"] = bool(existing.get("show_graphs", False))
                config["visible_graphs"] = _normalize_visible_graphs(existing.get("visible_graphs"))
                config["split_method"] = normalize_split_method(existing.get("split_method"))
                config["train_ratio"] = normalize_train_ratio(existing.get("train_ratio"))
                config["split_seed"] = normalize_random_seed(existing.get("split_seed"))
                config["split_stratify_col"] = existing.get("split_stratify_col")
                config["split_order_col"] = existing.get("split_order_col")
                config["apply_standardize"] = bool(existing.get("apply_standardize", False))
                config["lag_config_text"] = str(existing.get("lag_config_text") or "")
                config["feature_config_text"] = str(existing.get("feature_config_text") or "")
            if _query_requests_graphs(url_search):
                config["show_graphs"] = True
            return config

        current_overrides = normalize_type_overrides((existing or {}).get("type_overrides"), non_id_cols, id_col="id")
        if dtype_table_rows is not None:
            current_overrides = _extract_type_overrides(dtype_table_rows, non_id_cols)

        config = dict(existing)
        config["visible_graphs"] = _normalize_visible_graphs(visible_graphs)
        if dtype_table_rows is not None:
            config["type_overrides"] = current_overrides
        if analysis_options is not None:
            config["exclude_missing_rows"] = exclude_missing_rows
            config["treat_selected_as_missing"] = treat_selected_as_missing
        if split_method_value is not None:
            config["split_method"] = normalize_split_method(split_method_value)
        if split_ratio_value is not None:
            config["train_ratio"] = normalize_train_ratio(split_ratio_value)
        if split_seed_value is not None:
            config["split_seed"] = normalize_random_seed(split_seed_value)
        if split_stratify_col is not None:
            config["split_stratify_col"] = split_stratify_col
        if split_order_col is not None:
            config["split_order_col"] = split_order_col
        if standardize_check is not None:
            config["apply_standardize"] = standardize_enabled
        if lag_config_text is not None:
            config["lag_config_text"] = str(lag_config_text)
        if feature_config_text is not None:
            config["feature_config_text"] = str(feature_config_text)

        if trigger == "show-graphs-button":
            config["show_graphs"] = not bool(config.get("show_graphs", False))
        elif trigger == "url-location" and _query_requests_graphs(url_search):
            config["show_graphs"] = True
        else:
            config["show_graphs"] = bool(config.get("show_graphs", False))

        return config

    @app.callback(
        Output("view-config-store", "data"),
        Input("current-data-store", "data"),
        Input("x-dropdown", "value"),
        Input("y-dropdown", "value"),
        Input("hist-column-dropdown", "value"),
        Input("box-column-dropdown", "value"),
        Input("matrix-columns-dropdown", "value"),
        State("view-config-store", "data"),
        State("ui-config-store", "data"),
    )
    def sync_view_config(
        current_data: dict[str, Any] | None,
        x_col: str | None,
        y_col: str | None,
        hist_col: str | None,
        box_col: str | None,
        matrix_cols: list[str] | None,
        existing: dict[str, Any] | None,
        ui_config: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not current_data or not current_data.get("df_json"):
            return build_default_view_config({})

        typed_df = _prepare_typed_dataframe(current_data, ui_config or {})
        modeled_df, _ = _apply_modeling_config(typed_df, ui_config or {})
        runtime_metadata = build_runtime_metadata(modeled_df, id_col="id")
        all_cols = _plot_columns(list(runtime_metadata.get("columns", [])))
        numeric_cols = list(runtime_metadata.get("numeric_cols", []))

        trigger = callback_context.triggered_id
        if trigger == "current-data-store" or not existing:
            return build_default_view_config(runtime_metadata)

        config = dict(existing)
        config["x_col"] = pick_column(x_col, all_cols, 0)
        config["y_col"] = pick_column(y_col, all_cols, 1)
        config["hist_col"] = pick_column(hist_col, all_cols, 0)
        config["box_col"] = pick_column(box_col, numeric_cols, 0)
        config["matrix_cols"] = _pick_multi_columns(matrix_cols, numeric_cols, min(len(numeric_cols), 4))
        return config

    @app.callback(
        Output("x-dropdown", "options"),
        Output("y-dropdown", "options"),
        Output("hist-column-dropdown", "options"),
        Output("box-column-dropdown", "options"),
        Output("matrix-columns-dropdown", "options"),
        Output("split-stratify-column-dropdown", "options"),
        Output("split-order-column-dropdown", "options"),
        Output("x-dropdown", "value"),
        Output("y-dropdown", "value"),
        Output("hist-column-dropdown", "value"),
        Output("box-column-dropdown", "value"),
        Output("matrix-columns-dropdown", "value"),
        Output("visible-graphs-checklist", "value"),
        Output("analysis-options-check", "value"),
        Output("split-method-dropdown", "value"),
        Output("split-ratio-input", "value"),
        Output("split-seed-input", "value"),
        Output("split-stratify-column-dropdown", "value"),
        Output("split-order-column-dropdown", "value"),
        Output("standardize-check", "value"),
        Output("lag-config-text", "value"),
        Output("feature-config-text", "value"),
        Input("current-data-store", "data"),
        Input("app-run-store", "data"),
        Input("dtype-config-table", "data"),
        State("ui-config-store", "data"),
        State("view-config-store", "data"),
    )
    def refresh_column_controls(
        current_data: dict[str, Any] | None,
        app_run_data: dict[str, Any] | None,
        dtype_table_rows: list[dict[str, Any]] | None,
        ui_config: dict[str, Any] | None,
        view_config: dict[str, Any] | None,
    ) -> tuple[
        list[dict[str, str]],
        list[dict[str, str]],
        list[dict[str, str]],
        list[dict[str, str]],
        list[dict[str, str]],
        list[dict[str, str]],
        list[dict[str, str]],
        str | None,
        str | None,
        str | None,
        str | None,
        list[str],
        list[str],
        list[str],
        str,
        float,
        int,
        str | None,
        str | None,
        list[str],
        str,
        str,
    ]:
        cfg = ui_config or {}
        view_cfg = view_config or {}
        default_visible = _normalize_visible_graphs(cfg.get("visible_graphs"))
        analysis_values: list[str] = []
        if cfg.get("exclude_missing_rows"):
            analysis_values.append("exclude_missing_rows")
        if cfg.get("treat_selected_as_missing"):
            analysis_values.append("treat_selected_as_missing")
        standardize_values = ["on"] if cfg.get("apply_standardize") else []
        split_method = normalize_split_method(cfg.get("split_method"))
        train_ratio = normalize_train_ratio(cfg.get("train_ratio"))
        split_seed = normalize_random_seed(cfg.get("split_seed"))
        lag_config_text = str(cfg.get("lag_config_text") or "")
        feature_config_text = str(cfg.get("feature_config_text") or "")
        split_stratify_value = cfg.get("split_stratify_col")
        split_order_value = cfg.get("split_order_col")

        if not _is_current_run_data(current_data, app_run_data) or not current_data or not current_data.get("df_json"):
            empty_opts: list[dict[str, str]] = []
            return (
                empty_opts,
                empty_opts,
                empty_opts,
                empty_opts,
                empty_opts,
                empty_opts,
                empty_opts,
                None,
                None,
                None,
                None,
                [],
                default_visible,
                analysis_values,
                split_method,
                train_ratio,
                split_seed,
                split_stratify_value,
                split_order_value,
                standardize_values,
                lag_config_text,
                feature_config_text,
            )

        effective_cfg = dict(cfg)
        metadata_columns = list((current_data or {}).get("metadata", {}).get("columns", []))
        non_id_cols = [col for col in metadata_columns if col != "id"]
        if dtype_table_rows is not None:
            effective_cfg["type_overrides"] = _extract_type_overrides(dtype_table_rows, non_id_cols)

        typed_df = _prepare_typed_dataframe(current_data, effective_cfg)
        modeled_df, _ = _apply_modeling_config(typed_df, effective_cfg)
        runtime_metadata = build_runtime_metadata(modeled_df, id_col="id")
        all_cols: list[str] = _plot_columns(list(runtime_metadata.get("columns", [])))
        numeric_cols: list[str] = list(runtime_metadata.get("numeric_cols", []))

        all_options = [{"label": col, "value": col} for col in all_cols]
        box_options = [{"label": col, "value": col} for col in numeric_cols]
        matrix_options = [{"label": col, "value": col} for col in numeric_cols]
        split_col_options = [{"label": col, "value": col} for col in _plot_columns(list(typed_df.columns))]

        x_col = pick_column(view_cfg.get("x_col"), all_cols, 0)
        y_col = pick_column(view_cfg.get("y_col"), all_cols, 1)
        hist_col = pick_column(view_cfg.get("hist_col"), all_cols, 0)
        box_col = pick_column(view_cfg.get("box_col"), numeric_cols, 0)
        matrix_cols = _pick_multi_columns(view_cfg.get("matrix_cols"), numeric_cols, min(len(numeric_cols), 4))
        if split_stratify_value not in [opt["value"] for opt in split_col_options]:
            split_stratify_value = None
        if split_order_value not in [opt["value"] for opt in split_col_options]:
            split_order_value = None

        return (
            all_options,
            all_options,
            all_options,
            box_options,
            matrix_options,
            split_col_options,
            split_col_options,
            x_col,
            y_col,
            hist_col,
            box_col,
            matrix_cols,
            default_visible,
            analysis_values,
            split_method,
            train_ratio,
            split_seed,
            split_stratify_value,
            split_order_value,
            standardize_values,
            lag_config_text,
            feature_config_text,
        )

    @app.callback(
        Output("model-target-dropdown", "options"),
        Output("model-features-dropdown", "options"),
        Output("model-target-dropdown", "value"),
        Output("model-features-dropdown", "value"),
        Output("model-target-dropdown", "disabled"),
        Output("model-target-help", "children"),
        Input("current-data-store", "data"),
        Input("app-run-store", "data"),
        Input("ui-config-store", "data"),
        Input("model-method-dropdown", "value"),
        State("model-target-dropdown", "value"),
        State("model-features-dropdown", "value"),
    )
    def refresh_modeling_controls(
        current_data: dict[str, Any] | None,
        app_run_data: dict[str, Any] | None,
        ui_config: dict[str, Any] | None,
        model_method: str | None,
        current_target: str | None,
        current_features: list[str] | None,
    ) -> tuple[
        list[dict[str, str]],
        list[dict[str, str]],
        str | None,
        list[str],
        bool,
        str,
    ]:
        model_key = str(model_method or default_model_key())
        requires_target = model_requires_target(model_key)
        task = model_task(model_key)
        help_text = (
            "教師なしモデルでは目的変数は使用しません。"
            if not requires_target
            else ("回帰モデルの目的変数を選択してください。" if task == "regression" else "分類モデルの目的変数を選択してください。")
        )

        if not _is_current_run_data(current_data, app_run_data) or not current_data or not current_data.get("df_json"):
            return [], [], None, [], (not requires_target), help_text

        try:
            runtime_df = _prepare_modeling_runtime_dataframe(current_data, ui_config, [])
        except Exception:
            return [], [], None, [], (not requires_target), help_text

        candidate_cols = [col for col in _plot_columns(list(runtime_df.columns)) if col != "id"]
        target_options = [{"label": col, "value": col} for col in candidate_cols]

        if requires_target:
            target_value = pick_column(current_target, candidate_cols, 0)
        else:
            target_value = current_target if current_target in candidate_cols else None

        feature_candidates = candidate_cols.copy()
        if requires_target and target_value in feature_candidates:
            feature_candidates = [col for col in feature_candidates if col != target_value]
        feature_options = [{"label": col, "value": col} for col in feature_candidates]

        selected_features = [col for col in (current_features or []) if col in feature_candidates]
        if not selected_features:
            selected_features = feature_candidates[: min(8, len(feature_candidates))]

        return (
            target_options,
            feature_options,
            target_value,
            selected_features,
            (not requires_target),
            help_text,
        )

    @app.callback(
        Output("model-params-text", "value"),
        Output("modeling-suggest-status", "children"),
        Input("model-suggest-button", "n_clicks"),
        State("current-data-store", "data"),
        State("app-run-store", "data"),
        State("ui-config-store", "data"),
        State("selected-ids-store", "data"),
        State("model-method-dropdown", "value"),
        State("model-target-dropdown", "value"),
        State("model-features-dropdown", "value"),
        State("model-cv-fold-input", "value"),
        State("model-params-text", "value"),
        prevent_initial_call=True,
    )
    def suggest_model_hyperparams(
        _n_clicks: int | None,
        current_data: dict[str, Any] | None,
        app_run_data: dict[str, Any] | None,
        ui_config: dict[str, Any] | None,
        selected_ids: list[str] | None,
        model_method: str | None,
        target_col: str | None,
        feature_cols: list[str] | None,
        cv_folds: int | float | None,
        current_param_text: str | None,
    ) -> tuple[str | Any, str]:
        if not _is_current_run_data(current_data, app_run_data) or not current_data or not current_data.get("df_json"):
            return no_update, "先にデータを読み込んでください。"

        model_key = str(model_method or default_model_key())
        try:
            runtime_df = _prepare_modeling_runtime_dataframe(current_data, ui_config, selected_ids)
            if runtime_df.empty:
                return no_update, "モデリング対象データがありません。"

            cfg = ui_config or {}
            suggested, summary = suggest_hyperparameters(
                runtime_df,
                model_key=model_key,
                feature_cols=[str(col) for col in (feature_cols or [])],
                target_col=target_col,
                split_method=normalize_split_method(cfg.get("split_method")),
                train_ratio=normalize_train_ratio(cfg.get("train_ratio")),
                random_seed=normalize_random_seed(cfg.get("split_seed")),
                split_stratify_col=(cfg.get("split_stratify_col") or None),
                split_order_col=(cfg.get("split_order_col") or None),
                cv_folds=int(cv_folds) if isinstance(cv_folds, (int, float)) else 5,
            )
            return format_param_text(suggested), f"{model_label(model_key)} 推奨値: {summary}"
        except Exception as exc:
            fallback = current_param_text if isinstance(current_param_text, str) else "{}"
            return fallback, f"推奨ハイパーパラメータ計算に失敗しました: {exc}"

    @app.callback(
        Output("modeling-result-panel", "children"),
        Output("modeling-run-status", "children"),
        Output("model-artifact-store", "data"),
        Input("model-run-button", "n_clicks"),
        Input("current-data-store", "data"),
        State("app-run-store", "data"),
        State("ui-config-store", "data"),
        State("selected-ids-store", "data"),
        State("model-method-dropdown", "value"),
        State("model-target-dropdown", "value"),
        State("model-features-dropdown", "value"),
        State("model-params-text", "value"),
        prevent_initial_call=True,
    )
    def execute_modeling(
        _n_clicks: int | None,
        current_data: dict[str, Any] | None,
        app_run_data: dict[str, Any] | None,
        ui_config: dict[str, Any] | None,
        selected_ids: list[str] | None,
        model_method: str | None,
        target_col: str | None,
        feature_cols: list[str] | None,
        param_text: str | None,
    ) -> tuple[html.Div, str, dict[str, Any]]:
        trigger = callback_context.triggered_id
        if trigger == "current-data-store":
            return html.Div(), "", _model_store_payload(None)

        if not _is_current_run_data(current_data, app_run_data) or not current_data or not current_data.get("df_json"):
            return html.Div(), "先にデータを読み込んでください。", _model_store_payload(None)

        model_key = str(model_method or default_model_key())
        params, parse_error = parse_param_text(param_text)
        if parse_error:
            return html.Div(), parse_error, no_update

        try:
            runtime_df = _prepare_modeling_runtime_dataframe(current_data, ui_config, selected_ids)
            if runtime_df.empty:
                return html.Div(), "モデリング対象データがありません。", no_update

            cfg = ui_config or {}
            result = run_model(
                runtime_df,
                model_key=model_key,
                feature_cols=[str(col) for col in (feature_cols or [])],
                target_col=target_col,
                split_method=normalize_split_method(cfg.get("split_method")),
                train_ratio=normalize_train_ratio(cfg.get("train_ratio")),
                random_seed=normalize_random_seed(cfg.get("split_seed")),
                split_stratify_col=(cfg.get("split_stratify_col") or None),
                split_order_col=(cfg.get("split_order_col") or None),
                hyperparams=params,
            )
            artifact_bundle = result.get("artifact_bundle")
            if not isinstance(artifact_bundle, dict):
                artifact_bundle = {
                    "meta": {
                        "model_key": result.get("model_key"),
                        "model_label": result.get("model_label"),
                        "task": result.get("task"),
                        "target_col": target_col,
                        "feature_count": len(feature_cols or []),
                    },
                    "result": result,
                }
            elif "meta" not in artifact_bundle:
                artifact_bundle["meta"] = {
                    "model_key": result.get("model_key"),
                    "model_label": result.get("model_label"),
                    "task": result.get("task"),
                    "target_col": target_col,
                    "feature_count": len(feature_cols or []),
                }
            result_for_cache = dict(result)
            result_for_cache.pop("artifact_bundle", None)
            artifact_bundle["result"] = result_for_cache
            store_payload = _cache_model_artifact(artifact_bundle)
            return (
                _render_modeling_result_panel(result),
                f"{result.get('model_label', model_label(model_key))} を実行しました。",
                store_payload,
            )
        except Exception as exc:
            return html.Div(), f"モデル実行に失敗しました: {exc}", no_update

    @app.callback(
        Output("download-model-file", "data"),
        Output("modeling-model-io-status", "children", allow_duplicate=True),
        Input("model-save-button", "n_clicks"),
        State("model-artifact-store", "data"),
        prevent_initial_call=True,
    )
    def save_trained_model(
        _n_clicks: int | None,
        model_store: dict[str, Any] | None,
    ) -> tuple[Any, str]:
        artifact = _get_model_artifact_from_store(model_store)
        if not artifact:
            return no_update, "保存対象の学習済みモデルがありません。先にモデルを実行してください。"

        try:
            import joblib
        except ImportError as exc:
            return no_update, f"モデル保存に必要なライブラリが不足しています: {exc}"

        buf = io.BytesIO()
        try:
            joblib.dump(artifact, buf)
        except Exception as exc:
            return no_update, f"モデル保存に失敗しました: {exc}"
        buf.seek(0)
        meta = artifact.get("meta", {}) if isinstance(artifact, dict) else {}
        model_name = str(meta.get("model_key") or "model")
        filename = f"insighta_{model_name}.joblib"
        return dcc.send_bytes(buf.getvalue(), filename), f"学習済みモデルを保存しました: {filename}"

    @app.callback(
        Output("model-artifact-store", "data", allow_duplicate=True),
        Output("modeling-model-io-status", "children", allow_duplicate=True),
        Output("modeling-result-panel", "children", allow_duplicate=True),
        Output("modeling-run-status", "children", allow_duplicate=True),
        Output("loaded-model-summary", "children"),
        Input("model-upload-file", "contents"),
        State("model-upload-file", "filename"),
        prevent_initial_call=True,
    )
    def load_trained_model(
        upload_contents: str | None,
        upload_filename: str | None,
    ) -> tuple[dict[str, Any] | Any, str, Any, str, str]:
        if not upload_contents:
            return no_update, "モデルファイルが空です。", no_update, no_update, ""
        try:
            payload = _decode_base64_upload(upload_contents)
            try:
                import joblib
            except ImportError as exc:
                return no_update, f"モデル再読込に必要なライブラリが不足しています: {exc}", no_update, no_update, ""
            artifact = joblib.load(io.BytesIO(payload))
            if not isinstance(artifact, dict):
                return no_update, "モデルファイル形式が不正です。", no_update, no_update, ""
            if "meta" not in artifact:
                artifact["meta"] = {}
            store_payload = _cache_model_artifact(artifact)
            summary = _loaded_model_summary_text(store_payload)
            loaded_result = artifact.get("result")
            panel = _render_modeling_result_panel(loaded_result) if isinstance(loaded_result, dict) else no_update
            run_status = (
                f"再読込モデルの結果を表示しています: {artifact.get('meta', {}).get('model_label', '')}"
                if isinstance(loaded_result, dict)
                else no_update
            )
            return (
                store_payload,
                f"学習済みモデルを再読込しました: {upload_filename or 'model.joblib'}",
                panel,
                run_status,
                summary,
            )
        except Exception as exc:
            return no_update, f"学習済みモデルの再読込に失敗しました: {exc}", no_update, no_update, ""

    @app.callback(
        Output("download-table-file", "data"),
        Output("export-table-status", "children"),
        Input("export-table-csv-button", "n_clicks"),
        Input("export-table-xlsx-button", "n_clicks"),
        State("data-table", "data"),
        State("current-data-store", "data"),
        prevent_initial_call=True,
    )
    def export_processed_table(
        _csv_clicks: int | None,
        _xlsx_clicks: int | None,
        table_data: list[dict[str, Any]] | None,
        current_data: dict[str, Any] | None,
    ) -> tuple[Any, str]:
        trigger = callback_context.triggered_id
        if not table_data:
            return no_update, "出力対象のデータテーブルが空です。"

        df = pd.DataFrame(table_data)
        source_name = str((current_data or {}).get("source_name") or "insighta_data")
        safe_stem = (
            source_name.replace("\\", "_")
            .replace("/", "_")
            .replace(":", "_")
            .replace(" ", "_")
        )
        if "." in safe_stem:
            safe_stem = safe_stem.rsplit(".", 1)[0]
        if not safe_stem:
            safe_stem = "insighta_data"

        if trigger == "export-table-csv-button":
            return (
                dcc.send_data_frame(df.to_csv, f"{safe_stem}_processed.csv", index=False, encoding="utf-8-sig"),
                f"CSVを出力しました: {len(df)} 行",
            )
        if trigger == "export-table-xlsx-button":
            try:
                return (
                    dcc.send_data_frame(df.to_excel, f"{safe_stem}_processed.xlsx", index=False, sheet_name="data"),
                    f"Excelを出力しました: {len(df)} 行",
                )
            except ImportError as exc:
                return no_update, f"Excel出力に必要なライブラリが不足しています: {exc}"
            except Exception as exc:
                return no_update, f"Excel出力に失敗しました: {exc}"

        return no_update, ""

    @app.callback(
        Output("selected-ids-store", "data"),
        Input("scatter-graph", "selectedData"),
        Input("hist-graph", "selectedData"),
        Input("box-graph", "selectedData"),
        Input("matrix-graph", "selectedData"),
        Input("data-table", "derived_virtual_selected_row_ids"),
        Input("table-select-all-check", "value"),
        Input("clear-selection-button", "n_clicks"),
        Input("current-data-store", "data"),
        State("selected-ids-store", "data"),
        State("hist-graph", "figure"),
        State("box-graph", "figure"),
        State("keyboard-state-store", "data"),
        State("data-table", "derived_virtual_row_ids"),
        State("data-table", "data"),
        State("url-location", "search"),
        prevent_initial_call=True,
    )
    def sync_selected_ids(
        scatter_selected_data: dict[str, Any] | None,
        hist_selected_data: dict[str, Any] | None,
        box_selected_data: dict[str, Any] | None,
        matrix_selected_data: dict[str, Any] | None,
        table_selected_row_ids: list[str] | None,
        select_all_value: list[str] | None,
        _clear_clicks: int | None,
        current_data: dict[str, Any] | None,
        current_selected_ids: list[str] | None,
        hist_figure: dict[str, Any] | None,
        box_figure: dict[str, Any] | None,
        keyboard_state: dict[str, Any] | None,
        visible_row_ids: list[object] | None,
        table_data: list[dict[str, Any]] | None,
        url_search: str | None,
    ) -> list[str] | Any:
        current_ids = normalize_id_list(current_selected_ids)
        trigger = callback_context.triggered_id
        is_graph_window = _query_window_graph(url_search) is not None

        if trigger == "current-data-store":
            return no_update
        elif trigger == "clear-selection-button":
            next_ids = []
        elif trigger == "table-select-all-check":
            # 別ウィンドウ初期表示時に checklist の初期値 [] が流れてきて
            # 共有選択を消してしまうケースを回避する。
            if (
                is_graph_window
                and current_ids
                and (not select_all_value)
                and not visible_row_ids
            ):
                return no_update
            if select_all_value and "all" in select_all_value:
                next_ids = normalize_id_list(visible_row_ids)
            else:
                next_ids = []
        elif trigger == "data-table":
            # 別ウィンドウでは DataTable 初期化の空選択イベントで共有選択が消えやすいため、
            # 空選択は無視する（全解除は「選択をクリア」ボタンで実施）。
            if (
                is_graph_window
                and current_ids
                and (table_selected_row_ids is None or len(table_selected_row_ids) == 0)
            ):
                return no_update
            next_ids = normalize_id_list(table_selected_row_ids)
        elif trigger == "scatter-graph":
            if scatter_selected_data is None:
                return no_update
            new_ids = _extract_ids_from_selection(scatter_selected_data)
            next_ids = normalize_id_list([*current_ids, *new_ids]) if _is_additive_selection(keyboard_state) else normalize_id_list(new_ids)
        elif trigger == "matrix-graph":
            if matrix_selected_data is None:
                return no_update
            new_ids = _extract_ids_from_selection(matrix_selected_data)
            next_ids = normalize_id_list([*current_ids, *new_ids]) if _is_additive_selection(keyboard_state) else normalize_id_list(new_ids)
        elif trigger == "hist-graph":
            if hist_selected_data is None:
                return no_update
            new_ids = _extract_ids_from_distribution_selection(hist_selected_data, hist_figure)
            next_ids = normalize_id_list([*current_ids, *new_ids]) if _is_additive_selection(keyboard_state) else normalize_id_list(new_ids)
        elif trigger == "box-graph":
            if box_selected_data is None:
                return no_update
            new_ids = _extract_ids_from_distribution_selection(box_selected_data, box_figure)
            next_ids = normalize_id_list([*current_ids, *new_ids]) if _is_additive_selection(keyboard_state) else normalize_id_list(new_ids)
        else:
            return no_update

        if next_ids == current_ids:
            return no_update
        return next_ids

    @app.callback(
        Output("scatter-graph", "figure"),
        Output("hist-graph", "figure"),
        Output("box-graph", "figure"),
        Output("matrix-graph", "figure"),
        Output("summary-panel", "children"),
        Output("ranking-table", "data"),
        Output("ranking-table", "columns"),
        Output("data-table", "data"),
        Output("data-table", "columns"),
        Output("data-table", "selected_row_ids"),
        Output("data-table", "selected_rows"),
        Output("dataset-meta", "children"),
        Output("modeling-summary", "children"),
        Output("selection-message", "children"),
        Output("table-section", "style"),
        Output("graph-section", "style"),
        Output("show-graphs-button", "children"),
        Output("open-scatter-window-link", "style"),
        Output("open-hist-window-link", "style"),
        Output("open-box-window-link", "style"),
        Output("open-matrix-window-link", "style"),
        Output("scatter-card", "style"),
        Output("hist-card", "style"),
        Output("box-card", "style"),
        Output("matrix-card", "style"),
        Output("graph-window-message", "children"),
        Input("current-data-store", "data"),
        Input("selected-ids-store", "data"),
        Input("ui-config-store", "data"),
        Input("view-config-store", "data"),
        Input("table-display-check", "value"),
        Input("url-location", "search"),
        Input("app-run-store", "data"),
    )
    def render_views(
        current_data: dict[str, Any] | None,
        selected_ids: list[str] | None,
        ui_config: dict[str, Any] | None,
        view_config: dict[str, Any] | None,
        table_display_check: list[str] | None,
        url_search: str | None,
        app_run_data: dict[str, Any] | None,
    ) -> tuple[Any, ...]:
        config = ui_config or {}
        per_view = view_config or {}
        visible_graphs = set(_normalize_visible_graphs(config.get("visible_graphs")))
        requested_graph = _query_window_graph(url_search)
        base_show_graphs = bool(config.get("show_graphs", False))
        effective_show_graphs = base_show_graphs or (requested_graph is not None)
        graph_style = _graph_section_style(effective_show_graphs)
        toggle_label = "グラフ設定を隠す" if base_show_graphs else "グラフ設定を表示"
        show_table = "show" in set(table_display_check or [])
        table_section_style = {"display": "block"} if show_table else {"display": "none"}

        has_data = bool(current_data and current_data.get("df_json") and _is_current_run_data(current_data, app_run_data))
        if not has_data:
            empty_data: list[dict[str, Any]] = []
            empty_columns: list[dict[str, str]] = []
            no_data_meta = format_dataset_meta(None, {})
            empty_style = _window_link_style(False)
            hidden_card = _graph_card_style(False)
            return (
                empty_figure("データを読み込むと散布図を表示できます。"),
                empty_figure("データを読み込むとヒストグラムを表示できます。"),
                empty_figure("データを読み込むと箱ひげ図を表示できます。"),
                empty_figure("データを読み込むと散布図行列を表示できます。"),
                html.Div("CSV/ExcelアップロードまたはSQL実行でデータを読み込んでください。"),
                [],
                _ranking_columns(),
                empty_data,
                empty_columns,
                [],
                [],
                no_data_meta,
                html.Div(),
                "未選択 (No selection)",
                table_section_style,
                graph_style,
                toggle_label,
                empty_style,
                empty_style,
                empty_style,
                empty_style,
                hidden_card,
                hidden_card,
                hidden_card,
                hidden_card,
                "データを読み込むと、上部リンクから各グラフを別ウィンドウ表示できます。",
            )

        raw_df = prepare_dataframe(dataframe_from_json(current_data["df_json"]))
        source_name = current_data.get("source_name")
        selected_ids = normalize_id_list(selected_ids)

        typed_df = _prepare_typed_dataframe(current_data, config)
        modeled_df, modeling_meta = _apply_modeling_config(typed_df, config)
        all_cols: list[str] = _plot_columns(list(modeled_df.columns))

        exclude_missing_rows = bool(config.get("exclude_missing_rows", False))
        treat_selected_as_missing = bool(config.get("treat_selected_as_missing", False))
        filtered_df = apply_analysis_filters(
            modeled_df,
            exclude_missing_rows=exclude_missing_rows,
            selected_ids=selected_ids,
            treat_selected_as_missing=treat_selected_as_missing,
            id_col="id",
        )
        runtime_metadata = build_runtime_metadata(filtered_df, id_col="id")
        numeric_cols: list[str] = list(runtime_metadata.get("numeric_cols", []))

        existing_id_set = set(filtered_df["id"].astype(str))
        selected_ids = [row_id for row_id in selected_ids if row_id in existing_id_set]

        x_col = pick_column(per_view.get("x_col"), all_cols, 0)
        y_col = pick_column(per_view.get("y_col"), all_cols, 1)
        hist_col = pick_column(per_view.get("hist_col"), all_cols, 0)
        box_col = pick_column(per_view.get("box_col"), numeric_cols, 0)
        matrix_cols = _pick_multi_columns(per_view.get("matrix_cols"), numeric_cols, min(len(numeric_cols), 4))

        scatter_figure = create_scatter_figure(filtered_df, x_col, y_col, selected_ids)
        hist_figure = create_distribution_figure(filtered_df, hist_col, selected_ids, view_mode="hist")
        box_figure = create_distribution_figure(filtered_df, box_col, selected_ids, view_mode="box")
        matrix_figure = create_scatter_matrix_figure(filtered_df, matrix_cols, selected_ids)

        summary = _build_summary_panel(filtered_df, selected_ids, numeric_cols)

        ranking_df, ranking_message = rank_candidate_causes(filtered_df, selected_ids, top_n=10)
        ranking_display = ranking_df.copy()
        for col in ("p_value", "statistic", "effect_size", "mean_diff"):
            if col in ranking_display.columns:
                ranking_display[col] = ranking_display[col].map(
                    lambda value: "" if pd.isna(value) else f"{value:.4g}"
                )
        ranking_data = ranking_display.to_dict("records")

        table_data = _to_records(filtered_df)
        table_columns = [{"name": col, "id": col} for col in filtered_df.columns]
        selected_id_set = set(selected_ids)
        selected_rows = [idx for idx, row_id in enumerate(filtered_df["id"].astype(str)) if row_id in selected_id_set]

        raw_meta = build_runtime_metadata(raw_df, id_col="id")
        filtered_meta = build_runtime_metadata(filtered_df, id_col="id")
        dataset_meta = (
            f"{format_dataset_meta(source_name, raw_meta)} | "
            f"分析対象行={filtered_meta['row_count']}"
        )
        split_method = _split_method_label(str(modeling_meta.get("split_method", "")))
        train_count = int(modeling_meta.get("train_count", 0))
        test_count = int(modeling_meta.get("test_count", 0))
        dataset_meta += f" | 分割={split_method} ({train_count}/{test_count})"
        modeling_summary = _build_modeling_summary(modeling_meta)

        if ranking_message:
            selection_message = ranking_message
        else:
            selection_message = f"選択 {len(selected_ids)} / 非選択 {len(filtered_df) - len(selected_ids)}"

        if exclude_missing_rows:
            selection_message += " | 欠損行除外: ON"
        if treat_selected_as_missing:
            selection_message += " | 選択中を欠損扱い: ON"

        scatter_link_style = _window_link_style("scatter" in visible_graphs)
        hist_link_style = _window_link_style("hist" in visible_graphs)
        box_link_style = _window_link_style("box" in visible_graphs)
        matrix_link_style = _window_link_style("matrix" in visible_graphs)

        if requested_graph is None:
            # メイン画面ではグラフを重ねず、別ウィンドウで表示する運用。
            scatter_visible = False
            hist_visible = False
            box_visible = False
            matrix_visible = False
            if visible_graphs:
                graph_window_message = "グラフは別ウィンドウで表示します。上部リンクから開いてください。"
            else:
                graph_window_message = "表示対象グラフが未選択です。チェックボックスで選択してください。"
        else:
            scatter_visible = requested_graph == "scatter" and ("scatter" in visible_graphs)
            hist_visible = requested_graph == "hist" and ("hist" in visible_graphs)
            box_visible = requested_graph == "box" and ("box" in visible_graphs)
            matrix_visible = requested_graph == "matrix" and ("matrix" in visible_graphs)
            if not (scatter_visible or hist_visible or box_visible or matrix_visible):
                graph_window_message = "このグラフ種別は現在非表示設定です。メイン画面でチェックをONにしてください。"
            else:
                graph_window_message = f"{requested_graph} グラフを別ウィンドウ表示中です。"

        return (
            scatter_figure,
            hist_figure,
            box_figure,
            matrix_figure,
            summary,
            ranking_data,
            _ranking_columns(),
            table_data,
            table_columns,
            selected_ids,
            selected_rows,
            dataset_meta,
            modeling_summary,
            selection_message,
            table_section_style,
            graph_style,
            toggle_label,
            scatter_link_style,
            hist_link_style,
            box_link_style,
            matrix_link_style,
            _graph_card_style(scatter_visible, "420px"),
            _graph_card_style(hist_visible, "420px"),
            _graph_card_style(box_visible, "420px"),
            _graph_card_style(matrix_visible, "420px"),
            graph_window_message,
        )
