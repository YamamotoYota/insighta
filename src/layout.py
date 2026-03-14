# Copyright (c) 2026 Yota Yamamoto
# SPDX-License-Identifier: MIT

"""Dash layout definition."""

from __future__ import annotations

from typing import Any

from dash import dash_table, dcc, html

from .db_connectors import build_sql_connection_options
from .model_runner import default_model_key, model_description, model_options
from .ui_config import graph_card_style, graph_options, graph_window_links, visible_graph_keys


def create_layout(
    initial_current_data: dict[str, Any],
    initial_ui_config: dict[str, Any],
    initial_view_config: dict[str, Any],
    initial_status_message: str,
    app_run_id: str,
) -> html.Div:
    """Create app layout."""
    return html.Div(
        [
            dcc.Location(id="url-location", refresh=False),
            dcc.Store(id="app-run-store", data={"app_run_id": app_run_id}, storage_type="memory"),
            dcc.Store(id="current-data-store", data=initial_current_data, storage_type="local"),
            dcc.Store(id="selected-ids-store", data=None, storage_type="local"),
            dcc.Store(id="ui-config-store", data=initial_ui_config, storage_type="local"),
            dcc.Store(id="view-config-store", data=initial_view_config, storage_type="memory"),
            dcc.Store(
                id="sql-connection-store",
                data={
                    "connected": False,
                    "dbms": "sqlserver",
                    "server": "",
                    "port": None,
                    "database": "",
                    "schema": "",
                    "sqlite_path": "",
                    "username": "",
                    "password": "",
                    "tables": [],
                },
                storage_type="memory",
            ),
            dcc.Store(id="keyboard-state-store", data={"ctrl": False, "meta": False}, storage_type="memory"),
            dcc.Store(id="model-artifact-store", data={"cache_key": None, "meta": {}}, storage_type="memory"),
            dcc.Interval(id="keyboard-poll-interval", interval=200, n_intervals=0),
            dcc.Download(id="download-model-file"),
            html.H2("INSIGHTA"),
            html.P("データを多様なグラフで描画しながらGUI上で選択してデータの特徴を探索するEDAツール"),
            html.Div(
                [
                    dcc.Upload(
                        id="upload-data",
                        children=html.Button("CSV / Excel をアップロード"),
                        multiple=False,
                    ),
                    html.Button("グラフ設定を表示", id="show-graphs-button", n_clicks=0),
                    html.Button("選択をクリア", id="clear-selection-button", n_clicks=0),
                    html.Button(
                        "アプリを終了",
                        id="shutdown-button",
                        n_clicks=0,
                        style={"backgroundColor": "#b30000", "color": "#fff", "border": "none", "padding": "6px 10px"},
                    ),
                    html.Div(id="upload-status", children=initial_status_message),
                    html.Div(id="shutdown-status", style={"color": "#b30000"}),
                    html.Div(id="dataset-meta"),
                ],
                style={
                    "display": "flex",
                    "gap": "12px",
                    "alignItems": "center",
                    "marginBottom": "12px",
                    "flexWrap": "wrap",
                },
            ),
            html.Div(
                [
                    html.Span("グラフを別ウィンドウで開く: "),
                    *[
                        html.A(
                            item["label"],
                            id=item["id"],
                            href=item["href"],
                            target="_blank",
                            rel="noopener noreferrer",
                            style={"marginRight": "10px"} if item["key"] != "matrix" else {},
                        )
                        for item in graph_window_links()
                    ],
                ],
                style={"marginBottom": "12px"},
            ),
            dcc.Checklist(
                id="table-display-check",
                options=[{"label": "データテーブルを表示", "value": "show"}],
                value=["show"],
                style={"marginBottom": "8px"},
            ),
            html.Details(
                [
                    html.Summary("DB（SQL）から読み込む"),
                    html.Div(
                        [
                            dcc.Dropdown(
                                id="sql-dbms-dropdown",
                                options=build_sql_connection_options(),
                                value="sqlserver",
                                clearable=False,
                                style={"width": "180px"},
                            ),
                            dcc.Input(id="sql-server-input", type="text", placeholder="ホスト/サーバー名", style={"width": "190px"}),
                            dcc.Input(id="sql-port-input", type="number", placeholder="ポート", style={"width": "110px"}),
                            dcc.Input(
                                id="sql-database-input",
                                type="text",
                                placeholder="DB名（Oracleはサービス名）",
                                style={"width": "220px"},
                            ),
                            dcc.Input(id="sql-schema-input", type="text", placeholder="スキーマ（任意）", style={"width": "180px"}),
                            dcc.Input(id="sql-username-input", type="text", placeholder="ユーザー名", style={"width": "160px"}),
                            dcc.Input(id="sql-password-input", type="password", placeholder="パスワード", style={"width": "160px"}),
                            html.Button("接続", id="sql-connect-button", n_clicks=0),
                        ],
                        style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginTop": "10px"},
                    ),
                    dcc.Input(
                        id="sql-sqlite-path-input",
                        type="text",
                        placeholder="SQLite DBファイルパス（SQLite選択時に使用）",
                        style={"width": "100%", "marginTop": "8px"},
                    ),
                    html.Div(id="sql-connect-status", style={"marginTop": "6px"}),
                    html.Div(
                        [
                            dcc.Dropdown(
                                id="sql-table-dropdown",
                                options=[],
                                value=None,
                                placeholder="テーブルを選択",
                                style={"minWidth": "320px", "flex": "1"},
                            ),
                            dcc.Input(
                                id="sql-topn-input",
                                type="number",
                                min=1,
                                step=1,
                                value=1000,
                                style={"width": "120px"},
                            ),
                            html.Button("SELECT文を作成", id="sql-build-query-button", n_clicks=0),
                        ],
                        style={
                            "display": "flex",
                            "gap": "8px",
                            "alignItems": "center",
                            "marginTop": "10px",
                            "flexWrap": "wrap",
                        },
                    ),
                    dcc.Textarea(
                        id="sql-query-text",
                        value="",
                        placeholder="例: SELECT * FROM table_name LIMIT 1000（SQL Serverは TOP、Oracle は FETCH FIRST）",
                        style={"width": "100%", "height": "120px", "marginTop": "8px"},
                    ),
                    html.Button("SQLを実行して読み込み", id="sql-run-query-button", n_clicks=0, style={"marginTop": "8px"}),
                ],
                style={"marginBottom": "12px"},
            ),
            html.Details(
                [
                    html.Summary("PI（AF SDK）から読み込む"),
                    html.Div(
                        [
                            dcc.Dropdown(
                                id="pi-data-source-dropdown",
                                options=[
                                    {"label": "PI DAタグ", "value": "pi_da_tag"},
                                    {"label": "PI AF属性", "value": "af_attribute"},
                                    {"label": "PI AFイベントフレーム", "value": "af_event_frame"},
                                ],
                                value="pi_da_tag",
                                clearable=False,
                                style={"width": "220px"},
                            ),
                        ],
                        style={"display": "flex", "gap": "8px", "alignItems": "center", "flexWrap": "wrap", "marginTop": "10px"},
                    ),
                    html.Div(
                        [
                            dcc.Input(
                                id="pi-server-input",
                                type="text",
                                placeholder="PI Data Archiveサーバー名（空欄なら既定）",
                                style={"width": "320px"},
                            ),
                        ],
                        id="pi-da-server-block",
                        style={"display": "flex", "gap": "8px", "alignItems": "center", "flexWrap": "wrap", "marginTop": "8px"},
                    ),
                    html.Div(
                        [
                            dcc.Input(
                                id="pi-af-server-input",
                                type="text",
                                placeholder="AFサーバー名またはパス（例: AFSRV01, \\AFSRV01）",
                                style={"width": "260px"},
                            ),
                            dcc.Input(
                                id="pi-af-database-input",
                                type="text",
                                placeholder="AFデータベース名またはパス（例: FactoryAF, \\AFSRV01\FactoryAF）",
                                style={"width": "240px"},
                            ),
                        ],
                        id="pi-af-server-db-block",
                        style={"display": "none", "gap": "8px", "alignItems": "center", "flexWrap": "wrap", "marginTop": "8px"},
                    ),
                    html.Details(
                        [
                            html.Summary("入力ガイド（サーバ名/DB名/パス形式）"),
                            html.Ul(
                                [
                                    html.Li([html.B("PI Data Archiveサーバー名"), ": 通常はサーバー名のみ（例: ", html.Code("PISRV01"), "）。スラッシュは不要。"]),
                                    html.Li([html.B("AFサーバー名"), ": サーバー名（例: ", html.Code("AFSRV01"), "）または ", html.Code("\\AFSRV01"), " の形式で入力できます。"]),
                                    html.Li([html.B("AFデータベース名"), ": ", html.Code("FactoryAF"), " のような名前単体、または ", html.Code("\\AFSRV01\FactoryAF"), " のようなパス形式に対応します。"]),
                                    html.Li([html.B("AFエレメント名"), ": 単体名（例: ", html.Code("Unit01"), "）/ 階層パス（例: ", html.Code("\\AreaA\\Unit01"), "）/ 完全パス（例: ", html.Code("\\\\AFSRV01\\FactoryAF\\AreaA\\Unit01"), "）に対応します。"]),
                                    html.Li([html.B("イベントフレームテンプレート名"), ": テンプレート名のみ（例: ", html.Code("BatchEventTemplate"), "）。日本語名でも可。パス指定は不要です。"]),
                                ],
                                style={"margin": "6px 0 0 18px", "color": "#444"},
                            ),
                            html.Div(
                                [
                                    "補足: ",
                                    html.Code("/"),
                                    " 区切りは使わず、階層指定が必要な場合は ",
                                    html.Code("\\"),
                                    "（バックスラッシュ）を使ってください。",
                                ],
                                style={"marginTop": "6px", "color": "#555"},
                            ),
                        ],
                        style={"marginTop": "8px", "padding": "6px 8px", "border": "1px solid #e0e0e0", "borderRadius": "6px"},
                    ),
                    html.Div(
                        [
                            dcc.Dropdown(
                                id="pi-query-type-dropdown",
                                options=[
                                    {"label": "Snapshot（現在値）", "value": "snapshot"},
                                    {"label": "Recorded（記録値）", "value": "recorded"},
                                    {"label": "Interpolated（補間値）", "value": "interpolated"},
                                    {"label": "Summary（集計）", "value": "summary"},
                                ],
                                value="recorded",
                                clearable=False,
                                style={"width": "260px"},
                            ),
                        ],
                        id="pi-query-type-block",
                        style={"display": "flex", "gap": "8px", "alignItems": "center", "flexWrap": "wrap", "marginTop": "8px"},
                    ),
                    html.Div(
                        [
                            dcc.Input(
                                id="pi-max-rows-input",
                                type="number",
                                min=1,
                                max=500000,
                                step=1,
                                value=10000,
                                placeholder="対象ごとの最大行数",
                                style={"width": "180px"},
                            ),
                            html.Div(
                                "最大行数の上限です。10000 は『1対象あたり最大1万行』を意味します（例: タグごと / 属性ごと / イベント検索結果）。",
                                style={"width": "100%", "fontSize": 12, "color": "#555"},
                            ),
                        ],
                        id="pi-max-rows-block",
                        style={"display": "flex", "gap": "8px", "alignItems": "center", "flexWrap": "wrap", "marginTop": "8px"},
                    ),
                    html.Div(
                        [
                            dcc.Input(
                                id="pi-start-time-input",
                                type="text",
                                value="*-1d",
                                placeholder="開始時刻（例: *-1d, 2026-01-01 00:00:00）",
                                style={"width": "420px"},
                            ),
                            dcc.Input(
                                id="pi-end-time-input",
                                type="text",
                                value="*",
                                placeholder="終了時刻（例: *）",
                                style={"width": "320px"},
                            ),
                        ],
                        id="pi-time-range-block",
                        style={"display": "flex", "gap": "8px", "alignItems": "center", "flexWrap": "wrap", "marginTop": "8px"},
                    ),
                    html.Div(
                        [
                            dcc.Input(
                                id="pi-interval-input",
                                type="text",
                                value="1h",
                                placeholder="間隔（Snapshot以外で使用, 例: 10m, 1h）",
                                style={"width": "420px"},
                            ),
                        ],
                        id="pi-interval-block",
                        style={"display": "flex", "gap": "8px", "alignItems": "center", "flexWrap": "wrap", "marginTop": "8px"},
                    ),
                    html.Div(
                        [
                            html.Label("Summary関数", style={"marginRight": "6px"}),
                            dcc.Checklist(
                                id="pi-summary-functions-check",
                                options=[
                                    {"label": "Average", "value": "average"},
                                    {"label": "Min", "value": "min"},
                                    {"label": "Max", "value": "max"},
                                    {"label": "Sum", "value": "sum"},
                                    {"label": "Count", "value": "count"},
                                    {"label": "Std", "value": "std"},
                                ],
                                value=["average", "min", "max"],
                                inline=True,
                            ),
                            html.Div(
                                "Summaryは、指定期間を間隔で区切った各区間の記録値を集計します。Average=区間内の平均値、Min/Max=最小/最大、Sum=合計、Count=件数、Std=標準偏差。",
                                style={"width": "100%", "fontSize": 12, "color": "#555"},
                            ),
                        ],
                        id="pi-summary-settings-block",
                        style={"display": "flex", "alignItems": "center", "gap": "6px", "marginTop": "8px", "flexWrap": "wrap"},
                    ),
                    html.Div(
                        [
                            html.Label("PIタグ一覧（PI DAタグ用・複数指定可）"),
                            dcc.Textarea(
                                id="pi-tags-text",
                                value="",
                                placeholder="記入例（1行1タグ）:\nsinusoid\ncdt158\n\nカンマ区切りでも可:\nTAG_A, TAG_B, TAG_C",
                                style={"width": "100%", "height": "90px", "marginTop": "6px"},
                            ),
                            html.Div(
                                "複数タグは「改行」または「カンマ(,)」区切りで入力できます。入力ミスを減らすため、改行区切りを推奨します。",
                                style={"marginTop": "6px", "fontSize": 12, "color": "#555"},
                            ),
                        ],
                        id="pi-tags-block",
                        style={"marginTop": "6px"},
                    ),
                    html.Div(
                        [
                            html.Label("AFエレメント名（AF属性用）"),
                            dcc.Input(
                                id="pi-af-element-input",
                                type="text",
                                placeholder="例: Unit01 / \\AreaA\\Unit01 / \\\\AFSRV01\\FactoryAF\\AreaA\\Unit01（日本語名可）",
                                style={"width": "100%", "maxWidth": "760px"},
                            ),
                            html.Div(
                                "単体名・階層パス・完全パスに対応します。区切りは / または \\ を使用できます（全角記号も可）。",
                                style={"marginTop": "6px", "fontSize": 12, "color": "#555"},
                            ),
                            html.Label("AF属性名一覧（AF属性用・複数指定可）", style={"marginTop": "8px"}),
                            dcc.Textarea(
                                id="pi-af-attributes-text",
                                value="",
                                placeholder="記入例（1行1属性）:\nTemperature\nPressure\n流量\n\nカンマ区切りでも可:\nTAG_A, TAG_B, 温度",
                                style={"width": "100%", "height": "96px", "marginTop": "6px"},
                            ),
                            html.Div(
                                "複数属性は「改行」「カンマ(,)」「読点(、)」で区切れます。日本語属性名にも対応します。",
                                style={"marginTop": "6px", "fontSize": 12, "color": "#555"},
                            ),
                        ],
                        id="pi-af-attribute-target-block",
                        style={"display": "none", "marginTop": "8px"},
                    ),
                    html.Div(
                        [
                            html.Label("イベントフレームテンプレート名（AFイベントフレーム用）"),
                            dcc.Input(
                                id="pi-ef-template-input",
                                type="text",
                                placeholder="例: BatchEventTemplate（テンプレート名のみ。日本語名可）",
                                style={"width": "100%", "maxWidth": "760px"},
                            ),
                            html.Div(
                                "テンプレート名はパスではなく名前単体で入力してください。",
                                style={"marginTop": "6px", "fontSize": 12, "color": "#555"},
                            ),
                            html.Label("イベント生成分析名一覧（AFイベントフレーム用・複数指定可）", style={"marginTop": "8px"}),
                            dcc.Textarea(
                                id="pi-ef-analyses-text",
                                value="",
                                placeholder="記入例（1行1分析）:\nBatchStartAnalysis\nQualityCheckAnalysis\n\nカンマ区切りでも可:\nAnalysisA, AnalysisB",
                                style={"width": "100%", "height": "90px", "marginTop": "6px"},
                            ),
                            html.Div(
                                "複数分析は「改行」「カンマ(,)」「読点(、)」で区切れます。指定した分析名のいずれかに一致するイベントを取得します。",
                                style={"marginTop": "6px", "fontSize": 12, "color": "#555"},
                            ),
                        ],
                        id="pi-ef-target-block",
                        style={"display": "none", "marginTop": "8px"},
                    ),
                    html.Button("PIデータを読み込み", id="pi-run-query-button", n_clicks=0, style={"marginTop": "8px"}),
                    html.Div(
                        "※ PI AF SDK（AF Client）と pythonnet が必要です。PI DAタグ / PI AF属性は時刻をキーにした列形式（横持ち）で返します。",
                        style={"marginTop": "6px", "color": "#555"},
                    ),
                ],
                style={"marginBottom": "12px"},
            ),
            html.Div(
                [
                    html.H4("データテーブル"),
                    html.Div(
                        [
                            dcc.Checklist(
                                id="table-select-all-check",
                                options=[{"label": "すべての行を一括選択", "value": "all"}],
                                value=[],
                                style={"marginBottom": "6px"},
                            ),
                            html.Div(
                                [
                                    html.Button("CSV出力", id="export-table-csv-button", n_clicks=0),
                                    html.Button("Excel出力", id="export-table-xlsx-button", n_clicks=0),
                                    html.Div(id="export-table-status", style={"color": "#444"}),
                                ],
                                style={"display": "flex", "gap": "8px", "alignItems": "center", "flexWrap": "wrap"},
                            ),
                        ],
                        style={"display": "flex", "justifyContent": "space-between", "gap": "12px", "flexWrap": "wrap"},
                    ),
                    dash_table.DataTable(
                        id="data-table",
                        columns=[],
                        data=[],
                        row_selectable="multi",
                        selected_rows=[],
                        selected_row_ids=[],
                        filter_action="custom",
                        filter_query="",
                        sort_action="custom",
                        sort_mode="multi",
                        sort_by=[],
                        page_action="custom",
                        page_current=0,
                        page_size=200,
                        page_count=0,
                        style_table={
                            "height": "420px",
                            "overflowY": "auto",
                            "overflowX": "auto",
                            "minWidth": "100%",
                        },
                        style_cell={"fontSize": 12, "textAlign": "left", "maxWidth": 220},
                    ),
                    dcc.Download(id="download-table-file"),
                ],
                id="table-section",
                style={"display": "block", "marginBottom": "12px"},
            ),
            html.Details(
                [
                    html.Summary("分析前処理の設定"),
                    dcc.Checklist(
                        id="analysis-options-check",
                        options=[
                            {
                                "label": "欠損値がある行を分析対象から除外",
                                "value": "exclude_missing_rows",
                            },
                            {
                                "label": "選択中データを外れ値として欠損扱いにする",
                                "value": "treat_selected_as_missing",
                            },
                        ],
                        value=[],
                        style={"marginTop": "8px", "marginBottom": "8px"},
                    ),
                    html.P(
                        "列型を変更した際に変換できない値は欠損値になります。",
                        style={"margin": "4px 0 8px 0", "color": "#444"},
                    ),
                    dash_table.DataTable(
                        id="dtype-config-table",
                        columns=[
                            {"name": "列名", "id": "column", "editable": False},
                            {"name": "現在の型", "id": "current_type", "editable": False},
                            {
                                "name": "適用型",
                                "id": "target_type",
                                "presentation": "dropdown",
                                "editable": True,
                            },
                        ],
                        data=[],
                        editable=True,
                        dropdown={
                            "target_type": {
                                "options": [
                                    {"label": "自動", "value": "auto"},
                                    {"label": "小数", "value": "float"},
                                    {"label": "整数", "value": "int"},
                                    {"label": "文字列", "value": "string"},
                                    {"label": "カテゴリ", "value": "category"},
                                    {"label": "日時", "value": "datetime"},
                                ]
                            }
                        },
                        style_table={"maxHeight": "280px", "overflowY": "auto", "overflowX": "auto"},
                        style_cell={"fontSize": 12, "textAlign": "left", "maxWidth": 220},
                    ),
                ],
                style={"marginBottom": "12px"},
            ),
            html.Details(
                [
                    html.Summary("統計モデリング共通設定"),
                    html.Div(
                        [
                            html.Label("タイムラグ列追加 (1行に1設定: 例 `temp: 1,-1`)"),
                            dcc.Textarea(
                                id="lag-config-text",
                                value="",
                                style={"width": "100%", "height": "90px"},
                                placeholder="temp: 1,-1\npressure: 2",
                            ),
                        ],
                        style={"marginTop": "8px"},
                    ),
                    html.Div(
                        [
                            html.Label("特徴量追加 (1行に1式: 例 `temp_diff = temp - pressure`)"),
                            dcc.Textarea(
                                id="feature-config-text",
                                value="",
                                style={"width": "100%", "height": "100px"},
                                placeholder="temp_diff = temp - pressure\nvib_log = log(vibration)\npress_exp = exp(pressure)",
                            ),
                        ],
                        style={"marginTop": "8px"},
                    ),
                    html.Div(
                        [
                            html.Label("単純移動平均 (1行に1設定: 例 `temp: 5` または `temp: window=5, center=true`)"),
                            dcc.Textarea(
                                id="sma-config-text",
                                value="",
                                style={"width": "100%", "height": "90px"},
                                placeholder="temp: 5\npressure: window=7, min_periods=1, center=false",
                            ),
                        ],
                        style={"marginTop": "8px"},
                    ),
                    html.Div(
                        [
                            html.Label("指数移動平均 (1行に1設定: 例 `temp: 10` または `temp: span=10, adjust=false`)"),
                            dcc.Textarea(
                                id="ema-config-text",
                                value="",
                                style={"width": "100%", "height": "90px"},
                                placeholder="temp: 10\npressure: span=12, min_periods=1, adjust=false",
                            ),
                        ],
                        style={"marginTop": "8px"},
                    ),
                    html.Div(
                        [
                            html.Label("季節分解 (STL分解, 1行に1設定: 例 `temp: 24` または `temp: period=24, seasonal=13, robust=true`)"),
                            dcc.Textarea(
                                id="stl-config-text",
                                value="",
                                style={"width": "100%", "height": "100px"},
                                placeholder="temp: 24\nvibration: period=24, seasonal=13, trend=25, robust=true",
                            ),
                        ],
                        style={"marginTop": "8px"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("学習/テスト分割方法"),
                                    dcc.Dropdown(
                                        id="split-method-dropdown",
                                        options=[
                                            {"label": "ランダム", "value": "random"},
                                            {"label": "層別ランダム", "value": "stratified_random"},
                                            {"label": "前後", "value": "sequential"},
                                        ],
                                        value="random",
                                        clearable=False,
                                    ),
                                ],
                                style={"width": "240px"},
                            ),
                            html.Div(
                                [
                                    html.Label("学習データ比率"),
                                    dcc.Input(
                                        id="split-ratio-input",
                                        type="number",
                                        min=0.05,
                                        max=0.95,
                                        step=0.05,
                                        value=0.8,
                                        style={"width": "120px"},
                                    ),
                                ],
                                style={"width": "140px"},
                            ),
                            html.Div(
                                [
                                    html.Label("乱数シード"),
                                    dcc.Input(
                                        id="split-seed-input",
                                        type="number",
                                        step=1,
                                        value=42,
                                        style={"width": "120px"},
                                    ),
                                ],
                                style={"width": "140px"},
                            ),
                        ],
                        style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginTop": "8px"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("層別ランダムの層別列"),
                                    dcc.Dropdown(id="split-stratify-column-dropdown", options=[], value=None),
                                ],
                                style={"minWidth": "280px", "flex": "1"},
                            ),
                            html.Div(
                                [
                                    html.Label("前後分割の順序列"),
                                    dcc.Dropdown(id="split-order-column-dropdown", options=[], value=None),
                                ],
                                style={"minWidth": "280px", "flex": "1"},
                            ),
                        ],
                        style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginTop": "8px"},
                    ),
                    dcc.Checklist(
                        id="standardize-check",
                        options=[{"label": "学習データ基準で標準化 (平均0・標準偏差1)", "value": "on"}],
                        value=[],
                        style={"marginTop": "8px", "marginBottom": "8px"},
                    ),
                    html.Div(id="modeling-summary", style={"marginTop": "8px", "color": "#444"}),
                ],
                style={"marginBottom": "12px"},
            ),
            html.Details(
                [
                    html.Summary("統計モデリング実行"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("モデル手法"),
                                    dcc.Dropdown(
                                        id="model-method-dropdown",
                                        options=model_options(),
                                        value=default_model_key(),
                                        clearable=False,
                                    ),
                                    html.Div(
                                        model_description(default_model_key()),
                                        id="model-method-help",
                                        style={"marginTop": "8px", "fontSize": "0.92rem", "color": "#555"},
                                    ),
                                ],
                                style={"minWidth": "320px", "flex": "1"},
                            ),
                            html.Div(
                                [
                                    html.Label("目的変数"),
                                    dcc.Dropdown(id="model-target-dropdown", options=[], value=None),
                                ],
                                style={"minWidth": "280px", "flex": "1"},
                            ),
                            html.Div(
                                [
                                    html.Label("CV fold数"),
                                    dcc.Input(
                                        id="model-cv-fold-input",
                                        type="number",
                                        min=2,
                                        max=10,
                                        step=1,
                                        value=5,
                                        style={"width": "120px"},
                                    ),
                                ],
                                style={"width": "140px"},
                            ),
                            html.Div(
                                [
                                    html.Label("CV探索方式"),
                                    dcc.Dropdown(
                                        id="model-cv-search-method-dropdown",
                                        options=[
                                            {"label": "Grid Search", "value": "grid"},
                                            {"label": "Randomized Search", "value": "randomized"},
                                        ],
                                        value="grid",
                                        clearable=False,
                                    ),
                                ],
                                style={"minWidth": "220px", "flex": "1"},
                            ),
                        ],
                        style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginTop": "8px"},
                    ),
                    html.Div(
                        [
                            html.Label("説明変数（複数選択）"),
                            dcc.Dropdown(id="model-features-dropdown", options=[], value=[], multi=True),
                        ],
                        style={"marginTop": "8px"},
                    ),
                    html.Div(
                        id="model-target-help",
                        children="教師なしモデルでは目的変数は使用しません。",
                        style={"marginTop": "6px", "color": "#444"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("主成分数決定の累積寄与率閾値 (%)"),
                                    dcc.Input(
                                        id="pca-component-threshold-percent-input",
                                        type="number",
                                        min=50,
                                        max=99.9,
                                        step=0.5,
                                        value=90,
                                        style={"width": "140px"},
                                    ),
                                ],
                                style={"minWidth": "240px", "flex": "1"},
                            ),
                            html.Div(
                                [
                                    html.Label("T2 注意管理限界 (%)"),
                                    dcc.Input(
                                        id="pca-t2-warning-limit-input",
                                        type="number",
                                        min=1,
                                        step=1,
                                        value=95,
                                        style={"width": "120px"},
                                    ),
                                ],
                                style={"minWidth": "220px", "flex": "1"},
                            ),
                            html.Div(
                                [
                                    html.Label("T2 異常管理限界 (%)"),
                                    dcc.Input(
                                        id="pca-t2-alarm-limit-input",
                                        type="number",
                                        min=1,
                                        step=1,
                                        value=99,
                                        style={"width": "120px"},
                                    ),
                                ],
                                style={"minWidth": "220px", "flex": "1"},
                            ),
                            html.Div(
                                [
                                    html.Label("Q 注意管理限界 (%)"),
                                    dcc.Input(
                                        id="pca-q-warning-limit-input",
                                        type="number",
                                        min=1,
                                        step=1,
                                        value=95,
                                        style={"width": "120px"},
                                    ),
                                ],
                                style={"minWidth": "220px", "flex": "1"},
                            ),
                            html.Div(
                                [
                                    html.Label("Q 異常管理限界 (%)"),
                                    dcc.Input(
                                        id="pca-q-alarm-limit-input",
                                        type="number",
                                        min=1,
                                        step=1,
                                        value=99,
                                        style={"width": "120px"},
                                    ),
                                ],
                                style={"minWidth": "220px", "flex": "1"},
                            ),
                            html.Div(
                                "PCA異常予兆検知 (T2/Q) を選んだときのみ使用します。累積寄与率閾値は推奨主成分数の計算に使います。T2/Q の各管理限界は百分率入力です。100未満はその百分位で学習データから閾値を計算し、100以上は 90% 閾値を基準に倍率換算します（例: 180% -> 90% 閾値の 2 倍）。",
                                style={"width": "100%", "marginTop": "4px", "fontSize": 12, "color": "#555"},
                            ),
                        ],
                        id="pca-monitor-params-block",
                        style={"display": "none", "gap": "10px", "flexWrap": "wrap", "marginTop": "8px"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("CV推奨用サンプル比率"),
                                    dcc.Input(
                                        id="model-cv-sample-ratio-input",
                                        type="number",
                                        min=0.05,
                                        max=1.0,
                                        step=0.05,
                                        value=1.0,
                                        style={"width": "120px"},
                                    ),
                                ],
                                style={"width": "180px"},
                            ),
                            html.Div(
                                [
                                    html.Label("CV推奨用上限行数"),
                                    dcc.Input(
                                        id="model-cv-sample-maxrows-input",
                                        type="number",
                                        min=1,
                                        step=1,
                                        value=None,
                                        placeholder="未指定=上限なし",
                                        style={"width": "180px"},
                                    ),
                                ],
                                style={"width": "220px"},
                            ),
                        ],
                        style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginTop": "8px"},
                    ),
                    html.Div(
                        [
                            html.Label("CV探索用ハイパーパラメータ候補 (JSON / 値は配列)"),
                            dcc.Textarea(
                                id="model-candidate-grid-text",
                                value="{}",
                                style={"width": "100%", "height": "130px"},
                                placeholder='例: {"n_estimators": [100, 300, 500], "max_depth": [null, 8, 12]}',
                            ),
                            html.Div(
                                "モデル手法を切り替えると、そのモデルの既定候補に初期化されます。",
                                style={"marginTop": "4px", "color": "#666", "fontSize": 12},
                            ),
                        ],
                        style={"marginTop": "8px"},
                    ),
                    html.Div(id="model-cv-grid-estimate", style={"marginTop": "6px", "color": "#444"}),
                    html.Div(
                        [
                            html.Button(
                                "CVで推奨ハイパーパラメータを計算",
                                id="model-suggest-button",
                                n_clicks=0,
                            ),
                            html.Button(
                                "モデルを学習して評価",
                                id="model-run-button",
                                n_clicks=0,
                            ),
                            html.Button(
                                "学習済みモデルを保存",
                                id="model-save-button",
                                n_clicks=0,
                            ),
                            dcc.Upload(
                                id="model-upload-file",
                                children=html.Button("学習済みモデルを再読込"),
                                multiple=False,
                            ),
                        ],
                        style={"display": "flex", "gap": "8px", "marginTop": "8px", "flexWrap": "wrap"},
                    ),
                    html.Div(
                        [
                            html.Label("最終ハイパーパラメータ (JSON)"),
                            dcc.Textarea(
                                id="model-params-text",
                                value="{}",
                                style={"width": "100%", "height": "110px"},
                                placeholder='例: {"n_estimators": 300, "max_depth": 8}',
                            ),
                        ],
                        style={"marginTop": "8px"},
                    ),
                    html.Div(id="modeling-suggest-status", style={"marginTop": "6px", "color": "#444"}),
                    html.Div(id="modeling-run-status", style={"marginTop": "4px", "color": "#222"}),
                    html.Div(id="modeling-model-io-status", style={"marginTop": "4px", "color": "#444"}),
                    html.Div(id="loaded-model-summary", style={"marginTop": "4px", "color": "#444"}),
                    html.Div(id="modeling-result-panel", style={"marginTop": "10px"}),
                ],
                style={"marginBottom": "12px"},
            ),
            html.Div(
                id="graph-section",
                children=[
                    dcc.Checklist(
                        id="visible-graphs-checklist",
                        options=graph_options(),
                        value=visible_graph_keys(),
                        inline=True,
                        style={"marginBottom": "10px"},
                    ),
                    html.Div(id="graph-window-message", style={"marginBottom": "8px", "color": "#444"}),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H4("散布図", style={"margin": "0 0 6px 0"}),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Label("散布図の横軸"),
                                                    dcc.Dropdown(id="x-dropdown", clearable=False),
                                                ],
                                                style={"width": "50%"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Label("散布図の縦軸"),
                                                    dcc.Dropdown(id="y-dropdown", clearable=False),
                                                ],
                                                style={"width": "50%"},
                                            ),
                                        ],
                                        style={"display": "flex", "gap": "8px", "marginBottom": "8px"},
                                    ),
                                    dcc.Graph(
                                        id="scatter-graph",
                                        style={"width": "100%", "height": "calc(100% - 72px)"},
                                        config={"displaylogo": False, "responsive": True},
                                    ),
                                ],
                                id="scatter-card",
                                style=graph_card_style(height="460px"),
                            ),
                            html.Div(
                                [
                                    html.H4("ヒストグラム", style={"margin": "0 0 6px 0"}),
                                    html.Div(
                                        [
                                            html.Label("ヒストグラムの列"),
                                            dcc.Dropdown(id="hist-column-dropdown", clearable=False),
                                        ],
                                        style={"marginBottom": "8px"},
                                    ),
                                    dcc.Graph(
                                        id="hist-graph",
                                        style={"width": "100%", "height": "calc(100% - 72px)"},
                                        config={"displaylogo": False, "responsive": True},
                                    ),
                                ],
                                id="hist-card",
                                style=graph_card_style(height="460px"),
                            ),
                            html.Div(
                                [
                                    html.H4("箱ひげ図", style={"margin": "0 0 6px 0"}),
                                    html.Div(
                                        [
                                            html.Label("箱ひげ図の列"),
                                            dcc.Dropdown(id="box-column-dropdown", clearable=False),
                                        ],
                                        style={"marginBottom": "8px"},
                                    ),
                                    dcc.Graph(
                                        id="box-graph",
                                        style={"width": "100%", "height": "calc(100% - 72px)"},
                                        config={"displaylogo": False, "responsive": True},
                                    ),
                                ],
                                id="box-card",
                                style=graph_card_style(height="460px"),
                            ),
                            html.Div(
                                [
                                    html.H4("散布図行列", style={"margin": "0 0 6px 0"}),
                                    html.Div(
                                        [
                                            html.Label("散布図行列の列"),
                                            dcc.Dropdown(id="matrix-columns-dropdown", multi=True),
                                        ],
                                        style={"marginBottom": "8px"},
                                    ),
                                    dcc.Graph(
                                        id="matrix-graph",
                                        style={"width": "100%", "height": "calc(100% - 72px)"},
                                        config={"displaylogo": False, "responsive": True},
                                    ),
                                ],
                                id="matrix-card",
                                style=graph_card_style(height="460px"),
                            ),
                        ],
                        id="graphs-grid",
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "repeat(auto-fit, minmax(460px, 1fr))",
                            "gap": "12px",
                        },
                    ),
                    html.Div(
                        [
                            html.Div(id="selection-message", style={"fontWeight": "bold"}),
                            html.Div(id="summary-panel", style={"margin": "8px 0 12px 0"}),
                            html.H4("原因候補ランキング"),
                            dash_table.DataTable(
                                id="ranking-table",
                                columns=[],
                                data=[],
                                sort_action="native",
                                style_table={"overflowX": "auto"},
                                style_cell={"fontSize": 12, "textAlign": "left"},
                                page_size=10,
                            ),
                        ],
                        style={"marginTop": "10px"},
                    ),
                ],
                style={"display": "none", "marginTop": "12px"},
            ),
        ],
        style={"padding": "16px", "fontFamily": "Segoe UI, sans-serif"},
    )

