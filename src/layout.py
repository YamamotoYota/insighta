"""Dash layout definition."""

from __future__ import annotations

from typing import Any

from dash import dash_table, dcc, html


def _graph_card_style(height: str) -> dict[str, str]:
    """Return resizable style for graph containers."""
    return {
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
                    html.A(
                        "散布図",
                        id="open-scatter-window-link",
                        href="/?show_graphs=1&graph=scatter",
                        target="_blank",
                        rel="noopener noreferrer",
                        style={"marginRight": "10px"},
                    ),
                    html.A(
                        "ヒストグラム",
                        id="open-hist-window-link",
                        href="/?show_graphs=1&graph=hist",
                        target="_blank",
                        rel="noopener noreferrer",
                        style={"marginRight": "10px"},
                    ),
                    html.A(
                        "箱ひげ図",
                        id="open-box-window-link",
                        href="/?show_graphs=1&graph=box",
                        target="_blank",
                        rel="noopener noreferrer",
                        style={"marginRight": "10px"},
                    ),
                    html.A(
                        "散布図行列",
                        id="open-matrix-window-link",
                        href="/?show_graphs=1&graph=matrix",
                        target="_blank",
                        rel="noopener noreferrer",
                    ),
                ],
                style={"marginBottom": "12px"},
            ),
            dcc.Checklist(
                id="table-display-check",
                options=[{"label": "データテーブルを表示", "value": "show"}],
                value=["show"],
                style={"marginBottom": "8px"},
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
                        filter_action="native",
                        sort_action="native",
                        page_action="none",
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
                    html.Summary("DB（SQL）から読み込む"),
                    html.Div(
                        [
                            dcc.Dropdown(
                                id="sql-dbms-dropdown",
                                options=[
                                    {"label": "SQL Server", "value": "sqlserver"},
                                    {"label": "MySQL", "value": "mysql"},
                                    {"label": "SQLite", "value": "sqlite"},
                                    {"label": "Oracle Database", "value": "oracle"},
                                    {"label": "PostgreSQL", "value": "postgresql"},
                                ],
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
                                        options=[
                                            {"label": "教師なし: PCA", "value": "unsup_pca"},
                                            {"label": "教師なし: ICA", "value": "unsup_ica"},
                                            {"label": "回帰: 重回帰", "value": "reg_linear"},
                                            {"label": "回帰: PLS回帰", "value": "reg_pls"},
                                            {"label": "回帰: LightGBM", "value": "reg_lgbm"},
                                            {"label": "回帰: ランダムフォレスト", "value": "reg_rf"},
                                            {"label": "分類: ロジスティック回帰", "value": "cls_logistic"},
                                            {"label": "分類: LightGBM", "value": "cls_lgbm"},
                                            {"label": "分類: 決定木", "value": "cls_tree"},
                                            {"label": "分類: ランダムフォレスト", "value": "cls_rf"},
                                        ],
                                        value="reg_linear",
                                        clearable=False,
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
                        options=[
                            {"label": "散布図", "value": "scatter"},
                            {"label": "ヒストグラム", "value": "hist"},
                            {"label": "箱ひげ図", "value": "box"},
                            {"label": "散布図行列", "value": "matrix"},
                        ],
                        value=["scatter", "hist", "box", "matrix"],
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
                                style=_graph_card_style("460px"),
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
                                style=_graph_card_style("460px"),
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
                                style=_graph_card_style("460px"),
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
                                style=_graph_card_style("460px"),
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
