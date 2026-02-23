"""Figure creation helpers."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .utils import empty_figure, normalize_id_list

BASE_COLOR = "#9E9E9E"
SELECTED_COLOR = "#111111"


def _existing_columns(candidates: Iterable[str] | None, columns: Iterable[str]) -> list[str]:
    """Return candidate columns that exist in dataframe columns."""
    existing = set(columns)
    return [col for col in (candidates or []) if col in existing]


def create_scatter_figure(
    df: pd.DataFrame,
    x_col: str | None,
    y_col: str | None,
    selected_ids: list[str] | None,
    id_col: str = "id",
) -> go.Figure:
    """Build scatter plot with linked-selection highlight."""
    if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
        return empty_figure("X/Y 列を選択してください。")

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        custom_data=[id_col],
        hover_data={id_col: True},
    )
    fig.update_layout(
        dragmode="select",
        newselection_mode="gradual",
        clickmode="event+select",
        margin={"l": 24, "r": 24, "t": 48, "b": 72},
        legend={"orientation": "h"},
    )

    selected_set = set(normalize_id_list(selected_ids))
    if selected_set:
        selected_points = [
            idx for idx, row_id in enumerate(df[id_col].astype(str)) if row_id in selected_set
        ]
        fig.update_traces(
            selectedpoints=selected_points,
            selected={"marker": {"size": 9, "color": SELECTED_COLOR}},
            unselected={"marker": {"opacity": 0.35}},
            marker={"size": 7, "color": BASE_COLOR},
        )
    else:
        fig.update_traces(marker={"size": 7, "color": BASE_COLOR})

    return fig


def create_distribution_figure(
    df: pd.DataFrame,
    value_col: str | None,
    selected_ids: list[str] | None,
    view_mode: str = "hist",
    id_col: str = "id",
) -> go.Figure:
    """Create a box/hist figure with id-bearing customdata for selection sync."""
    if not value_col or value_col not in df.columns:
        return empty_figure("列を選択してください。")

    working = df[[id_col, value_col]].dropna(subset=[value_col]).copy()
    if working.empty:
        return empty_figure("選択した列に描画可能なデータがありません。")
    working[id_col] = working[id_col].astype(str)

    selected_set = set(normalize_id_list(selected_ids))
    selected_df = working.loc[working[id_col].isin(selected_set)]

    if view_mode == "box":
        if not pd.api.types.is_numeric_dtype(working[value_col]):
            return empty_figure("箱ひげ図は数値列のみ対応しています。")

        fig = go.Figure()
        fig.add_trace(
            go.Box(
                y=working[value_col],
                x=["全体"] * len(working),
                name="全体",
                marker_color=BASE_COLOR,
                line_color=BASE_COLOR,
                fillcolor="rgba(158,158,158,0.30)",
                customdata=working[id_col],
                boxpoints="all",
                jitter=0.25,
                pointpos=0.0,
                marker={"size": 4},
                hovertemplate=f"{value_col}: %{{y}}<br>id: %{{customdata}}<extra>全体</extra>",
            )
        )
        if not selected_df.empty:
            fig.add_trace(
                go.Box(
                    y=selected_df[value_col],
                    x=["全体"] * len(selected_df),
                    name="選択",
                    marker_color=SELECTED_COLOR,
                    line_color=SELECTED_COLOR,
                    fillcolor="rgba(17,17,17,0.35)",
                    customdata=selected_df[id_col],
                    boxpoints="all",
                    jitter=0.25,
                    pointpos=0.0,
                    marker={"size": 4},
                    hovertemplate=f"{value_col}: %{{y}}<br>id: %{{customdata}}<extra>選択</extra>",
                )
            )

        fig.update_layout(
            title=f"箱ひげ図: {value_col}",
            margin={"l": 24, "r": 24, "t": 48, "b": 72},
            boxmode="overlay",
            legend={"orientation": "h"},
            xaxis_title="",
            yaxis_title=value_col,
            dragmode="select",
            newselection_mode="gradual",
            clickmode="event+select",
        )
        return fig

    fig = go.Figure()
    is_numeric = pd.api.types.is_numeric_dtype(working[value_col])
    base_hist_kwargs = {"nbinsx": 40} if is_numeric else {}
    fig.add_trace(
        go.Histogram(
            x=working[value_col],
            name="全体",
            marker_color=BASE_COLOR,
            opacity=0.60,
            customdata=working[id_col],
            hovertemplate=f"{value_col}: %{{x}}<br>件数: %{{y}}<extra>全体</extra>",
            **base_hist_kwargs,
        )
    )
    if not selected_df.empty:
        fig.add_trace(
            go.Histogram(
                x=selected_df[value_col],
                name="選択",
                marker_color=SELECTED_COLOR,
                opacity=0.85,
                customdata=selected_df[id_col],
                hovertemplate=f"{value_col}: %{{x}}<br>件数: %{{y}}<extra>選択</extra>",
                **base_hist_kwargs,
            )
        )

    fig.update_layout(
        title=f"ヒストグラム: {value_col}",
        margin={"l": 24, "r": 24, "t": 48, "b": 72},
        barmode="overlay",
        bargap=0.05,
        legend={"orientation": "h"},
        dragmode="select",
        newselection_mode="gradual",
        clickmode="event+select",
    )
    return fig


def create_scatter_matrix_figure(
    df: pd.DataFrame,
    dimensions: list[str] | None,
    selected_ids: list[str] | None,
    id_col: str = "id",
) -> go.Figure:
    """Build matrix view: lower=scatter, diagonal=hist, upper=correlation."""
    dims = _existing_columns(dimensions, df.columns)
    dims = [col for col in dims if pd.api.types.is_numeric_dtype(df[col])]
    if len(dims) < 2:
        return empty_figure("散布図行列は2列以上の数値列を選択してください。")

    n_dim = len(dims)
    h_space = 0.02
    v_space = 0.02
    fig = make_subplots(
        rows=n_dim,
        cols=n_dim,
        horizontal_spacing=h_space,
        vertical_spacing=v_space,
    )

    id_series = df[id_col].astype(str)
    selected_set = set(normalize_id_list(selected_ids))
    selected_points = [
        idx for idx, row_id in enumerate(id_series) if row_id in selected_set
    ]

    for row_idx, y_col in enumerate(dims, start=1):
        for col_idx, x_col in enumerate(dims, start=1):
            show_x_tick = row_idx == n_dim
            show_y_tick = col_idx == 1

            if row_idx == col_idx:
                fig.add_trace(
                    go.Histogram(
                        x=df[x_col],
                        name="全体",
                        marker_color=BASE_COLOR,
                        opacity=0.65,
                        nbinsx=30,
                        customdata=id_series,
                        hovertemplate=f"{x_col}: %{{x}}<br>件数: %{{y}}<extra></extra>",
                        showlegend=False,
                    ),
                    row=row_idx,
                    col=col_idx,
                )
                if selected_set:
                    selected_mask = id_series.isin(selected_set)
                    if selected_mask.any():
                        fig.add_trace(
                            go.Histogram(
                                x=df.loc[selected_mask, x_col],
                                name="選択",
                                marker_color=SELECTED_COLOR,
                                opacity=0.85,
                                nbinsx=30,
                                customdata=id_series[selected_mask],
                                hovertemplate=f"{x_col}: %{{x}}<br>件数: %{{y}}<extra></extra>",
                                showlegend=False,
                            ),
                            row=row_idx,
                            col=col_idx,
                        )
                fig.update_layout(barmode="overlay", bargap=0.05)

            elif row_idx > col_idx:
                fig.add_trace(
                    go.Scattergl(
                        x=df[x_col],
                        y=df[y_col],
                        mode="markers",
                        marker={"size": 5, "color": BASE_COLOR},
                        customdata=id_series,
                        hovertemplate=(
                            f"{x_col}: %{{x}}<br>{y_col}: %{{y}}<br>"
                            f"{id_col}: %{{customdata}}<extra></extra>"
                        ),
                        selectedpoints=selected_points if selected_points else None,
                        selected={"marker": {"opacity": 1.0, "color": SELECTED_COLOR, "size": 6}},
                        unselected={"marker": {"opacity": 0.25}},
                        showlegend=False,
                    ),
                    row=row_idx,
                    col=col_idx,
                )

            else:
                corr_df = df[[x_col, y_col]].dropna()
                corr = float("nan")
                if len(corr_df) >= 2:
                    corr = float(corr_df[x_col].corr(corr_df[y_col]))

                corr_text = "r = N/A" if pd.isna(corr) else f"r = {corr:.3f}"
                cell_w = (1.0 - h_space * (n_dim - 1)) / n_dim
                cell_h = (1.0 - v_space * (n_dim - 1)) / n_dim
                x_paper = (col_idx - 1) * (cell_w + h_space) + cell_w / 2.0
                y_paper = 1.0 - ((row_idx - 1) * (cell_h + v_space) + cell_h / 2.0)
                fig.add_annotation(
                    x=x_paper,
                    y=y_paper,
                    xref="paper",
                    yref="paper",
                    text=corr_text,
                    showarrow=False,
                    font={"size": 14, "color": "#333"},
                )

            fig.update_xaxes(
                showgrid=False,
                zeroline=False,
                showticklabels=show_x_tick and row_idx >= col_idx,
                title_text=x_col if show_x_tick else "",
                row=row_idx,
                col=col_idx,
            )
            fig.update_yaxes(
                showgrid=False,
                zeroline=False,
                showticklabels=show_y_tick and row_idx >= col_idx,
                title_text=y_col if show_y_tick else "",
                row=row_idx,
                col=col_idx,
            )

    fig.update_layout(
        margin={"l": 24, "r": 24, "t": 48, "b": 72},
        dragmode="select",
        newselection_mode="gradual",
        clickmode="event+select",
    )
    return fig
