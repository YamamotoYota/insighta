# Copyright (c) 2026 Yota Yamamoto
# SPDX-License-Identifier: MIT

"""Shared UI constants and style helpers."""

from __future__ import annotations

GRAPH_DEFINITIONS: tuple[tuple[str, str], ...] = (
    ("scatter", "散布図"),
    ("hist", "ヒストグラム"),
    ("box", "箱ひげ図"),
    ("matrix", "散布図行列"),
)

DEFAULT_GRAPH_CARD_HEIGHT = "420px"


def visible_graph_keys() -> list[str]:
    """Return normalized graph key order used across the app."""
    return [key for key, _label in GRAPH_DEFINITIONS]


def graph_options() -> list[dict[str, str]]:
    """Return Dash dropdown/checklist options for visible graphs."""
    return [{"label": label, "value": key} for key, label in GRAPH_DEFINITIONS]


def graph_window_links() -> list[dict[str, str]]:
    """Return graph window link definitions for the main screen."""
    return [
        {
            "key": key,
            "label": label,
            "id": f"open-{key}-window-link",
            "href": f"/?show_graphs=1&graph={key}",
        }
        for key, label in GRAPH_DEFINITIONS
    ]


def graph_card_style(*, visible: bool = True, height: str = DEFAULT_GRAPH_CARD_HEIGHT) -> dict[str, str]:
    """Return resizable graph card style with optional visibility control."""
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
