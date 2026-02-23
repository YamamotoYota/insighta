"""INSIGHTA Dash application entry point."""

from __future__ import annotations

import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path
from uuid import uuid4

from dash import Dash

from src.callbacks import register_callbacks
from src.layout import create_layout
from src.state import build_default_ui_config, build_default_view_config, build_empty_data_state


def _build_initial_state(app_run_id: str) -> tuple[dict, dict, dict, str]:
    """Build initial app stores and status message."""
    current_data = build_empty_data_state(app_run_id=app_run_id)
    ui_config = build_default_ui_config(current_data["metadata"])
    view_config = build_default_view_config(current_data["metadata"])
    status_message = "CSV/Excelをアップロード、またはSQL Serverから読み込んでください。"
    return current_data, ui_config, view_config, status_message


def _resolve_assets_folder() -> str:
    """Resolve Dash assets folder for source run / PyInstaller bundle."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base_dir = Path(getattr(sys, "_MEIPASS"))
    else:
        base_dir = Path(__file__).resolve().parent
    return str(base_dir / "assets")


def _open_browser_when_ready(host: str, port: int, timeout_seconds: float = 30.0) -> None:
    """Open the default browser after the local Dash server starts listening."""
    url = f"http://{host}:{port}"
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                webbrowser.open_new(url)
                return
        except OSError:
            time.sleep(0.5)

    # Fallback: try opening anyway so users can see the expected URL.
    try:
        webbrowser.open_new(url)
    except Exception:
        return


def _start_browser_launcher_if_frozen(host: str, port: int) -> None:
    """Start background browser launcher only for PyInstaller-frozen app."""
    if not bool(getattr(sys, "frozen", False)):
        return
    threading.Thread(
        target=_open_browser_when_ready,
        args=(host, port),
        daemon=True,
    ).start()


def create_app() -> Dash:
    """Create and configure Dash app."""
    app_run_id = str(uuid4())
    current_data, ui_config, view_config, status_message = _build_initial_state(app_run_id=app_run_id)
    app = Dash(__name__, title="INSIGHTA", assets_folder=_resolve_assets_folder())
    app.layout = create_layout(
        initial_current_data=current_data,
        initial_ui_config=ui_config,
        initial_view_config=view_config,
        initial_status_message=status_message,
        app_run_id=app_run_id,
    )
    register_callbacks(app)
    return app


app = create_app()


if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8050
    debug_mode = not bool(getattr(sys, "frozen", False))
    _start_browser_launcher_if_frozen(host=host, port=port)
    app.run(host=host, port=port, debug=debug_mode)
