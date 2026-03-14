# Copyright (c) 2026 Yota Yamamoto
# SPDX-License-Identifier: MIT

"""Runtime configuration helpers for portable local execution."""

from __future__ import annotations

import os

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8050
HOST_ENV_NAME = "INSIGHTA_HOST"
PORT_ENV_NAME = "INSIGHTA_PORT"
PYTHONNET_RUNTIME_ENV_NAME = "PYTHONNET_RUNTIME"
FORCED_PYTHONNET_RUNTIME = "netfx"


def apply_pythonnet_runtime_env() -> str:
    """Force the pythonnet runtime selection for this process."""
    os.environ[PYTHONNET_RUNTIME_ENV_NAME] = FORCED_PYTHONNET_RUNTIME
    return FORCED_PYTHONNET_RUNTIME


def resolve_server_host(value: str | None = None) -> str:
    """Resolve local server host from explicit value or environment."""
    host = str(value or os.environ.get(HOST_ENV_NAME, "")).strip()
    return host or DEFAULT_HOST


def resolve_server_port(value: int | str | None = None) -> int:
    """Resolve local server port from explicit value or environment."""
    raw_value = value
    if raw_value in (None, ""):
        raw_value = os.environ.get(PORT_ENV_NAME, "")
    try:
        port = int(raw_value)
    except (TypeError, ValueError):
        return DEFAULT_PORT
    if 1 <= port <= 65535:
        return port
    return DEFAULT_PORT


def build_local_url(host: str | None = None, port: int | str | None = None) -> str:
    """Build the local browser URL for the current runtime settings."""
    resolved_host = resolve_server_host(host)
    resolved_port = resolve_server_port(port)
    return f"http://{resolved_host}:{resolved_port}"
