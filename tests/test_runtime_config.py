# Copyright (c) 2026 Yota Yamamoto
# SPDX-License-Identifier: MIT

"""Tests for portable runtime configuration helpers."""

from __future__ import annotations

from src.runtime_config import (
    FORCED_PYTHONNET_RUNTIME,
    apply_pythonnet_runtime_env,
    build_local_url,
    resolve_server_host,
    resolve_server_port,
)


def test_runtime_config_defaults(monkeypatch) -> None:
    monkeypatch.delenv("INSIGHTA_HOST", raising=False)
    monkeypatch.delenv("INSIGHTA_PORT", raising=False)
    assert resolve_server_host() == "127.0.0.1"
    assert resolve_server_port() == 8050
    assert build_local_url() == "http://127.0.0.1:8050"


def test_runtime_config_uses_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("INSIGHTA_HOST", "0.0.0.0")
    monkeypatch.setenv("INSIGHTA_PORT", "9001")
    assert resolve_server_host() == "0.0.0.0"
    assert resolve_server_port() == 9001
    assert build_local_url() == "http://0.0.0.0:9001"


def test_runtime_config_invalid_port_falls_back_to_default(monkeypatch) -> None:
    monkeypatch.setenv("INSIGHTA_PORT", "99999")
    assert resolve_server_port() == 8050


def test_apply_pythonnet_runtime_env_sets_netfx(monkeypatch) -> None:
    monkeypatch.delenv("PYTHONNET_RUNTIME", raising=False)
    assert apply_pythonnet_runtime_env() == FORCED_PYTHONNET_RUNTIME
