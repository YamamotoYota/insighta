# Copyright (c) 2026 Yota Yamamoto
# SPDX-License-Identifier: MIT

"""Tests for DB connector utility functions (driver-independent parts)."""

from __future__ import annotations

from src.db_connectors import DatabaseConfig, build_select_sample_query, normalize_dbms


def test_normalize_dbms_fallback() -> None:
    assert normalize_dbms("mysql") == "mysql"
    assert normalize_dbms("PostgreSQL") == "postgresql"
    assert normalize_dbms("unknown") == "sqlserver"


def test_build_select_sample_query_by_dbms() -> None:
    assert "TOP (10)" in build_select_sample_query(DatabaseConfig(dbms="sqlserver"), "dbo.t1", 10)
    assert "LIMIT 10" in build_select_sample_query(DatabaseConfig(dbms="mysql"), "db1.t1", 10)
    assert "LIMIT 10" in build_select_sample_query(DatabaseConfig(dbms="sqlite"), "t1", 10)
    assert "FETCH FIRST 10 ROWS ONLY" in build_select_sample_query(DatabaseConfig(dbms="oracle"), "SCHEMA1.T1", 10)
    assert "LIMIT 10" in build_select_sample_query(DatabaseConfig(dbms="postgresql"), "public.t1", 10)
