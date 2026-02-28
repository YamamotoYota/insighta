# Copyright (c) 2026 Yota Yamamoto
# SPDX-License-Identifier: MIT

"""Tests for PI AF SDK helper input normalization."""

from __future__ import annotations

import pytest

from src.pi_af_sdk import build_pi_query_config, normalize_summary_functions, parse_name_list, parse_tag_list


def test_parse_name_list_accepts_newline_and_comma() -> None:
    tags = parse_name_list("tag_a\ntag_b,tag_c; tag_b")
    assert tags == ("tag_a", "tag_b", "tag_c")


def test_parse_tag_list_alias() -> None:
    tags = parse_tag_list("sinusoid\ncdt158")
    assert tags == ("sinusoid", "cdt158")


def test_normalize_summary_functions_fallback() -> None:
    values = normalize_summary_functions(["unknown", "max"])
    assert values == ("max",)

    fallback = normalize_summary_functions([])
    assert fallback == ("average", "min", "max")


def test_build_pi_da_query_config_defaults() -> None:
    cfg = build_pi_query_config(
        data_source="pi_da_tag",
        pi_server="PISRV01",
        af_server=None,
        af_database=None,
        query_type="recorded",
        tags_text="sinusoid",
        af_element=None,
        af_attributes_text=None,
        start_time=None,
        end_time=None,
        interval=None,
        summary_functions=None,
        max_rows_per_tag=None,
        ef_template=None,
        ef_analyses_text=None,
    )
    assert cfg.data_source == "pi_da_tag"
    assert cfg.pi_server == "PISRV01"
    assert cfg.query_type == "recorded"
    assert cfg.tags == ("sinusoid",)
    assert cfg.start_time == "*-1d"
    assert cfg.end_time == "*"
    assert cfg.interval == "1h"
    assert cfg.max_rows_per_tag == 10000


def test_build_af_attribute_config() -> None:
    cfg = build_pi_query_config(
        data_source="af_attribute",
        pi_server=None,
        af_server="AFSRV01",
        af_database="FactoryAF",
        query_type="interpolated",
        tags_text=None,
        af_element="Unit01",
        af_attributes_text="Temperature\nPressure",
        start_time="*-8h",
        end_time="*",
        interval="15m",
        summary_functions=["average"],
        max_rows_per_tag=2000,
        ef_template=None,
        ef_analyses_text=None,
    )
    assert cfg.data_source == "af_attribute"
    assert cfg.af_server == "AFSRV01"
    assert cfg.af_database == "FactoryAF"
    assert cfg.af_element == "Unit01"
    assert cfg.af_attributes == ("Temperature", "Pressure")
    assert cfg.query_type == "interpolated"


def test_build_event_frame_config_requires_analyses() -> None:
    with pytest.raises(Exception):
        build_pi_query_config(
            data_source="af_event_frame",
            pi_server=None,
            af_server="AFSRV01",
            af_database="FactoryAF",
            query_type="recorded",
            tags_text=None,
            af_element=None,
            af_attributes_text=None,
            start_time="2026-01-01 00:00:00",
            end_time="2026-01-31 23:59:59",
            interval="1h",
            summary_functions=None,
            max_rows_per_tag=10000,
            ef_template="BatchEvent",
            ef_analyses_text="",
        )


def test_build_event_frame_config() -> None:
    cfg = build_pi_query_config(
        data_source="af_event_frame",
        pi_server=None,
        af_server="AFSRV01",
        af_database="FactoryAF",
        query_type="recorded",
        tags_text=None,
        af_element=None,
        af_attributes_text=None,
        start_time="2026-01-01 00:00:00",
        end_time="2026-01-31 23:59:59",
        interval="1h",
        summary_functions=None,
        max_rows_per_tag=10000,
        ef_template="BatchEvent",
        ef_analyses_text="BatchStartAnalysis\nQualityCheckAnalysis",
    )
    assert cfg.data_source == "af_event_frame"
    assert cfg.ef_template == "BatchEvent"
    assert cfg.ef_analyses == ("BatchStartAnalysis", "QualityCheckAnalysis")
