"""Tests for grade_common thresholds and pytest log parsing."""

from __future__ import annotations

import os

from examples.claudecode_ags.swe_eval.grade_common import (
    parse_list,
    resolved_from_report,
    resolved_from_status_map,
    tests_report,
)
from examples.claudecode_ags.swe_eval.log_parse import parse_pytest_log


def test_parse_list_json_and_pythonish():
    assert parse_list('["a::t1", "b::t2"]') == ["a::t1", "b::t2"]
    assert parse_list(["x", "y"]) == ["x", "y"]
    assert parse_list(None) == []


def test_resolved_defaults_f2p_1_p2p_099(monkeypatch):
    monkeypatch.delenv("SLIME_CC_REWARD_F2P_THRESHOLD", raising=False)
    monkeypatch.delenv("SLIME_CC_REWARD_P2P_THRESHOLD", raising=False)
    report = {
        "FAIL_TO_PASS": {"pass_ratio": 1.0},
        "PASS_TO_PASS": {"pass_ratio": 0.99},
    }
    assert resolved_from_report(report) is True
    report["PASS_TO_PASS"]["pass_ratio"] = 0.98
    assert resolved_from_report(report) is False
    report["PASS_TO_PASS"]["pass_ratio"] = 1.0
    report["FAIL_TO_PASS"]["pass_ratio"] = 0.99
    assert resolved_from_report(report) is False


def test_resolved_env_thresholds(monkeypatch):
    monkeypatch.setenv("SLIME_CC_REWARD_F2P_THRESHOLD", "0.5")
    monkeypatch.setenv("SLIME_CC_REWARD_P2P_THRESHOLD", "0.5")
    report = {
        "FAIL_TO_PASS": {"pass_ratio": 0.5},
        "PASS_TO_PASS": {"pass_ratio": 0.5},
    }
    assert resolved_from_report(report) is True


def test_empty_p2p_counts_as_full_ratio():
    status = {"tests/t.py::test_a": "PASSED"}
    resolved, report = resolved_from_status_map(status, ["tests/t.py::test_a"], [])
    assert report["PASS_TO_PASS"]["pass_ratio"] == 1.0
    assert resolved is True


def test_parse_pytest_status_first_and_nodeid_first():
    log = "\n".join(
        [
            "PASSED tests/a.py::test_ok",
            "tests/b.py::test_fail FAILED",
            "FAILED tests/c.py::test_err",
        ]
    )
    sm = parse_pytest_log(log)
    assert sm["tests/a.py::test_ok"] == "PASSED"
    assert sm["tests/b.py::test_fail"] == "FAILED"
    assert sm["tests/c.py::test_err"] == "FAILED"


def test_tests_report_missing_counts_as_not_pass():
    report = tests_report({"a::t": "PASSED"}, ["a::t", "a::missing"], [])
    assert report["FAIL_TO_PASS"]["pass_count"] == 1
    assert report["FAIL_TO_PASS"]["total"] == 2
    assert report["FAIL_TO_PASS"]["pass_ratio"] == 0.5
    assert "a::missing" in report["FAIL_TO_PASS"]["missing"]
