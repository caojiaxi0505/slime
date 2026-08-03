"""Offline grade_logs fixtures for swebench / scaleswe / rebench."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from examples.claudecode_ags.swe_eval import rebench as rebench_mod
from examples.claudecode_ags.swe_eval import scaleswe as scaleswe_mod
from examples.claudecode_ags.swe_eval import swebench as swebench_mod
from examples.claudecode_ags.swe_eval import swegym as swegym_mod

_F2P = "tests/test_bug.py::test_fixed"
_P2P = "tests/test_reg.py::test_ok"

_PASS_LOG = f"""
PASSED {_F2P}
PASSED {_P2P}
"""

_FAIL_F2P_LOG = f"""
FAILED {_F2P}
PASSED {_P2P}
"""


def test_scaleswe_resolved_when_all_pass():
    g = scaleswe_mod.grade_logs(
        fail_to_pass=[_F2P],
        pass_to_pass=[_P2P],
        stdout=_PASS_LOG,
        stderr="",
    )
    assert g["resolved"] is True
    assert g["parser"] == "scaleswe_pytest"


def test_scaleswe_unresolved_when_f2p_fails():
    g = scaleswe_mod.grade_logs(
        fail_to_pass=[_F2P],
        pass_to_pass=[_P2P],
        stdout=_FAIL_F2P_LOG,
        stderr="",
    )
    assert g["resolved"] is False


def test_swebench_grade_requires_official_test_spec():
    with pytest.raises(swebench_mod.SwebenchHarnessError, match="TestSpec is missing"):
        swebench_mod.grade_logs(
            repo="unknown/repo",
            fail_to_pass=[_F2P],
            pass_to_pass=[_P2P],
            stdout=_PASS_LOG,
            stderr="",
        )


def test_swegym_grade_does_not_require_official_test_spec():
    grade = swegym_mod.grade_logs(
        repo="conan-io/conan",
        fail_to_pass=[_F2P],
        pass_to_pass=[_P2P],
        stdout=_PASS_LOG,
        stderr="",
    )
    assert grade["resolved"] is True
    assert grade["reward_tests_status"]["FAIL_TO_PASS"]["pass_ratio"] == 1.0
    assert grade["reward_tests_status"]["PASS_TO_PASS"]["pass_ratio"] == 1.0


def test_swebench_grade_uses_only_official_parser(monkeypatch):
    test_spec = SimpleNamespace(
        instance_id="x__x-1",
        repo="x/x",
        FAIL_TO_PASS=[_F2P],
        PASS_TO_PASS=[_P2P],
    )
    monkeypatch.setattr(
        swebench_mod,
        "parse_official_eval_log",
        lambda **kwargs: ({_F2P: "PASSED", _P2P: "PASSED"}, "parse_log_x"),
    )
    monkeypatch.setattr(
        swebench_mod,
        "grade_official_status_map",
        lambda **kwargs: {
            "resolved": True,
            "resolution_status": "RESOLVED_FULL",
            "eval_type": "PASS_AND_FAIL",
            "official_tests_status": {},
            "reward_tests_status": {},
        },
    )
    monkeypatch.setattr(swebench_mod, "require_swebench_version", lambda: "4.1.0")
    g = swebench_mod.grade_logs(
        repo="x/x",
        fail_to_pass=[_F2P],
        pass_to_pass=[_P2P],
        stdout=_PASS_LOG,
        stderr="",
        test_spec=test_spec,
    )
    assert g["resolved"] is True
    assert g["parser"] == "parse_log_x"
    assert g["swebench_version"] == "4.1.0"
    assert g["resolution_status"] == "RESOLVED_FULL"


def test_swebench_grade_delegates_resolution_to_official_evaluator(monkeypatch):
    test_spec = SimpleNamespace(
        instance_id="x__x-1",
        repo="x/x",
        FAIL_TO_PASS=[_F2P],
        PASS_TO_PASS=[_P2P],
    )
    monkeypatch.setattr(
        swebench_mod,
        "parse_official_eval_log",
        lambda **kwargs: ({_F2P: "PASSED", _P2P: "FAILED"}, "parse_log_x"),
    )
    seen = {}

    def official_grade(**kwargs):
        seen.update(kwargs)
        return {
            "resolved": False,
            "resolution_status": "RESOLVED_NO",
            "eval_type": "PASS_AND_FAIL",
            "official_tests_status": {},
            "reward_tests_status": {},
        }

    monkeypatch.setattr(swebench_mod, "grade_official_status_map", official_grade)
    monkeypatch.setattr(swebench_mod, "require_swebench_version", lambda: "4.1.0")
    g = swebench_mod.grade_logs(
        repo="x/x",
        fail_to_pass=[_F2P],
        pass_to_pass=[_P2P],
        stdout=_PASS_LOG,
        stderr="",
        test_spec=test_spec,
    )
    assert g["resolved"] is False
    assert seen["test_spec"] is test_spec
    assert seen["status_map"] == {_F2P: "PASSED", _P2P: "FAILED"}


def test_rebench_grade_pytest_fallback():
    g = rebench_mod.grade_logs(
        fail_to_pass=[_F2P],
        pass_to_pass=[_P2P],
        stdout=_PASS_LOG,
        stderr="",
        log_parser="parse_log_pytest",
    )
    assert g["resolved"] is True
