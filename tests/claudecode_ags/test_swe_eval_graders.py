"""Offline grade_logs fixtures for swebench / scaleswe / rebench."""

from __future__ import annotations

from examples.claudecode_ags.swe_eval import rebench as rebench_mod
from examples.claudecode_ags.swe_eval import scaleswe as scaleswe_mod
from examples.claudecode_ags.swe_eval import swebench as swebench_mod

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


def test_swebench_grade_without_swebench_package():
    g = swebench_mod.grade_logs(
        repo="unknown/repo",
        fail_to_pass=[_F2P],
        pass_to_pass=[_P2P],
        stdout=_PASS_LOG,
        stderr="",
    )
    assert g["resolved"] is True
    assert "pytest" in g["parser"]


def test_rebench_grade_pytest_fallback():
    g = rebench_mod.grade_logs(
        fail_to_pass=[_F2P],
        pass_to_pass=[_P2P],
        stdout=_PASS_LOG,
        stderr="",
        log_parser="parse_log_pytest",
    )
    assert g["resolved"] is True
