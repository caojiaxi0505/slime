"""Tests for EvalPlan / mode detection."""

from __future__ import annotations

from examples.claudecode_ags.swe_eval.cmd_resolve import EvalMode, detect_eval_mode, resolve_eval_plan


def test_detect_scaleswe_by_pre_commands():
    assert detect_eval_mode({"pre_commands": "git checkout x"}) == EvalMode.SCALESWE


def test_detect_scaleswe_by_f2p_script():
    assert detect_eval_mode({"f2p_script": "def test_x(): pass"}) == EvalMode.SCALESWE


def test_detect_rebench_by_test_cmd():
    md = {"install_config": {"test_cmd": ["pytest -q"]}}
    assert detect_eval_mode(md) == EvalMode.REBENCH


def test_detect_swebench_by_repo_and_f2p():
    md = {"repo": "django/django", "FAIL_TO_PASS": ["t::a"], "PASS_TO_PASS": []}
    assert detect_eval_mode(md) == EvalMode.SWEBENCH


def test_detect_simple_cmd_fallback():
    assert detect_eval_mode({"eval_cmd": "true"}) == EvalMode.SIMPLE_CMD


def test_detect_none():
    assert detect_eval_mode({}) == EvalMode.NONE


def test_scaleswe_plan_includes_pytest_and_f2p_file():
    plan = resolve_eval_plan(
        {
            "pre_commands": "echo hi",
            "f2p_script": "def test_x(): assert True",
            "FAIL_TO_PASS": ["test_fail_to_pass.py::test_x"],
            "PASS_TO_PASS": [],
            "workdir": "/testbed",
        }
    )
    assert plan.mode == EvalMode.SCALESWE
    assert "pytest" in plan.eval_cmd
    assert "test_fail_to_pass.py" in plan.eval_cmd
    assert plan.f2p_script.startswith("def test_x")


def test_rebench_plan_uses_test_cmd():
    plan = resolve_eval_plan(
        {
            "install_config": {"test_cmd": "pytest -q tests/", "log_parser": "parse_log_pytest"},
            "FAIL_TO_PASS": ["a::t"],
            "workdir": "/workspace",
        }
    )
    assert plan.mode == EvalMode.REBENCH
    assert "pytest -q tests/" in plan.eval_cmd
    assert plan.log_parser == "parse_log_pytest"


def test_swebench_plan_pytest_nodeids():
    plan = resolve_eval_plan(
        {
            "repo": "psf/requests",
            "FAIL_TO_PASS": ["tests/test_x.py::test_a"],
            "PASS_TO_PASS": ["tests/test_x.py::test_b"],
            "test_patch": "diff --git a/x b/x\n",
        }
    )
    assert plan.mode == EvalMode.SWEBENCH
    assert "tests/test_x.py::test_a" in plan.eval_cmd
    assert plan.test_patch.startswith("diff")


def test_priority_scaleswe_over_rebench():
    plan = resolve_eval_plan(
        {
            "pre_commands": "echo",
            "install_config": {"test_cmd": "pytest"},
            "repo": "a/b",
            "FAIL_TO_PASS": ["t"],
        }
    )
    assert plan.mode == EvalMode.SCALESWE
