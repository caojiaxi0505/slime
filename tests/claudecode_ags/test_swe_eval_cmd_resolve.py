"""Tests for EvalPlan / mode detection."""

from __future__ import annotations

from types import SimpleNamespace

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


def test_detect_swegym_before_official_swebench():
    md = {
        "dataset_type": "swegym",
        "data_source": "swegym",
        "repo": "conan-io/conan",
        "FAIL_TO_PASS": ["test::a"],
        "PASS_TO_PASS": ["test::b"],
    }
    assert detect_eval_mode(md) == EvalMode.SWEGYM


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


def test_swebench_plan_uses_official_eval_script(monkeypatch):
    test_spec = SimpleNamespace(
        eval_script="official repo-specific test command\n",
        instance_id="psf__requests-1",
        repo="psf/requests",
    )
    monkeypatch.setattr(
        "examples.claudecode_ags.swe_eval.official.make_official_test_spec",
        lambda metadata: test_spec,
    )
    plan = resolve_eval_plan(
        {
            "instance_id": "psf__requests-1",
            "repo": "psf/requests",
            "version": "2.0",
            "base_commit": "abc",
            "FAIL_TO_PASS": ["tests/test_x.py::test_a"],
            "PASS_TO_PASS": ["tests/test_x.py::test_b"],
            "test_patch": "diff --git a/x b/x\n",
        }
    )
    assert plan.mode == EvalMode.SWEBENCH
    assert plan.eval_cmd == test_spec.eval_script
    assert plan.test_patch == ""
    assert plan.official_test_spec is test_spec
    assert plan.details["runner"] == "official_swebench"


def test_swegym_plan_uses_explicit_tests_without_official_spec(monkeypatch):
    def fail_if_called(metadata):
        raise AssertionError("SWE-Gym must not use the official TestSpec registry")

    monkeypatch.setattr(
        "examples.claudecode_ags.swe_eval.official.make_official_test_spec",
        fail_if_called,
    )
    plan = resolve_eval_plan(
        {
            "dataset_type": "swegym",
            "data_source": "swegym",
            "repo": "conan-io/conan",
            "version": "1.0",
            "FAIL_TO_PASS": ["test::fails_before_patch"],
            "PASS_TO_PASS": ["test::regression"],
            "test_patch": "diff --git a/test.py b/test.py\n",
        }
    )
    assert plan.mode == EvalMode.SWEGYM
    assert "python -m pytest" in plan.eval_cmd
    assert "test::fails_before_patch" in plan.eval_cmd
    assert "test::regression" in plan.eval_cmd
    assert plan.test_patch.startswith("diff --git")
    assert plan.official_test_spec is None
    assert plan.details["runner"] == "swegym_pytest"


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
