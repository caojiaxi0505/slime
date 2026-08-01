from __future__ import annotations

from types import SimpleNamespace

import pytest

from examples.claudecode_ags.swe_eval.official import (
    SwebenchHarnessError,
    _apply_sphinx_pr_582,
    _apply_sphinx_pr_582_to_eval_script_list,
    _require_sphinx_pr_582_eval_script,
)


def test_sphinx_pr_582_adds_pytest_report_all_without_mutating_upstream_specs():
    original = {"test_cmd": "tox --current-env -epy39 -v --", "python": "3.9"}

    patched = _apply_sphinx_pr_582("sphinx-doc/sphinx", original)

    assert patched == {
        "test_cmd": "tox --current-env -epy39 -v -- -rA",
        "python": "3.9",
    }
    assert original["test_cmd"] == "tox --current-env -epy39 -v --"


def test_sphinx_pr_582_is_idempotent_and_does_not_change_other_repositories():
    already_fixed = {"test_cmd": "tox --current-env -epy39 -v -- -rA"}
    other_repo = {"test_cmd": "pytest -rA"}

    assert _apply_sphinx_pr_582("sphinx-doc/sphinx", already_fixed) is already_fixed
    assert _apply_sphinx_pr_582("django/django", other_repo) is other_repo


def test_sphinx_pr_582_rejects_unexpected_upstream_command():
    with pytest.raises(SwebenchHarnessError, match="unexpected Sphinx test_cmd"):
        _apply_sphinx_pr_582(
            "sphinx-doc/sphinx",
            {"test_cmd": "tox --current-env -epy39 -v -- --some-other-flag"},
        )


def test_sphinx_pr_582_patches_generated_command_with_test_directives():
    original = [
        "source /opt/miniconda3/bin/activate",
        ": '>>>>> Start Test Output'",
        "tox --current-env -epy39 -v -- tests/test_x.py",
        ": '>>>>> End Test Output'",
    ]

    patched = _apply_sphinx_pr_582_to_eval_script_list(
        "sphinx-doc/sphinx",
        original,
    )

    assert patched == [
        "source /opt/miniconda3/bin/activate",
        ": '>>>>> Start Test Output'",
        "tox --current-env -epy39 -v -- -rA tests/test_x.py",
        ": '>>>>> End Test Output'",
    ]
    assert original[2] == "tox --current-env -epy39 -v -- tests/test_x.py"


def test_sphinx_pr_582_generated_command_patch_is_scoped_and_strict():
    other = ["tox --current-env -epy39 -v -- tests/test_x.py"]
    assert (
        _apply_sphinx_pr_582_to_eval_script_list("django/django", other)
        is other
    )

    already_fixed = ["tox --current-env -epy39 -v -- -rA tests/test_x.py"]
    assert _apply_sphinx_pr_582_to_eval_script_list(
        "sphinx-doc/sphinx",
        already_fixed,
    ) == already_fixed

    with pytest.raises(SwebenchHarnessError, match="found 0 unpatched"):
        _apply_sphinx_pr_582_to_eval_script_list(
            "sphinx-doc/sphinx",
            ["pytest tests/test_x.py"],
        )


def test_sphinx_eval_script_must_contain_pr_582_command():
    fixed = SimpleNamespace(
        instance_id="sphinx-doc__sphinx-1",
        repo="sphinx-doc/sphinx",
        eval_script="tox --current-env -epy39 -v -- -rA tests/test_x.py",
    )
    _require_sphinx_pr_582_eval_script(fixed)

    missing = SimpleNamespace(
        instance_id="sphinx-doc__sphinx-2",
        repo="sphinx-doc/sphinx",
        eval_script="tox --current-env -epy39 -v -- tests/test_x.py",
    )
    with pytest.raises(SwebenchHarnessError, match="PR #582 is missing"):
        _require_sphinx_pr_582_eval_script(missing)
