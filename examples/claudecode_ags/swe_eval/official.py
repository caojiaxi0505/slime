"""Strict integration with the pinned official SWE-bench harness."""

from __future__ import annotations

import os
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version as package_version
from typing import Any

from examples.claudecode_ags.swe_eval.grade_common import parse_list

DEFAULT_SWEBENCH_VERSION = "4.1.0"
_SPHINX_REPO = "sphinx-doc/sphinx"
_SPHINX_TEST_CMD_BEFORE_PR_582 = "tox --current-env -epy39 -v --"
_SPHINX_TEST_CMD_AFTER_PR_582 = "tox --current-env -epy39 -v -- -rA"


class SwebenchHarnessError(RuntimeError):
    """The official harness is missing, incompatible, or cannot grade a log."""


def _apply_sphinx_pr_582(repo: str, specs: dict[str, Any]) -> dict[str, Any]:
    """Apply upstream PR #582 without mutating SWE-bench's global specs."""
    if repo != _SPHINX_REPO:
        return specs

    test_cmd = str(specs.get("test_cmd") or "").strip()
    if test_cmd == _SPHINX_TEST_CMD_AFTER_PR_582:
        return specs
    if test_cmd != _SPHINX_TEST_CMD_BEFORE_PR_582:
        raise SwebenchHarnessError(
            "cannot apply SWE-bench PR #582: "
            f"unexpected Sphinx test_cmd={test_cmd!r}"
        )

    patched = dict(specs)
    patched["test_cmd"] = _SPHINX_TEST_CMD_AFTER_PR_582
    return patched


def _apply_sphinx_pr_582_to_eval_script_list(
    repo: str,
    eval_script_list: list[str],
) -> list[str]:
    """Patch the generated Sphinx command without mutating harness globals.

    SWE-bench 4.1.0's Python script builder reads ``test_cmd`` from the
    module-global ``MAP_REPO_VERSION_TO_SPECS`` instead of from the ``specs``
    argument passed to it.  Consequently, patching a copied specs dict alone
    does not affect the generated command.  Patch the one generated test
    command after script construction instead.
    """
    if repo != _SPHINX_REPO:
        return eval_script_list

    patched = list(eval_script_list)
    fixed_indexes = [
        index
        for index, command in enumerate(patched)
        if command == _SPHINX_TEST_CMD_AFTER_PR_582
        or command.startswith(f"{_SPHINX_TEST_CMD_AFTER_PR_582} ")
    ]
    if len(fixed_indexes) == 1:
        return patched
    if fixed_indexes:
        raise SwebenchHarnessError(
            "cannot apply SWE-bench PR #582: "
            f"found {len(fixed_indexes)} already-patched Sphinx test commands"
        )

    old_indexes = [
        index
        for index, command in enumerate(patched)
        if command == _SPHINX_TEST_CMD_BEFORE_PR_582
        or command.startswith(f"{_SPHINX_TEST_CMD_BEFORE_PR_582} ")
    ]
    if len(old_indexes) != 1:
        raise SwebenchHarnessError(
            "cannot apply SWE-bench PR #582: "
            f"found {len(old_indexes)} unpatched Sphinx test commands"
        )

    index = old_indexes[0]
    command = patched[index]
    patched[index] = (
        _SPHINX_TEST_CMD_AFTER_PR_582
        + command[len(_SPHINX_TEST_CMD_BEFORE_PR_582) :]
    )
    return patched


def _require_sphinx_pr_582_eval_script(test_spec) -> None:
    if test_spec.repo != _SPHINX_REPO:
        return
    if _SPHINX_TEST_CMD_AFTER_PR_582 not in test_spec.eval_script:
        raise SwebenchHarnessError(
            f"{test_spec.instance_id}: SWE-bench PR #582 is missing from the Sphinx eval script"
        )


def expected_version() -> str:
    return (
        os.environ.get("SLIME_SWEBENCH_VERSION")
        or DEFAULT_SWEBENCH_VERSION
    ).strip()


@lru_cache(maxsize=None)
def require_swebench_version(required: str | None = None) -> str:
    wanted = str(required or expected_version()).strip()
    try:
        installed = package_version("swebench")
    except PackageNotFoundError as exc:
        raise SwebenchHarnessError(
            f"official swebench package is missing; required version={wanted}"
        ) from exc
    if installed != wanted:
        raise SwebenchHarnessError(
            f"official swebench version mismatch: installed={installed}, required={wanted}"
        )
    return installed


def _required_text(metadata: dict[str, Any], key: str) -> str:
    value = str(metadata.get(key) or "").strip()
    if not value:
        instance_id = str(metadata.get("instance_id") or "<unknown>")
        raise SwebenchHarnessError(f"{instance_id}: missing required field {key}")
    return value


def _required_blob(metadata: dict[str, Any], key: str) -> str:
    value = str(metadata.get(key) or "")
    if not value.strip():
        instance_id = str(metadata.get("instance_id") or "<unknown>")
        raise SwebenchHarnessError(f"{instance_id}: missing required field {key}")
    return value


def official_instance(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return the exact fields consumed by ``make_test_spec``."""
    return {
        "instance_id": _required_text(metadata, "instance_id"),
        "repo": _required_text(metadata, "repo"),
        "version": _required_text(metadata, "version"),
        "base_commit": _required_text(metadata, "base_commit"),
        "problem_statement": str(metadata.get("problem_statement") or ""),
        # Patch whitespace is data. In particular, removing its final newline
        # can make the official unidiff parser reject a valid last hunk.
        "test_patch": _required_blob(metadata, "test_patch"),
        "FAIL_TO_PASS": parse_list(metadata.get("FAIL_TO_PASS")),
        "PASS_TO_PASS": parse_list(metadata.get("PASS_TO_PASS")),
    }


def make_official_test_spec(metadata: dict[str, Any]):
    require_swebench_version()
    instance = official_instance(metadata)
    try:
        from swebench.harness.constants import (
            MAP_REPO_TO_EXT,
            MAP_REPO_VERSION_TO_SPECS,
        )
        from swebench.harness.test_spec.create_scripts import make_eval_script_list
        from swebench.harness.test_spec.test_spec import TestSpec

        repo = instance["repo"]
        spec_version = instance["version"]
        specs = MAP_REPO_VERSION_TO_SPECS[repo][spec_version]
        specs = _apply_sphinx_pr_582(repo, specs)
        # AGS starts from the prebuilt official instance image. We need the
        # official eval script and parser TestSpec, but must not regenerate the
        # image's repo/environment setup scripts in the main Pod.
        eval_script_list = make_eval_script_list(
            instance,
            specs,
            "testbed",
            "/testbed",
            instance["base_commit"],
            instance["test_patch"],
        )
        eval_script_list = _apply_sphinx_pr_582_to_eval_script_list(
            repo,
            eval_script_list,
        )
        test_spec = TestSpec(
            instance_id=instance["instance_id"],
            repo=repo,
            version=spec_version,
            repo_script_list=[],
            eval_script_list=eval_script_list,
            env_script_list=[],
            arch="x86_64",
            FAIL_TO_PASS=instance["FAIL_TO_PASS"],
            PASS_TO_PASS=instance["PASS_TO_PASS"],
            language=MAP_REPO_TO_EXT[repo],
            docker_specs=specs.get("docker_specs", {}),
            namespace=None,
        )
        _require_sphinx_pr_582_eval_script(test_spec)
        return test_spec
    except Exception as exc:
        raise SwebenchHarnessError(
            f"{instance['instance_id']}: failed to build official SWE-bench TestSpec: {exc}"
        ) from exc


def require_official_parser(test_spec):
    require_swebench_version()
    try:
        from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
    except Exception as exc:
        raise SwebenchHarnessError(
            f"{test_spec.instance_id}: official SWE-bench parser registry is unavailable"
        ) from exc

    parser = MAP_REPO_TO_PARSER.get(test_spec.repo)
    if parser is None:
        raise SwebenchHarnessError(
            f"{test_spec.instance_id}: official parser is missing for repo={test_spec.repo}"
        )
    return parser


def parse_official_eval_log(*, test_spec, stdout: str, stderr: str) -> tuple[dict[str, str], str]:
    """Parse only the official test-output section and never silently fall back."""
    parser = require_official_parser(test_spec)
    log = "\n".join([stdout or "", stderr or ""])
    try:
        from swebench.harness.constants import END_TEST_OUTPUT, START_TEST_OUTPUT
    except Exception as exc:
        raise SwebenchHarnessError(
            f"{test_spec.instance_id}: official SWE-bench output markers are unavailable"
        ) from exc

    if START_TEST_OUTPUT not in log or END_TEST_OUTPUT not in log:
        raise SwebenchHarnessError(
            f"{test_spec.instance_id}: official test-output markers are missing"
        )
    test_log = log.split(START_TEST_OUTPUT, 1)[1].split(END_TEST_OUTPUT, 1)[0]
    try:
        parsed = parser(test_log, test_spec)
        if not parsed:
            parsed = parser(log, test_spec)
    except Exception as exc:
        raise SwebenchHarnessError(
            f"{test_spec.instance_id}: official parser {parser.__name__} failed: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise SwebenchHarnessError(
            f"{test_spec.instance_id}: official parser {parser.__name__} returned "
            f"{type(parsed).__name__}, expected dict"
        )
    return {str(key): str(value) for key, value in parsed.items()}, parser.__name__


def grade_official_status_map(*, test_spec, status_map: dict[str, str]) -> dict[str, Any]:
    """Grade a parsed status map with SWE-bench's own evaluator functions."""
    require_swebench_version()
    try:
        from swebench.harness.constants import (
            FAIL_ONLY_REPOS,
            FAIL_TO_PASS,
            KEY_INSTANCE_ID,
            PASS_TO_PASS,
            EvalType,
            ResolvedStatus,
        )
        from swebench.harness.grading import (
            get_eval_tests_report,
            get_resolution_status,
        )

        gold_results = {
            KEY_INSTANCE_ID: test_spec.instance_id,
            FAIL_TO_PASS: list(test_spec.FAIL_TO_PASS),
            PASS_TO_PASS: list(test_spec.PASS_TO_PASS),
        }
        eval_type = (
            EvalType.FAIL_ONLY
            if test_spec.repo in FAIL_ONLY_REPOS
            else EvalType.PASS_AND_FAIL
        )
        official_report = get_eval_tests_report(
            status_map,
            gold_results,
            eval_type=eval_type,
        )
        resolution_status = get_resolution_status(official_report)
        resolved = resolution_status == ResolvedStatus.FULL.value
    except Exception as exc:
        raise SwebenchHarnessError(
            f"{test_spec.instance_id}: official SWE-bench evaluator failed: {exc}"
        ) from exc

    def summarize(bucket: str) -> dict[str, Any]:
        result = official_report.get(bucket) or {}
        success = [str(item) for item in result.get("success") or []]
        failure = [str(item) for item in result.get("failure") or []]
        total = len(success) + len(failure)
        return {
            "pass_count": len(success),
            "total": total,
            "pass_ratio": (len(success) / total) if total else 1.0,
            "missing": [],
        }

    return {
        "resolved": resolved,
        "resolution_status": resolution_status,
        "eval_type": str(getattr(eval_type, "value", eval_type)),
        "official_tests_status": official_report,
        "reward_tests_status": {
            FAIL_TO_PASS: summarize(FAIL_TO_PASS),
            PASS_TO_PASS: summarize(PASS_TO_PASS),
        },
    }
