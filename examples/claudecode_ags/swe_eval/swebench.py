"""Strict official SWE-bench log grading."""

from __future__ import annotations

from typing import Any

from examples.claudecode_ags.swe_eval.official import (
    SwebenchHarnessError,
    grade_official_status_map,
    parse_official_eval_log,
    require_swebench_version,
)


def grade_logs(
    *,
    repo: str,
    fail_to_pass: list[str],
    pass_to_pass: list[str],
    stdout: str,
    stderr: str,
    test_spec: Any | None = None,
) -> dict[str, Any]:
    if test_spec is None:
        raise SwebenchHarnessError(f"official TestSpec is missing for repo={repo}")
    if test_spec.repo != repo:
        raise SwebenchHarnessError(
            f"{test_spec.instance_id}: TestSpec repo={test_spec.repo} does not match {repo}"
        )
    status_map, parser_name = parse_official_eval_log(
        test_spec=test_spec,
        stdout=stdout,
        stderr=stderr,
    )
    official_grade = grade_official_status_map(
        test_spec=test_spec,
        status_map=status_map,
    )
    return {
        **official_grade,
        "status_map": status_map,
        "parser": parser_name,
        "swebench_version": require_swebench_version(),
    }
