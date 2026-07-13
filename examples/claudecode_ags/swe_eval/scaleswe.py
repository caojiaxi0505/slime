"""Scale-SWE log grading (pytest)."""

from __future__ import annotations

from typing import Any

from examples.claudecode_ags.swe_eval.grade_common import resolved_from_status_map
from examples.claudecode_ags.swe_eval.log_parse import normalize_log, parse_pytest_log


def grade_logs(
    *,
    fail_to_pass: list[str],
    pass_to_pass: list[str],
    stdout: str,
    stderr: str,
) -> dict[str, Any]:
    log = normalize_log("\n".join([stdout or "", stderr or ""]))
    status_map = parse_pytest_log(log)
    resolved, report = resolved_from_status_map(status_map, fail_to_pass, pass_to_pass)
    return {
        "resolved": resolved,
        "reward_tests_status": report,
        "status_map": status_map,
        "parser": "scaleswe_pytest",
    }
