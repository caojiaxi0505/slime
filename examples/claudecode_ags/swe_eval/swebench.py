"""SWE-bench-family log grading."""

from __future__ import annotations

import logging
from typing import Any

from examples.claudecode_ags.swe_eval.grade_common import resolved_from_status_map
from examples.claudecode_ags.swe_eval.log_parse import normalize_log, parse_pytest_log

logger = logging.getLogger(__name__)


def grade_logs(
    *,
    repo: str,
    fail_to_pass: list[str],
    pass_to_pass: list[str],
    stdout: str,
    stderr: str,
) -> dict[str, Any]:
    log = normalize_log("\n".join([stdout or "", stderr or ""]))
    status_map = parse_pytest_log(log)
    parser_name = "pytest_robust"

    if repo and "/" in repo:
        try:
            from swebench.harness.log_parsers import MAP_REPO_TO_PARSER

            parser = MAP_REPO_TO_PARSER.get(repo)
            if parser is not None:
                from types import SimpleNamespace

                test_spec = SimpleNamespace(
                    instance_id="",
                    repo=repo,
                    FAIL_TO_PASS=fail_to_pass,
                    PASS_TO_PASS=pass_to_pass,
                )
                try:
                    parsed = parser(log, test_spec) or {}
                except TypeError:
                    parsed = parser(log) or {}
                if isinstance(parsed, dict) and parsed:
                    for k, v in parsed.items():
                        status_map.setdefault(str(k), str(v))
                    parser_name = f"{getattr(parser, '__name__', 'swebench')}+pytest_robust"
        except Exception as e:
            logger.debug("[swebench grade] MAP_REPO_TO_PARSER unavailable: %s", e)

    resolved, report = resolved_from_status_map(status_map, fail_to_pass, pass_to_pass)
    return {
        "resolved": resolved,
        "reward_tests_status": report,
        "status_map": status_map,
        "parser": parser_name,
    }
