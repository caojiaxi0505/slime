"""SWE-rebench log grading."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from examples.claudecode_ags.swe_eval.grade_common import resolved_from_status_map
from examples.claudecode_ags.swe_eval.log_parse import normalize_log, parse_pytest_log

logger = logging.getLogger(__name__)


def _try_external_parser(parser_name: str, log: str) -> dict[str, str]:
    root = (os.environ.get("SLIME_REBENCH_ROOT") or "").strip()
    if root and root not in sys.path:
        sys.path.insert(0, root)
        lib = os.path.join(root, "lib")
        if os.path.isdir(lib) and lib not in sys.path:
            sys.path.insert(0, lib)
    try:
        from agent import log_parsers  # type: ignore

        parser = log_parsers.NAME_TO_PARSER.get(parser_name) or getattr(log_parsers, parser_name, None)
        if parser is None:
            return {}
        out = parser(log) or {}
        return {str(k): str(v) for k, v in out.items()} if isinstance(out, dict) else {}
    except Exception as e:
        logger.debug("[rebench grade] external parser %s unavailable: %s", parser_name, e)
        return {}


def grade_logs(
    *,
    fail_to_pass: list[str],
    pass_to_pass: list[str],
    stdout: str,
    stderr: str,
    log_parser: str = "parse_log_pytest",
) -> dict[str, Any]:
    log = normalize_log("\n".join([stdout or "", stderr or ""]))
    status_map = parse_pytest_log(log)
    parser_name = "pytest_robust"
    if log_parser and log_parser not in {"parse_log_pytest", "pytest", ""}:
        ext = _try_external_parser(log_parser, log)
        if ext:
            for k, v in ext.items():
                status_map.setdefault(k, v)
            parser_name = f"{log_parser}+pytest_robust"
    resolved, report = resolved_from_status_map(status_map, fail_to_pass, pass_to_pass)
    return {
        "resolved": resolved,
        "reward_tests_status": report,
        "status_map": status_map,
        "parser": parser_name,
    }
