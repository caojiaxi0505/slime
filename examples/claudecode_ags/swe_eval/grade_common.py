"""F2P/P2P pass ratios and resolved thresholds (SLIME_* only)."""

from __future__ import annotations

import json
import os
import re
from typing import Any

_PASS_VALUES = {"PASSED", "XFAIL"}
_QUOTED_STRING_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')


def parse_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if x is not None and str(x).strip()]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if x is not None and str(x).strip()]
            if isinstance(parsed, str) and parsed:
                return [parsed]
        except Exception:
            recovered = [m.group(1).strip() for m in _QUOTED_STRING_RE.finditer(s) if m.group(1).strip()]
            if recovered:
                return recovered
        return [s]
    return []


def f2p_threshold() -> float:
    try:
        return float(os.environ.get("SLIME_CC_REWARD_F2P_THRESHOLD", "1.0"))
    except (TypeError, ValueError):
        return 1.0


def p2p_threshold() -> float:
    try:
        return float(os.environ.get("SLIME_CC_REWARD_P2P_THRESHOLD", "0.99"))
    except (TypeError, ValueError):
        return 0.99


def _normalize_test_id(name: str) -> str:
    return str(name or "").strip()


def _status_for_test_id(test_id: str, status_map: dict[str, str]) -> str | None:
    tid = _normalize_test_id(test_id)
    if tid in status_map:
        return status_map[tid]
    # suffix / prefix soft match
    for key, status in status_map.items():
        if key.endswith(tid) or tid.endswith(key):
            return status
    return None


def bucket_report(status_map: dict[str, str], test_ids: list[str]) -> dict[str, Any]:
    ids = [_normalize_test_id(x) for x in test_ids if _normalize_test_id(x)]
    if not ids:
        return {"pass_count": 0, "total": 0, "pass_ratio": 1.0, "missing": []}
    passed = 0
    missing: list[str] = []
    for tid in ids:
        st = _status_for_test_id(tid, status_map)
        if st is None:
            missing.append(tid)
            continue
        if st in _PASS_VALUES:
            passed += 1
    total = len(ids)
    return {
        "pass_count": passed,
        "total": total,
        "pass_ratio": (passed / total) if total else 1.0,
        "missing": missing,
    }


def tests_report(
    status_map: dict[str, str],
    fail_to_pass: list[str],
    pass_to_pass: list[str],
) -> dict[str, Any]:
    return {
        "FAIL_TO_PASS": bucket_report(status_map, fail_to_pass),
        "PASS_TO_PASS": bucket_report(status_map, pass_to_pass),
    }


def resolved_from_report(report: dict[str, Any]) -> bool:
    f2p = float((report.get("FAIL_TO_PASS") or {}).get("pass_ratio") or 0.0)
    p2p = float((report.get("PASS_TO_PASS") or {}).get("pass_ratio") or 0.0)
    # Empty P2P list → pass_ratio 1.0 from bucket_report; still require F2P threshold.
    return f2p >= f2p_threshold() and p2p >= p2p_threshold()


def resolved_from_status_map(
    status_map: dict[str, str],
    fail_to_pass: list[str],
    pass_to_pass: list[str],
) -> tuple[bool, dict[str, Any]]:
    report = tests_report(status_map, fail_to_pass, pass_to_pass)
    return resolved_from_report(report), report
