"""Default CC reward: binary resolved → {0,1}; keep F2P/P2P ratios in details."""

from __future__ import annotations

from typing import Any


def compose(*, base_eval: dict[str, Any], sample: Any = None, args: Any = None) -> tuple[float, dict[str, Any]]:
    del sample, args
    resolved = bool(base_eval["resolved"])
    details: dict[str, Any] = {"mode": "binary", "resolved": resolved}
    report = base_eval.get("reward_tests_status") or {}
    f2p = report.get("FAIL_TO_PASS") or {}
    p2p = report.get("PASS_TO_PASS") or {}
    if "pass_ratio" in f2p:
        details["test_f2p_ratio"] = float(f2p["pass_ratio"])
        details["test_f2p_passed"] = int(f2p.get("pass_count") or 0)
        details["test_f2p_total"] = int(f2p.get("total") or 0)
    if "pass_ratio" in p2p:
        details["test_p2p_ratio"] = float(p2p["pass_ratio"])
        details["test_p2p_passed"] = int(p2p.get("pass_count") or 0)
        details["test_p2p_total"] = int(p2p.get("total") or 0)
    return (1.0 if resolved else 0.0), details
