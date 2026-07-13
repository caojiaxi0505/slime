"""Pure decision helpers for SWE-Gym gold / passk filtering."""

from __future__ import annotations

from typing import Any


def decide_gold_task(runs: list[dict[str, Any]], *, expected_repeats: int) -> dict[str, Any]:
    """Keep only if all expected gold attempts resolved with infra_ok."""
    if any(not r.get("infra_ok", True) for r in runs):
        return {"keep": False, "exclude_reason": "infra"}
    if any(not bool(r.get("resolved")) for r in runs):
        return {"keep": False, "exclude_reason": "gold_unresolved"}
    if len(runs) < expected_repeats:
        return {"keep": False, "exclude_reason": "incomplete"}
    return {"keep": True, "exclude_reason": None}


def summarize_passk_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(runs)
    n_resolved = sum(1 for r in runs if r.get("resolved"))
    n_infra_fail = sum(1 for r in runs if not r.get("infra_ok", True))
    n_nonempty = sum(1 for r in runs if int(r.get("diff_chars") or 0) > 0)
    return {
        "n_runs": n,
        "n_resolved": n_resolved,
        "pass_rate": (n_resolved / n) if n else 0.0,
        "pass_at_k": n_resolved >= 1,
        "nonempty_diff_rate": (n_nonempty / n) if n else 0.0,
        "infra_fail_count": n_infra_fail,
    }


def decide_passk_task(summary: dict[str, Any], *, expected_repeats: int) -> dict[str, Any]:
    """Drop only always-resolved (n_resolved == expected_repeats). Keep 0..N-1."""
    if int(summary.get("n_runs") or 0) < expected_repeats:
        return {"keep": False, "exclude_reason": "incomplete"}
    if int(summary.get("n_resolved") or 0) == expected_repeats:
        return {"keep": False, "exclude_reason": "always_resolved"}
    return {"keep": True, "exclude_reason": None}
