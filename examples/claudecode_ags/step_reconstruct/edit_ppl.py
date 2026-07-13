"""Per-patch-turn edit-PPL for hybrid branch selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PatchTurnPPL:
    """One repo-mutating tool step and its edit-PPL."""

    step_t: int
    edit_ppl: float
    n_tokens: int


def _clip_neg_logp(lp: float, clip: float) -> float:
    return min(-float(lp), clip)


def iter_patch_turn_ppls(
    step_diffs: Sequence[str],
    turn_logprobs: Sequence[Sequence[float]],
    *,
    clip: float = 20.0,
) -> list[PatchTurnPPL]:
    """Return edit-PPL for each step whose cumulative workspace diff changed.

    Aligns ``turn_logprobs[i]`` with ``step_diffs[i]`` (i-th assistant turn
    triggers the i-th captured tool snapshot). A patch turn is an index ``i``
    where ``normalize_diff(step_diffs[i]) != normalize_diff(step_diffs[i-1])``
    (for ``i==0``, any non-empty diff counts).

    Returns ``[]`` on turn/step length mismatch (caller should skip branching
    for that trial rather than use a misleading fallback).
    """
    from examples.claudecode_ags.step_reconstruct._common import normalize_diff

    if not step_diffs or not turn_logprobs:
        return []
    if len(step_diffs) != len(turn_logprobs):
        return []

    out: list[PatchTurnPPL] = []
    prev = ""
    for i, diff in enumerate(step_diffs):
        cur = normalize_diff(diff)
        if cur == prev:
            continue
        lps = turn_logprobs[i] or []
        if not lps:
            prev = cur
            continue
        neg = [_clip_neg_logp(lp, clip) for lp in lps]
        out.append(PatchTurnPPL(step_t=i, edit_ppl=sum(neg) / len(neg), n_tokens=len(neg)))
        prev = cur
    return out
