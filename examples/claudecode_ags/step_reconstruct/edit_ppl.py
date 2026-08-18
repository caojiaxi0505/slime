"""Per-patch-turn edit-PPL for hybrid branch selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


class StepTurnAlignmentError(ValueError):
    """Raised when a PostToolUse step cannot be joined to its emitting turn."""


@dataclass(frozen=True)
class PatchTurnPPL:
    """One repo-mutating tool step and its edit-PPL."""

    step_t: int
    edit_ppl: float
    n_tokens: int


def _clip_neg_logp(lp: float, clip: float) -> float:
    return min(-float(lp), clip)


def align_logprobs_to_steps(
    turn_entries: Sequence[tuple[Sequence[float], Sequence[str]]],
    step_tool_use_ids: Sequence[str],
) -> list[list[float]]:
    """Join PostToolUse steps to turns by ``tool_use_id``.

    Each turn entry is ``(output_log_probs, tool_use_ids_emitted)``. Extra
    emitted tools with no snapshot are ignored (hooks may drop/timeout). Every
    snapshot id must resolve to exactly one turn's logprobs — otherwise raise
    :class:`StepTurnAlignmentError`.
    """
    id_to_lps: dict[str, list[float]] = {}
    dup_ids: list[str] = []
    for logprobs, tool_ids in turn_entries:
        lps = list(logprobs or [])
        for tid in tool_ids or []:
            tid = str(tid or "").strip()
            if not tid:
                continue
            if tid in id_to_lps:
                dup_ids.append(tid)
                continue
            id_to_lps[tid] = lps
    if dup_ids:
        raise StepTurnAlignmentError(
            f"step_turn_mismatch: duplicate_tool_use_ids={dup_ids[:5]} n={len(dup_ids)}"
        )

    aligned: list[list[float]] = []
    missing: list[str] = []
    for i, tid in enumerate(step_tool_use_ids):
        tid = str(tid or "").strip()
        if not tid:
            missing.append(f"step[{i}]<empty>")
            continue
        lps = id_to_lps.get(tid)
        if lps is None:
            missing.append(tid)
            continue
        aligned.append(lps)
    if missing:
        raise StepTurnAlignmentError(
            f"step_turn_mismatch: missing_turn_for_steps={len(missing)} "
            f"example={missing[:3]} emitted_ids={len(id_to_lps)} "
            f"num_steps={len(step_tool_use_ids)}"
        )
    if len(aligned) != len(step_tool_use_ids):
        raise StepTurnAlignmentError(
            f"step_turn_mismatch: aligned={len(aligned)} num_steps={len(step_tool_use_ids)}"
        )
    return aligned


def iter_turn_ppls(
    step_diffs: Sequence[str],
    turn_logprobs: Sequence[Sequence[float]],
    *,
    clip: float = 20.0,
    patch_only: bool = True,
) -> list[PatchTurnPPL]:
    """Return per-step PPL, restricted to patch turns when ``patch_only``.

    Aligns ``turn_logprobs[i]`` with ``step_diffs[i]`` (i-th tool snapshot).
    A patch turn is an index ``i`` where
    ``normalize_diff(step_diffs[i]) != normalize_diff(step_diffs[i-1])``
    (for ``i==0``, any non-empty diff counts). With ``patch_only=False`` every
    tool turn is returned, which teacher relabeling uses to cover read/search
    turns as well.

    Raises :class:`StepTurnAlignmentError` on length mismatch (callers must
    not silently skip Stage-2).
    """
    from examples.claudecode_ags.step_reconstruct._common import normalize_diff

    if not step_diffs:
        if turn_logprobs:
            raise StepTurnAlignmentError(
                f"step_turn_mismatch: aligned_tool_turns={len(turn_logprobs)} num_steps=0"
            )
        return []
    if len(step_diffs) != len(turn_logprobs):
        raise StepTurnAlignmentError(
            f"step_turn_mismatch: aligned_tool_turns={len(turn_logprobs)} "
            f"num_steps={len(step_diffs)}"
        )

    out: list[PatchTurnPPL] = []
    prev = ""
    for i, diff in enumerate(step_diffs):
        cur = normalize_diff(diff)
        if patch_only and cur == prev:
            continue
        lps = turn_logprobs[i] or []
        if not lps:
            prev = cur
            continue
        neg = [_clip_neg_logp(lp, clip) for lp in lps]
        out.append(PatchTurnPPL(step_t=i, edit_ppl=sum(neg) / len(neg), n_tokens=len(neg)))
        prev = cur
    return out
