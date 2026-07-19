"""Global top-B patch-turn selection for hybrid Stage-2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PatchTurnCandidate:
    source_trial_idx: int
    edit_step_i: int
    branch_step_t: int
    edit_ppl: float


@dataclass(frozen=True)
class SelectedTurn:
    source_trial_idx: int
    edit_step_i: int
    branch_step_t: int
    edit_ppl: float


def select_top_patch_turns(
    candidates: Sequence[PatchTurnCandidate],
    k: int,
) -> list[SelectedTurn]:
    """Pool candidates across failed trials; take top-B by edit-PPL.

    ``B = min(k, len(candidates))``. Ties break by higher ``source_trial_idx``
    then higher target edit index for determinism.
    """
    if k <= 0 or not candidates:
        return []
    ranked = sorted(
        candidates,
        key=lambda c: (c.edit_ppl, c.source_trial_idx, c.edit_step_i),
        reverse=True,
    )
    b = min(int(k), len(ranked))
    return [
        SelectedTurn(
            source_trial_idx=c.source_trial_idx,
            edit_step_i=c.edit_step_i,
            branch_step_t=c.branch_step_t,
            edit_ppl=c.edit_ppl,
        )
        for c in ranked[:b]
    ]
