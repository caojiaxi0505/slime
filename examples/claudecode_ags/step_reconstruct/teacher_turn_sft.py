"""Stage-2 teacher relabeling of student turns into SFT samples.

For selected turns of failed Stage-1 trials, rebuild the workspace at that turn,
resume the native Claude Code session token-exactly, and let the remote teacher
take a bounded number of real turns (``STEP_GRPO_TEACHER_MAX_STEPS``, default 2).
The teacher executes its tool calls, so this needs a sandbox per relabeled turn;
in exchange the SFT target contains a tool call plus the teacher's reaction to
that tool's real result, instead of an unexecuted first action.

Selection is the cost knob, since one relabeled turn is one sandbox:
``STEP_GRPO_TEACHER_TURN_SELECT`` picks the turn pool and
``STEP_GRPO_TEACHER_TURN_MAX_PER_TRIAL`` bounds how many of them per trial.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from examples.claudecode_ags.step_reconstruct.hybrid_generate import (
    collect_patch_candidates,
    collect_stage2_samples,
    hybrid_k,
    stage2_train_context_limit,
)
from examples.claudecode_ags.step_reconstruct.selection import PatchTurnCandidate, SelectedTurn, select_top_patch_turns

from slime.utils.types import Sample

logger = logging.getLogger(__name__)

_SELECT_MODES = ("all", "patch", "patch_ppl")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return int(raw)


def teacher_turn_select() -> str:
    """Which student turns the teacher relabels.

    ``all`` takes every resume-able tool turn (reads and searches included),
    ``patch`` only turns that changed the workspace, ``patch_ppl`` only the
    top-``k`` patch turns by edit-PPL (the Stage-2 GRPO selection rule).
    """
    mode = (os.environ.get("STEP_GRPO_TEACHER_TURN_SELECT") or "all").strip().lower()
    if mode not in _SELECT_MODES:
        raise ValueError(f"STEP_GRPO_TEACHER_TURN_SELECT must be one of {_SELECT_MODES}, got {mode!r}")
    return mode


def _evenly_spaced(items: list[Any], keep: int) -> list[Any]:
    if keep <= 0 or len(items) <= keep:
        return items
    stride = len(items) / keep
    return [items[min(len(items) - 1, int(i * stride))] for i in range(keep)]


def relabel_turns(
    candidates: list[PatchTurnCandidate],
    *,
    max_per_trial: int,
) -> list[SelectedTurn]:
    """Keep at most ``max_per_trial`` evenly spaced turns of each trial.

    Even spacing (rather than the earliest or highest-PPL turns) keeps the
    relabeled turns spread over early exploration and late edits.
    """
    by_trial: dict[int, list[PatchTurnCandidate]] = {}
    for candidate in candidates:
        by_trial.setdefault(candidate.source_trial_idx, []).append(candidate)
    return [
        SelectedTurn(
            source_trial_idx=c.source_trial_idx,
            edit_step_i=c.edit_step_i,
            branch_step_t=c.branch_step_t,
            edit_ppl=c.edit_ppl,
        )
        for trial_idx in sorted(by_trial)
        for c in _evenly_spaced(by_trial[trial_idx], max_per_trial)
    ]


def select_relabel_turns(trials: list[Any]) -> tuple[list[PatchTurnCandidate], list[SelectedTurn]]:
    mode = teacher_turn_select()
    candidates = collect_patch_candidates(trials, patch_only=mode != "all")
    if mode == "patch_ppl":
        return candidates, select_top_patch_turns(candidates, hybrid_k())
    return candidates, relabel_turns(
        candidates,
        max_per_trial=_env_int("STEP_GRPO_TEACHER_TURN_MAX_PER_TRIAL", 0),
    )


async def teacher_turn_samples(
    *,
    args: Any,
    sample: Sample,
    sampling_params: dict[str, Any],
    trials: list[Any],
    group_index: int,
    hybrid_stats: dict[str, Any] | None = None,
    branch_runner: Any = None,
) -> list[Sample]:
    """Relabel the selected turns of every failed Stage-1 trial."""
    stats: dict[str, Any] = hybrid_stats if hybrid_stats is not None else {}
    stats.setdefault("hybrid_stage2_context_limit_tokens", stage2_train_context_limit(args))
    instance_id = str((sample.metadata or {}).get("instance_id") or "")
    candidates, selected = select_relabel_turns(trials)
    stats["hybrid_num_patch_candidates"] = len(candidates)
    stats["hybrid_num_selected_edits"] = len(selected)
    stats["hybrid_num_branch_tasks"] = len(selected)
    if not selected:
        logger.warning(
            "[teacher-turn] no relabelable turns instance=%s candidates=%d",
            instance_id,
            len(candidates),
        )
        return []

    if branch_runner is None:
        from examples.claudecode_ags.step_reconstruct.teacher_branch_runner import teacher_branch_runner

        branch_runner = teacher_branch_runner

    by_trial = {trial.trial_idx: trial for trial in trials}
    logger.info(
        "[teacher-turn] instance=%s select=%s candidates=%d relabels=%d",
        instance_id,
        teacher_turn_select(),
        len(candidates),
        len(selected),
    )
    raw = await asyncio.gather(
        *(
            branch_runner(
                args=args,
                sample=sample,
                sampling_params=sampling_params,
                bundle=by_trial[sel.source_trial_idx].bundle,
                source_trial_idx=sel.source_trial_idx,
                edit_step_i=sel.edit_step_i,
                branch_step_t=sel.branch_step_t,
                branch_idx=0,
                group_index=group_index,
                edit_ppl=sel.edit_ppl,
            )
            for sel in selected
        ),
        return_exceptions=True,
    )
    return collect_stage2_samples(raw, hybrid_stats=stats, sample_kind="teacher_sft")
