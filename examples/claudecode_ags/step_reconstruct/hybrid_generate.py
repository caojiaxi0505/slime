"""Hybrid generate: K vanilla GRPO trials + conditional step-level branching.

Path A entry::

    --custom-generate-function-path \\
        examples.claudecode_ags.step_reconstruct.hybrid_generate.hybrid_generate
    --n-samples-per-prompt 1
    # STEP_GRPO_HYBRID_K=8 (default)
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Sequence

from examples.claudecode_ags.step_reconstruct.edit_ppl import iter_patch_turn_ppls
from examples.claudecode_ags.step_reconstruct.selection import (
    PatchTurnCandidate,
    SelectedTurn,
    select_top_patch_turns,
)
from examples.claudecode_ags.step_reconstruct.session_capture import SessionBundle
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

VanillaRunner = Callable[..., Awaitable[tuple[SessionBundle, list[Sample], bool, list[list[float]]]]]
BranchRunner = Callable[..., Awaitable[list[Sample]]]


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return int(raw)


def hybrid_k() -> int:
    return max(1, _env_int("STEP_GRPO_HYBRID_K", 8))


def _ppl_clip() -> float:
    raw = os.environ.get("STEP_GRPO_PPL_CLIP")
    if raw is None or str(raw).strip() == "":
        return 20.0
    return float(raw)


@dataclass
class VanillaTrialResult:
    trial_idx: int
    bundle: SessionBundle
    samples: list[Sample]
    is_solved: bool
    turn_logprobs: list[list[float]]


def collect_patch_candidates(trials: Sequence[VanillaTrialResult]) -> list[PatchTurnCandidate]:
    """Pool patch turns from all **failed** trials."""
    clip = _ppl_clip()
    out: list[PatchTurnCandidate] = []
    for tr in trials:
        if tr.is_solved:
            continue
        diffs = [tr.bundle.step_diff(i) for i in range(tr.bundle.num_steps)]
        for p in iter_patch_turn_ppls(diffs, tr.turn_logprobs, clip=clip):
            out.append(
                PatchTurnCandidate(
                    source_trial_idx=tr.trial_idx,
                    step_t=p.step_t,
                    edit_ppl=p.edit_ppl,
                )
            )
    return out


def select_branch_turns(trials: Sequence[VanillaTrialResult], k: int) -> list[SelectedTurn]:
    return select_top_patch_turns(collect_patch_candidates(trials), k)


async def hybrid_generate(
    args,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
    *,
    vanilla_runner: VanillaRunner | None = None,
    branch_runner: BranchRunner | None = None,
):
    """Hybrid orchestration. Inject runners for unit tests (no live AGS)."""
    if evaluation:
        from examples.claudecode_ags.generate import generate as plain_generate

        return await plain_generate(args, sample, sampling_params, evaluation=True)

    k = hybrid_k()
    group_index = int(sample.group_index) if sample.group_index is not None else int(sample.index or 0)
    base_index = int(sample.index) if sample.index is not None else 0

    if vanilla_runner is None or branch_runner is None:
        from examples.claudecode_ags.step_reconstruct.live_runners import (
            live_branch_runner,
            live_vanilla_runner,
        )

        if vanilla_runner is None:
            vanilla_runner = live_vanilla_runner
        if branch_runner is None:
            branch_runner = live_branch_runner

    trial_tasks = [
        vanilla_runner(
            args=args,
            sample=sample,
            sampling_params=sampling_params,
            trial_idx=i,
            group_index=group_index,
            base_index=base_index,
        )
        for i in range(k)
    ]
    raw = await asyncio.gather(*trial_tasks, return_exceptions=True)

    trials: list[VanillaTrialResult] = []
    vanilla_samples: list[Sample] = []
    for i, res in enumerate(raw):
        if isinstance(res, Exception):
            logger.warning("[hybrid] vanilla trial=%d skipped: %s", i, res)
            continue
        bundle, samples, is_solved, turn_lps = res
        for s in samples:
            s.metadata = s.metadata or {}
            s.metadata["sample_kind"] = "vanilla"
            s.metadata["trial_idx"] = i
            s.group_index = group_index
        vanilla_samples.extend(samples)
        trials.append(
            VanillaTrialResult(
                trial_idx=i,
                bundle=bundle,
                samples=samples,
                is_solved=bool(is_solved),
                turn_logprobs=turn_lps,
            )
        )

    if not trials:
        # Match Path A generate._abort shape so Megatron get_batch never sees
        # empty tokens (pad narrow with prompt_length-1 would go negative).
        sample.tokens = [0, 0]
        sample.response = ""
        sample.response_length = 1
        sample.loss_mask = [0]
        sample.rollout_log_probs = [0.0]
        sample.reward = 0.0
        sample.remove_sample = True
        sample.status = Sample.Status.ABORTED
        sample.metadata = {
            **(sample.metadata or {}),
            "abort_reason": "all_vanilla_trials_failed",
            "sample_kind": "vanilla",
        }
        logger.warning("[hybrid] all vanilla trials failed; returning aborted sample")
        return [sample]

    if all(t.is_solved for t in trials):
        return vanilla_samples

    selected = select_branch_turns(trials, k)
    if not selected:
        return vanilla_samples

    by_trial = {t.trial_idx: t for t in trials}
    branch_tasks = []
    for sel in selected:
        src = by_trial[sel.source_trial_idx]
        for b in range(k):
            branch_tasks.append(
                branch_runner(
                    args=args,
                    sample=sample,
                    sampling_params=sampling_params,
                    bundle=src.bundle,
                    source_trial_idx=sel.source_trial_idx,
                    step_t=sel.step_t,
                    branch_idx=b,
                    group_index=group_index,
                    edit_ppl=sel.edit_ppl,
                )
            )

    branch_raw = await asyncio.gather(*branch_tasks, return_exceptions=True)
    branch_samples: list[Sample] = []
    for res in branch_raw:
        if isinstance(res, Exception):
            logger.warning("[hybrid] dropped branch: %s", res)
            continue
        for s in res:
            s.metadata = s.metadata or {}
            s.metadata.setdefault("sample_kind", "branch")
            # Full continuation is trainable (not first_action).
            if s.loss_mask is None and s.tokens is not None:
                s.loss_mask = [1] * max(0, int(getattr(s, "response_length", 0) or len(s.tokens or [])))
            branch_samples.append(s)

    return vanilla_samples + branch_samples
