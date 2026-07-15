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
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Sequence

from examples.claudecode_ags.step_reconstruct.edit_ppl import StepTurnAlignmentError, iter_patch_turn_ppls
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


def _shared_rollout_id(sample: Sample) -> int:
    """All siblings from one hybrid_generate must share this id (slime compact)."""
    if sample.rollout_id is not None:
        return int(sample.rollout_id)
    return int(sample.index or 0)


def _stamp_shared_rollout_id(samples: list[Sample], parent: Sample) -> list[Sample]:
    rid = _shared_rollout_id(parent)
    for s in samples:
        s.rollout_id = rid
    return samples


def _stamp_hybrid_walls(
    samples: list[Sample],
    *,
    stage1_wall_sec: float,
    stage2_wall_sec: float,
    total_wall_sec: float,
) -> list[Sample]:
    """Attach per-prompt hybrid phase walls for wandb (step-GRPO extras)."""
    for s in samples:
        s.metadata = s.metadata or {}
        s.metadata["hybrid_stage1_wall_sec"] = float(stage1_wall_sec)
        s.metadata["hybrid_stage2_wall_sec"] = float(stage2_wall_sec)
        s.metadata["hybrid_total_wall_sec"] = float(total_wall_sec)
    return samples


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
    t_hybrid0 = time.time()
    t_stage1 = time.time()
    raw = await asyncio.gather(*trial_tasks, return_exceptions=True)
    stage1_wall = time.time() - t_stage1

    trials: list[VanillaTrialResult] = []
    vanilla_samples: list[Sample] = []
    for i, res in enumerate(raw):
        if isinstance(res, StepTurnAlignmentError):
            # Strict: never silently degrade to vanilla-only on alignment bugs.
            raise res
        if isinstance(res, Exception):
            logger.warning("[hybrid] vanilla trial=%d skipped: %s", i, res)
            continue
        bundle, samples, is_solved, turn_lps = res
        for s in samples:
            s.metadata = s.metadata or {}
            s.metadata["sample_kind"] = "vanilla"
            s.metadata["trial_idx"] = i
            s.metadata.setdefault("branch_uid", f"v:{group_index}:t{i}")
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

    def _finish(samples: list[Sample], *, stage2_wall: float = 0.0) -> list[Sample]:
        total = time.time() - t_hybrid0
        samples = _stamp_shared_rollout_id(samples, sample)
        return _stamp_hybrid_walls(
            samples,
            stage1_wall_sec=stage1_wall,
            stage2_wall_sec=stage2_wall,
            total_wall_sec=total,
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
        sample.rollout_id = _shared_rollout_id(sample)
        return _finish([sample])

    if all(t.is_solved for t in trials):
        return _finish(vanilla_samples)

    selected = select_branch_turns(trials, k)
    if not selected:
        return _finish(vanilla_samples)

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

    t_stage2 = time.time()
    branch_raw = await asyncio.gather(*branch_tasks, return_exceptions=True)
    stage2_wall = time.time() - t_stage2
    branch_samples: list[Sample] = []
    for res in branch_raw:
        if isinstance(res, Exception):
            logger.warning("[hybrid] dropped branch: %s", res)
            continue
        for s in res:
            s.metadata = s.metadata or {}
            s.metadata.setdefault("sample_kind", "branch")
            # Do not rewrite loss_mask: merge_turns already marks every assistant
            # token in the continuation as trainable (vs first_action), and keeps
            # tool/context tails at 0 with placeholder rollout_log_probs.
            branch_samples.append(s)

    return _finish(vanilla_samples + branch_samples, stage2_wall=stage2_wall)
