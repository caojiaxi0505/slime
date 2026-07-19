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
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

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


def _branch_drop_bucket(error: BaseException) -> str:
    """Classify one failed branch into one stable, low-cardinality bucket."""
    message = str(error).lower()
    if isinstance(error, TimeoutError) or "timeout" in message or "timed out" in message:
        return "timeout"
    if "expected one echo for tool_use" in message or "expected at most one echo" in message:
        return "resume_tool_echo"
    if "expected one result for tool_use" in message:
        return "resume_missing_result"
    if "task/agent subagent dispatch" in message:
        return "resume_subagent"
    if (
        "after a resumed end_turn" in message
        or "without pending resumed tool calls" in message
        or "without pending tool calls" in message
    ):
        return "resume_no_pending"
    if "tool schema differs" in message or "runtime tool schema" in message:
        return "resume_tool_schema"
    if "token_exact" in message or "resumed request" in message:
        return "resume_other"
    if "rebuild" in message or "workspace" in message:
        return "workspace_rebuild"
    return "other"


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


def _stamp_hybrid_loss_group_ids(samples: list[Sample]) -> list[Sample]:
    """Give each independent hybrid episode its own loss identity.

    ``rollout_id`` must stay shared by every sample returned for one outer
    prompt because slime uses it for the fixed-GBS schedule.  Loss grouping is
    finer: compact segments from one trial/branch share an id, while different
    Stage-1 trials and Stage-2 branches do not.
    """
    for s in samples:
        md = s.metadata or {}
        kind = str(md.get("sample_kind") or "unknown")
        episode_uid = md.get("branch_uid")
        if episode_uid is None and kind == "vanilla" and md.get("trial_idx") is not None:
            episode_uid = f"trial:{md['trial_idx']}"
        if episode_uid is None and kind == "branch":
            episode_uid = f"{md.get('step_group_key', 'group:?')}:{md.get('branch_idx', 'branch:?')}"
        if episode_uid is None:
            episode_uid = s.session_id or f"sample:{s.index}"
        s.loss_group_id = f"hybrid:{s.rollout_id}:{kind}:{episode_uid}"
    return samples


def _stamp_hybrid_walls(
    samples: list[Sample],
    *,
    stage1_wall_sec: float,
    stage2_wall_sec: float,
    total_wall_sec: float,
    hybrid_stats: dict[str, int] | None = None,
) -> list[Sample]:
    """Attach per-prompt hybrid phase walls for wandb (step-GRPO extras)."""
    for s in samples:
        s.metadata = s.metadata or {}
        s.metadata["hybrid_stage1_wall_sec"] = float(stage1_wall_sec)
        s.metadata["hybrid_stage2_wall_sec"] = float(stage2_wall_sec)
        s.metadata["hybrid_total_wall_sec"] = float(total_wall_sec)
        if hybrid_stats:
            s.metadata.update(hybrid_stats)
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


def pre_checkpoint_snapshot_index(
    snapshot_tool_use_ids: Sequence[str],
    checkpoint_tool_use_ids: Sequence[str],
    target_tool_use_id: str,
) -> int:
    """Return the workspace snapshot immediately before one model turn.

    A prompt checkpoint owns the complete set of tools generated by that
    assistant turn. Parallel tools can finish in either order, so the pre-turn
    state is the snapshot before the first member of that set, not simply the
    snapshot before the selected edit.
    """
    snapshot_ids = [str(value or "") for value in snapshot_tool_use_ids]
    checkpoint_ids = [str(value or "") for value in checkpoint_tool_use_ids]
    target = str(target_tool_use_id or "")
    if not snapshot_ids or any(not value for value in snapshot_ids):
        raise ValueError("snapshot_tool_use_ids_missing")
    if len(set(snapshot_ids)) != len(snapshot_ids):
        raise ValueError("snapshot_tool_use_ids_duplicate")
    if not checkpoint_ids or any(not value for value in checkpoint_ids):
        raise ValueError("checkpoint_tool_use_ids_missing")
    if len(set(checkpoint_ids)) != len(checkpoint_ids):
        raise ValueError("checkpoint_tool_use_ids_duplicate")
    if target not in checkpoint_ids:
        raise ValueError(f"checkpoint_does_not_generate_target_tool:{target}")
    missing = [value for value in checkpoint_ids if value not in snapshot_ids]
    if missing:
        raise ValueError(f"checkpoint_tools_missing_snapshots:{missing[:3]}")
    positions = sorted(snapshot_ids.index(value) for value in checkpoint_ids)
    if positions != list(range(positions[0], positions[-1] + 1)):
        raise ValueError(
            f"checkpoint_snapshot_group_not_contiguous:positions={positions[:8]}"
        )
    return positions[0] - 1


def collect_patch_candidates(trials: Sequence[VanillaTrialResult]) -> list[PatchTurnCandidate]:
    """Pool patch turns from all **failed** trials."""
    clip = _ppl_clip()
    out: list[PatchTurnCandidate] = []
    for tr in trials:
        if tr.is_solved:
            continue
        exact_ready, exact_error = tr.bundle.token_exact_readiness()
        if not exact_ready:
            logger.warning(
                "[hybrid] Stage-2 ineligible: token-exact state incomplete "
                "instance=%s trial=%d bundle=%s reason=%s",
                tr.bundle.instance_id,
                tr.trial_idx,
                tr.bundle.dir,
                exact_error,
            )
            continue
        # Transcript is retained for audit only. Token-exact correctness comes
        # from prompt checkpoint + native session + workspace state. Never
        # replay it or silently start a fresh agent when audit validation fails.
        if not tr.bundle.transcript_valid:
            logger.warning(
                "[hybrid] transcript audit invalid; token-exact state retained "
                "instance=%s trial=%d bundle=%s reason=%s",
                tr.bundle.instance_id,
                tr.trial_idx,
                tr.bundle.dir,
                tr.bundle.transcript_error or "unknown",
            )
        snapshot_ids = [str(step.tool_use_id or "") for step in tr.bundle.steps]
        diffs = [tr.bundle.step_diff(i) for i in range(tr.bundle.num_steps)]
        for p in iter_patch_turn_ppls(diffs, tr.turn_logprobs, clip=clip):
            target_tool_use_id = snapshot_ids[p.step_t]
            try:
                checkpoint = tr.bundle.checkpoint_for_tool_use_id(target_tool_use_id)
            except (OSError, ValueError) as exc:
                logger.warning(
                    "[hybrid] Stage-2 candidate dropped: checkpoint lookup failed "
                    "instance=%s trial=%d edit=%d tool=%s reason=%s",
                    tr.bundle.instance_id,
                    tr.trial_idx,
                    p.step_t,
                    target_tool_use_id,
                    exc,
                )
                continue
            if checkpoint is None:
                logger.warning(
                    "[hybrid] Stage-2 candidate dropped: no checkpoint for target tool "
                    "instance=%s trial=%d edit=%d tool=%s",
                    tr.bundle.instance_id,
                    tr.trial_idx,
                    p.step_t,
                    target_tool_use_id,
                )
                continue
            if checkpoint.get("chain_kind") != "main" or checkpoint.get("request_kind") == "wipe":
                logger.warning(
                    "[hybrid] Stage-2 candidate dropped: unsupported checkpoint chain/request "
                    "instance=%s trial=%d edit=%d chain=%s request=%s",
                    tr.bundle.instance_id,
                    tr.trial_idx,
                    p.step_t,
                    checkpoint.get("chain_kind"),
                    checkpoint.get("request_kind"),
                )
                continue
            checkpoint_tool_ids = [
                str(value) for value in checkpoint.get("generated_tool_use_ids") or []
            ]
            checkpoint_tool_names = {
                str(tool_use_id): str(name)
                for tool_use_id, name in (
                    checkpoint.get("generated_tool_use_names") or {}
                ).items()
            }
            if set(checkpoint_tool_names) != set(checkpoint_tool_ids) or any(
                not name for name in checkpoint_tool_names.values()
            ):
                logger.warning(
                    "[hybrid] Stage-2 candidate dropped: checkpoint tool names missing "
                    "instance=%s trial=%d edit=%d",
                    tr.bundle.instance_id,
                    tr.trial_idx,
                    p.step_t,
                )
                continue
            unsupported_tools = sorted(
                {
                    name
                    for name in checkpoint_tool_names.values()
                    if name in {"Task", "Agent"}
                }
            )
            if unsupported_tools:
                logger.warning(
                    "[hybrid] Stage-2 candidate dropped: unsupported generated tool "
                    "instance=%s trial=%d edit=%d tools=%s",
                    tr.bundle.instance_id,
                    tr.trial_idx,
                    p.step_t,
                    unsupported_tools,
                )
                continue
            try:
                branch_step_t = pre_checkpoint_snapshot_index(
                    snapshot_ids,
                    checkpoint_tool_ids,
                    target_tool_use_id,
                )
            except ValueError as exc:
                logger.warning(
                    "[hybrid] Stage-2 candidate dropped instance=%s trial=%d edit=%d bundle=%s reason=%s",
                    tr.bundle.instance_id,
                    tr.trial_idx,
                    p.step_t,
                    tr.bundle.dir,
                    exc,
                )
                continue
            workspace_ready, workspace_error = tr.bundle.token_exact_readiness(branch_step_t)
            if not workspace_ready:
                logger.warning(
                    "[hybrid] Stage-2 candidate dropped: pre-turn workspace metadata invalid "
                    "instance=%s trial=%d edit=%d pre=%d reason=%s",
                    tr.bundle.instance_id,
                    tr.trial_idx,
                    p.step_t,
                    branch_step_t,
                    workspace_error,
                )
                continue
            out.append(
                PatchTurnCandidate(
                    source_trial_idx=tr.trial_idx,
                    edit_step_i=p.step_t,
                    branch_step_t=branch_step_t,
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
    hybrid_stats = {
        "hybrid_num_patch_candidates": 0,
        "hybrid_num_selected_edits": 0,
        "hybrid_num_branch_tasks": 0,
        "hybrid_num_dropped_branches": 0,
        "hybrid_num_dropped_timeout": 0,
        "hybrid_num_dropped_resume_tool_echo": 0,
        "hybrid_num_dropped_resume_missing_result": 0,
        "hybrid_num_dropped_resume_no_pending": 0,
        "hybrid_num_dropped_resume_tool_schema": 0,
        "hybrid_num_dropped_resume_subagent": 0,
        "hybrid_num_dropped_resume_other": 0,
        "hybrid_num_dropped_workspace_rebuild": 0,
        "hybrid_num_dropped_other": 0,
    }

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
        samples = _stamp_hybrid_loss_group_ids(samples)
        return _stamp_hybrid_walls(
            samples,
            stage1_wall_sec=stage1_wall,
            stage2_wall_sec=stage2_wall,
            total_wall_sec=total,
            hybrid_stats=hybrid_stats,
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

    candidates = collect_patch_candidates(trials)
    selected = select_top_patch_turns(candidates, k)
    hybrid_stats["hybrid_num_patch_candidates"] = len(candidates)
    hybrid_stats["hybrid_num_selected_edits"] = len(selected)
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
                    edit_step_i=sel.edit_step_i,
                    branch_step_t=sel.branch_step_t,
                    branch_idx=b,
                    group_index=group_index,
                    edit_ppl=sel.edit_ppl,
                )
            )
    hybrid_stats["hybrid_num_branch_tasks"] = len(branch_tasks)

    t_stage2 = time.time()
    branch_raw = await asyncio.gather(*branch_tasks, return_exceptions=True)
    stage2_wall = time.time() - t_stage2
    branch_samples: list[Sample] = []
    for res in branch_raw:
        if isinstance(res, Exception):
            bucket = _branch_drop_bucket(res)
            logger.warning("[hybrid] dropped branch bucket=%s: %s", bucket, res)
            hybrid_stats["hybrid_num_dropped_branches"] += 1
            hybrid_stats[f"hybrid_num_dropped_{bucket}"] += 1
            continue
        for s in res:
            s.metadata = s.metadata or {}
            s.metadata.setdefault("sample_kind", "branch")
            # Do not rewrite loss_mask: merge_turns already marks every assistant
            # token in the continuation as trainable (vs first_action), and keeps
            # tool/context tails at 0 with placeholder rollout_log_probs.
            branch_samples.append(s)

    return _finish(vanilla_samples + branch_samples, stage2_wall=stage2_wall)
