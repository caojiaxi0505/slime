"""Hybrid generate with turn-level teacher SFT.

Stage-1 is the usual student GRPO group (it also captures the bundles that make
resume possible). Stage-2 relabels selected turns of the failed trials: the
workspace is rebuilt at that turn, the native session is resumed token-exactly,
and the remote teacher takes ``STEP_GRPO_TEACHER_MAX_STEPS`` real turns whose
tool calls actually run. Those teacher turns become the trainable suffix of a
student SFT sample.

``STEP_GRPO_TEACHER_SFT_MODE`` decides which rows are returned for training.

* ``sft_only`` (default): return only ``sample_kind=teacher_sft`` rows.
* ``hybrid``: return Stage-1 ``vanilla`` rows + teacher SFT rows.

Cost knobs (one relabeled turn is one sandbox):
``STEP_GRPO_TEACHER_TURN_SELECT`` and ``STEP_GRPO_TEACHER_TURN_MAX_PER_TRIAL``
choose which turns get relabeled; ``STEP_GRPO_TEACHER_MAX_STEPS`` (``0`` runs to
the end and grades) bounds the teacher inside each relabel.

Wire training with::

    --custom-generate-function-path \\
      examples.claudecode_ags.step_reconstruct.hybrid_sft_generate.hybrid_sft_generate

    # sft_only
    --loss-type sft_loss --disable-compute-advantages-and-returns \
    --rollout-sample-filter-path \
      examples.claudecode_ags.step_reconstruct.hybrid_sft_generate.sft_only_filter

    # hybrid
    --loss-type custom_loss \\
    --custom-loss-function-path \\
      examples.claudecode_ags.step_reconstruct.hybrid_teacher_sft_loss.hybrid_teacher_sft_loss \\
    --custom-reward-post-process-path \\
      examples.claudecode_ags.step_reconstruct.step_grpo_advantage.post_process_rewards \\
    --rollout-sample-filter-path \\
      examples.claudecode_ags.step_reconstruct.step_grpo_advantage.filter
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Any

from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def _flatten_samples(data: list[Any]) -> list[Sample]:
    flat: list[Sample] = []
    for item in data:
        if isinstance(item, list):
            flat.extend(_flatten_samples(item))
        else:
            flat.append(item)
    return flat


def _outer_rollout_id(sample: Sample) -> int:
    """Return the scheduler identity shared by one original SWE task."""
    value = sample.rollout_id
    if value is None:
        value = sample.group_index
    if value is None:
        value = sample.index
    if value is None:
        raise ValueError("teacher SFT sample is missing rollout/group/index identity")
    return int(value)


def _active_loss_tokens(sample: Sample) -> int:
    if sample.remove_sample or bool(getattr(sample, "is_filtered_out", False)):
        return 0
    if sample.loss_mask is None:
        return int(sample.response_length or 0)
    return sum(int(value) for value in sample.loss_mask)


def _empty_teacher_placeholder(samples: list[Sample]) -> Sample:
    """Represent one completed raw task that produced no teacher target.

    slime schedules and divides the loss by outer ``rollout_id`` units.  A
    zero-mask placeholder keeps such a task inside the fixed 16-task barrier
    without adding a teacher target or an expensive full-trajectory forward.
    """
    if not samples:
        raise ValueError("cannot build teacher SFT placeholder without a Stage-1 sample")

    base = min(samples, key=lambda row: len(row.tokens or []))
    out = copy.copy(base)
    tokens = list(base.tokens or [])
    if not tokens:
        tokens = [0, 0]
    elif len(tokens) == 1:
        tokens = [tokens[0], tokens[0]]
    else:
        tokens = tokens[:2]

    outer = _outer_rollout_id(base)
    out.tokens = tokens
    out.response = ""
    out.response_length = 1
    out.loss_mask = [0]
    out.rollout_log_probs = [0.0]
    out.rollout_top_p_token_ids = None
    out.rollout_top_p_token_offsets = None
    out.rollout_routed_experts = None
    out.teacher_log_probs = None
    out.multimodal_train_inputs = None
    out.train_metadata = None
    out.reward = 0.0
    out.remove_sample = True
    out.status = Sample.Status.COMPLETED
    out.loss_weight = 0.0
    out.loss_group_id = f"hybrid:{outer}:teacher_sft:empty"
    out.metadata = {
        **(base.metadata or {}),
        "sample_kind": "teacher_sft",
        "teacher_sft_placeholder": True,
        "teacher_sft_placeholder_reason": "no_teacher_targets",
        "teacher_trainable_tokens": 0,
        "teacher_prefix_tokens": 0,
        "teacher_num_turns": 0,
        "branch_uid": f"teacher-empty:{outer}",
    }
    return out


def sft_only_filter(args: Any, data: list[Any]) -> None:
    """Assign the exact SFT-only objective after the 16-task barrier.

    The raw task count is fixed, while the number of tasks with usable teacher
    targets and the number of relabels per task are variable.  First average
    relabel episodes within each active task, then scale active tasks so the
    downstream fixed-GBS divisor produces a mean over active tasks.
    """
    from examples.claudecode_ags.step_reconstruct.step_grpo_advantage import _assign_loss_weights

    flat = _flatten_samples(data)
    if not flat:
        raise ValueError("SFT-only rollout barrier returned no samples")
    bad_kinds = [
        str((sample.metadata or {}).get("sample_kind") or "")
        for sample in flat
        if (sample.metadata or {}).get("sample_kind") != "teacher_sft"
    ]
    if bad_kinds:
        raise ValueError(f"SFT-only batch contains non-teacher rows: {sorted(set(bad_kinds))}")

    scheduled_outers = {_outer_rollout_id(sample) for sample in flat}
    expected = int(getattr(args, "rollout_batch_size", len(scheduled_outers)))
    global_batch_size = int(getattr(args, "global_batch_size", expected))
    if len(data) != expected:
        raise ValueError(
            f"SFT-only barrier has {len(data)} completed task groups, expected {expected}"
        )
    group_outers: list[int] = []
    for position, group in enumerate(data):
        outers = {_outer_rollout_id(sample) for sample in _flatten_samples([group])}
        if len(outers) != 1:
            raise ValueError(
                f"SFT-only task group {position} mixes scheduler identities: {sorted(outers)}"
            )
        group_outers.append(next(iter(outers)))
    if len(set(group_outers)) != len(group_outers):
        raise ValueError("SFT-only barrier contains duplicate raw task scheduler identities")
    planned_values = [
        (sample.metadata or {}).get("hybrid_num_stage1_planned_trials") for sample in flat
    ]
    planned_trials = {int(value) for value in planned_values if value is not None}
    if any(value is None for value in planned_values) or planned_trials != {8}:
        raise ValueError(
            "SFT-only task groups must each contain 8 planned student trials; "
            f"observed metadata values={sorted(planned_trials)}"
        )
    if len(scheduled_outers) != expected:
        raise ValueError(
            f"SFT-only barrier has {len(scheduled_outers)} raw tasks, expected {expected}"
        )
    if global_batch_size != expected:
        raise ValueError(
            "SFT-only fixed-task barrier currently requires "
            f"global_batch_size == rollout_batch_size ({global_batch_size} != {expected})"
        )

    _assign_loss_weights(flat)
    active_outers = {
        _outer_rollout_id(sample) for sample in flat if _active_loss_tokens(sample) > 0
    }
    if not active_outers:
        raise RuntimeError(
            "all scheduled SFT tasks produced zero teacher targets; refusing a zero-loss optimizer step"
        )

    # The train side divides by fixed GBS.  Scaling each active task from 1 to
    # B/A therefore gives (1/A) * sum_g mean_relabels(mean_tokens(NLL)).
    prompt_scale = expected / len(active_outers)
    for sample in flat:
        outer = _outer_rollout_id(sample)
        if outer in active_outers:
            if sample.loss_weight is None:
                raise ValueError("active teacher SFT row is missing its base loss weight")
            sample.loss_weight = float(sample.loss_weight) * prompt_scale
        else:
            sample.loss_weight = 0.0
        sample.metadata = sample.metadata or {}
        sample.metadata.update(
            {
                "teacher_sft_scheduled_prompts": expected,
                "teacher_sft_active_prompts": len(active_outers),
                "teacher_sft_prompt_weight": prompt_scale if outer in active_outers else 0.0,
                "teacher_sft_total_loss_weight": float(expected),
                "teacher_sft_weighting": "mean_active_prompt_mean_relabel_mean_token",
                "loss_weight": float(sample.loss_weight),
            }
        )

    # Validate unique loss groups, not rows: a future compact segment fan-out
    # may repeat one episode's coefficient on several segments.
    groups_by_outer: dict[int, dict[str | int, float]] = {}
    for sample in flat:
        outer = _outer_rollout_id(sample)
        if sample.loss_group_id is None or sample.loss_weight is None:
            raise ValueError("teacher SFT row is missing explicit loss identity/weight")
        groups = groups_by_outer.setdefault(outer, {})
        previous = groups.setdefault(sample.loss_group_id, float(sample.loss_weight))
        if previous != float(sample.loss_weight):
            raise ValueError(f"teacher loss group {sample.loss_group_id!r} has inconsistent weights")

    for outer, groups in groups_by_outer.items():
        actual = sum(groups.values())
        expected_outer = prompt_scale if outer in active_outers else 0.0
        if abs(actual - expected_outer) > 1e-8:
            raise ValueError(
                f"teacher task {outer} loss weights sum to {actual}, expected {expected_outer}"
            )
    total = sum(sum(groups.values()) for groups in groups_by_outer.values())
    if abs(total - expected) > 1e-8:
        raise ValueError(f"teacher batch loss weights sum to {total}, expected {expected}")

    logger.info(
        "[hybrid-sft] fixed-task barrier scheduled=%d active=%d rows=%d "
        "active_prompt_weight=%.8g total_loss_weight=%.8g",
        expected,
        len(active_outers),
        len(flat),
        prompt_scale,
        total,
    )


def teacher_sft_mode() -> str:
    mode = (os.environ.get("STEP_GRPO_TEACHER_SFT_MODE") or "sft_only").strip().lower()
    if mode not in {"sft_only", "hybrid"}:
        raise ValueError("STEP_GRPO_TEACHER_SFT_MODE must be sft_only or hybrid, " f"got {mode!r}")
    return mode


async def hybrid_sft_generate(
    args,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
    *,
    vanilla_runner=None,
    stage2_fn=None,
):
    """Stage-1 student capture + Stage-2 teacher SFT."""
    from examples.claudecode_ags.step_reconstruct.hybrid_generate import hybrid_generate

    mode = teacher_sft_mode()
    if stage2_fn is None:
        from examples.claudecode_ags.step_reconstruct.teacher_turn_sft import teacher_turn_samples

        stage2_fn = teacher_turn_samples

    if not evaluation:
        from examples.claudecode_ags.step_reconstruct.teacher_branch_runner import ensure_teacher_adapter

        ensure_teacher_adapter(args)

    samples = await hybrid_generate(
        args,
        sample,
        sampling_params,
        evaluation=evaluation,
        vanilla_runner=vanilla_runner,
        stage2_fn=stage2_fn,
    )

    if evaluation or mode == "hybrid":
        return samples

    teacher_only = [s for s in samples if (s.metadata or {}).get("sample_kind") == "teacher_sft"]
    num_teacher_targets = len(teacher_only)
    if not teacher_only:
        teacher_only = [_empty_teacher_placeholder(samples)]
    logger.info(
        "[hybrid-sft] mode=sft_only kept_teacher=%d placeholders=%d dropped_other=%d",
        sum(not bool((s.metadata or {}).get("teacher_sft_placeholder")) for s in teacher_only),
        sum(bool((s.metadata or {}).get("teacher_sft_placeholder")) for s in teacher_only),
        len(samples) - num_teacher_targets,
    )
    return teacher_only
