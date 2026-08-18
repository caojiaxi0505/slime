"""Tests for hybrid teacher-SFT generate mode filtering and advantage skip."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from examples.claudecode_ags.step_reconstruct.hybrid_sft_generate import (
    _empty_teacher_placeholder,
    sft_only_filter,
    teacher_sft_mode,
)
from examples.claudecode_ags.step_reconstruct.step_grpo_advantage import post_process_rewards

from slime.utils.loss_groups import build_loss_group_fields
from slime.utils.types import Sample


def _s(kind: str, *, reward: float = 1.0, index: int = 0, trial_idx: int = 0) -> Sample:
    return Sample(
        prompt="p",
        index=index,
        group_index=0,
        reward=reward,
        loss_mask=[1, 1],
        metadata={"sample_kind": kind, "trial_idx": trial_idx, "step_group_key": "0:0:edit:1"},
        loss_group_id=f"g-{kind}-{index}",
        loss_weight=1.0,
    )


def _teacher_row(
    outer: int,
    episode: int,
    *,
    trainable_tokens: int = 2,
    placeholder: bool = False,
) -> Sample:
    response_length = max(trainable_tokens, 1)
    return Sample(
        prompt="p",
        tokens=[1] * (response_length + 1),
        response_length=response_length,
        loss_mask=[0] * response_length if placeholder else [1] * response_length,
        rollout_log_probs=[0.0] * response_length,
        index=outer * 1000 + episode,
        group_index=outer,
        rollout_id=outer,
        reward=0.0,
        remove_sample=placeholder,
        status=Sample.Status.COMPLETED,
        metadata={
            "sample_kind": "teacher_sft",
            "branch_uid": f"teach:{outer}:{episode}",
            "teacher_sft_placeholder": placeholder,
            "teacher_trainable_tokens": 0 if placeholder else trainable_tokens,
            "teacher_prefix_tokens": 1,
            "hybrid_num_stage1_planned_trials": 8,
        },
        loss_group_id=f"hybrid:{outer}:teacher_sft:{episode}",
        loss_weight=0.0 if placeholder else None,
    )


def test_teacher_sft_mode_validation(monkeypatch):
    monkeypatch.setenv("STEP_GRPO_TEACHER_SFT_MODE", "hybrid")
    assert teacher_sft_mode() == "hybrid"
    monkeypatch.setenv("STEP_GRPO_TEACHER_SFT_MODE", "nope")
    with pytest.raises(ValueError, match="sft_only or hybrid"):
        teacher_sft_mode()


def test_post_process_skips_teacher_sft_in_branch_groups():
    samples = [
        _s("vanilla", reward=1.0, index=0, trial_idx=0),
        _s("vanilla", reward=0.0, index=1, trial_idx=1),
        _s("teacher_sft", reward=1.0, index=2),
        _s("branch", reward=1.0, index=3),
        _s("branch", reward=0.0, index=4),
    ]
    # Give branch samples distinct step keys matching post_process expectations.
    samples[3].metadata["step_group_key"] = "0:0:edit:1"
    samples[3].metadata["source_trial_idx"] = 0
    samples[4].metadata["step_group_key"] = "0:0:edit:1"
    samples[4].metadata["source_trial_idx"] = 0
    raw, adv = post_process_rewards(SimpleNamespace(advantage_estimator="grpo", reward_key=None), samples)
    assert len(raw) == 5
    assert adv[2] == 0.0  # teacher_sft never gets GRPO advantage
    assert adv[0] != 0.0 or adv[1] != 0.0  # vanilla group has signal


def test_loss_stage_ids_mark_teacher_sft_as_2():
    samples = [_s("vanilla", index=0), _s("branch", index=1), _s("teacher_sft", index=2)]
    fields = build_loss_group_fields(samples, [0, 0, 0], [[1], [1], [1]])
    assert fields["loss_stage_ids"] == [0, 1, 2]


def test_teacher_sft_requires_explicit_weight_before_training():
    teacher = _teacher_row(0, 0)
    with pytest.raises(ValueError, match="missing explicit loss_weight"):
        build_loss_group_fields([teacher], [0], [[1, 1]])


def test_sft_only_filter_waits_for_16_tasks_and_normalizes_variable_targets():
    # Two raw tasks produce 2 and 4 relabels. The other 14 completed tasks
    # contribute scheduler placeholders but no SFT loss.
    data = []
    for outer in range(16):
        if outer == 0:
            rows = [_teacher_row(outer, episode) for episode in range(2)]
        elif outer == 1:
            rows = [_teacher_row(outer, episode) for episode in range(4)]
        else:
            rows = [_teacher_row(outer, 0, trainable_tokens=0, placeholder=True)]
        # Same nested shape as one n_samples=1 custom-generate group.
        data.append([rows])

    sft_only_filter(SimpleNamespace(rollout_batch_size=16, global_batch_size=16), data)
    flat = [row for group in data for generated in group for row in generated]

    active0 = [row for row in flat if row.rollout_id == 0]
    active1 = [row for row in flat if row.rollout_id == 1]
    placeholders = [row for row in flat if row.rollout_id not in {0, 1}]
    assert [row.loss_weight for row in active0] == [4.0, 4.0]
    assert [row.loss_weight for row in active1] == [2.0, 2.0, 2.0, 2.0]
    assert all(row.loss_weight == 0.0 for row in placeholders)
    assert sum(row.loss_weight for row in flat) == 16.0
    assert all(row.metadata["teacher_sft_scheduled_prompts"] == 16 for row in flat)
    assert all(row.metadata["teacher_sft_active_prompts"] == 2 for row in flat)


def test_sft_only_filter_rejects_partial_task_barrier():
    data = [[[_teacher_row(outer, 0)]] for outer in range(15)]
    with pytest.raises(ValueError, match="15 completed task groups, expected 16"):
        sft_only_filter(SimpleNamespace(rollout_batch_size=16, global_batch_size=16), data)


def test_sft_only_filter_refuses_zero_loss_optimizer_step():
    data = [
        [[_teacher_row(outer, 0, trainable_tokens=0, placeholder=True)]]
        for outer in range(16)
    ]
    with pytest.raises(RuntimeError, match="zero teacher targets"):
        sft_only_filter(SimpleNamespace(rollout_batch_size=16, global_batch_size=16), data)


def test_empty_teacher_placeholder_is_tiny_and_clears_replay_fields():
    source = _teacher_row(3, 7)
    source.metadata["sample_kind"] = "vanilla"
    source.rollout_top_p_token_ids = [1, 2]
    source.rollout_top_p_token_offsets = [0, 1, 2]
    source.rollout_routed_experts = [[1], [1]]
    source.teacher_log_probs = [0.1, 0.2]

    placeholder = _empty_teacher_placeholder([source])
    assert placeholder.rollout_id == 3
    assert placeholder.response_length == 1
    assert placeholder.loss_mask == [0]
    assert placeholder.remove_sample is True
    assert placeholder.loss_weight == 0.0
    assert placeholder.metadata["teacher_sft_placeholder"] is True
    assert placeholder.rollout_top_p_token_ids is None
    assert placeholder.rollout_top_p_token_offsets is None
    assert placeholder.rollout_routed_experts is None
    assert placeholder.teacher_log_probs is None
