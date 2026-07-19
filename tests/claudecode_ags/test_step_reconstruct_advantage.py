import os
from types import SimpleNamespace

import pytest

from examples.claudecode_ags.step_reconstruct.step_grpo_advantage import filter, post_process_rewards
from slime.utils.types import Sample


def _s(
    *,
    group_index,
    index,
    reward,
    kind,
    step_group_key=None,
    loss_mask=None,
    trial_idx=0,
    rollout_id=None,
    branch_uid=None,
):
    md = {"sample_kind": kind, "trial_idx": trial_idx}
    if step_group_key is not None:
        md["step_group_key"] = step_group_key
    if branch_uid is not None:
        md["branch_uid"] = branch_uid
    return Sample(
        group_index=group_index,
        index=index,
        # Production hybrid stamps one shared rollout_id on all siblings.
        rollout_id=rollout_id if rollout_id is not None else index,
        reward=reward,
        prompt="p",
        metadata=md,
        loss_mask=loss_mask if loss_mask is not None else [1, 1],
    )


def test_vanilla_and_branch_groups():
    # vanilla: two trials rewards 1 and 0
    samples = [
        _s(group_index=0, index=10, reward=0.5, kind="vanilla", trial_idx=0),
        _s(group_index=0, index=10, reward=0.5, kind="vanilla", trial_idx=0),  # same trial segments
        _s(group_index=0, index=11, reward=0.0, kind="vanilla", trial_idx=1),
        # branch group at s_t: rewards 1 and 0
        _s(group_index=0, index=100, reward=1.0, kind="branch", step_group_key="0:1:2"),
        _s(group_index=0, index=101, reward=0.0, kind="branch", step_group_key="0:1:2"),
    ]
    args = SimpleNamespace(advantage_estimator="grpo", reward_key=None)
    os.environ["STEP_GRPO_STD_NORMALIZATION"] = "0"
    raw, adv = post_process_rewards(args, samples)
    assert raw[0] == 0.5
    # vanilla episodes [1.0, 0.0] → mean 0.5 → +0.5 / -0.5 broadcast
    assert adv[0] == adv[1] == 0.5
    assert adv[2] == -0.5
    assert adv[3] == 0.5
    assert adv[4] == -0.5


def test_shared_rollout_id_does_not_collapse_vanilla_trials():
    """Regression: hybrid siblings share rollout_id but K trials must stay distinct."""
    shared_rid = 42
    samples = [
        _s(
            group_index=0,
            index=10,
            reward=1.0,
            kind="vanilla",
            trial_idx=0,
            rollout_id=shared_rid,
        ),
        _s(
            group_index=0,
            index=11,
            reward=0.0,
            kind="vanilla",
            trial_idx=1,
            rollout_id=shared_rid,
        ),
    ]
    args = SimpleNamespace(advantage_estimator="grpo", reward_key=None)
    os.environ["STEP_GRPO_STD_NORMALIZATION"] = "0"
    os.environ["STEP_GRPO_FILTER"] = "1"
    _, adv = post_process_rewards(args, samples)
    assert adv[0] == 0.5
    assert adv[1] == -0.5
    filter(args, samples)
    assert not any(getattr(s, "remove_sample", False) for s in samples)


def test_vanilla_std_zero_is_audit_only_like_plain_grpo():
    samples = [
        _s(group_index=0, index=10, reward=1.0, kind="vanilla", trial_idx=0),
        _s(group_index=0, index=11, reward=1.0, kind="vanilla", trial_idx=1),
    ]
    os.environ["STEP_GRPO_FILTER"] = "1"
    filter(SimpleNamespace(), samples)
    assert not any(s.remove_sample for s in samples)
    assert [s.loss_weight for s in samples] == [0.5, 0.5]


def test_branch_std_zero_is_still_filtered():
    samples = [
        _s(
            group_index=0,
            index=20,
            reward=1.0,
            kind="branch",
            step_group_key="0:0:edit:2",
            branch_uid="0:0:edit:2:0",
        ),
        _s(
            group_index=0,
            index=21,
            reward=1.0,
            kind="branch",
            step_group_key="0:0:edit:2",
            branch_uid="0:0:edit:2:1",
        ),
    ]
    os.environ["STEP_GRPO_FILTER"] = "1"
    filter(SimpleNamespace(), samples)
    assert all(s.remove_sample for s in samples)
    assert all(s.loss_weight == 0.0 for s in samples)


def test_filter_disabled():
    samples = [
        _s(group_index=0, index=10, reward=1.0, kind="vanilla", trial_idx=0),
        _s(group_index=0, index=11, reward=1.0, kind="vanilla", trial_idx=1),
    ]
    os.environ["STEP_GRPO_FILTER"] = "0"
    filter(SimpleNamespace(), samples)
    assert not any(getattr(s, "remove_sample", False) for s in samples)
    assert [s.loss_weight for s in samples] == [0.5, 0.5]


def test_loss_weights_equalize_episodes_and_edit_groups(monkeypatch):
    """Stage-1 sums to 1; Stage-2 sums to lambda via equal edit groups."""
    monkeypatch.setenv("STEP_GRPO_FILTER", "0")
    monkeypatch.setenv("STEP_GRPO_BRANCH_LOSS_WEIGHT", "2")
    rid = 42
    samples = [
        # Two compact segments from vanilla episode v0, plus episode v1.
        _s(
            group_index=0,
            index=10,
            reward=1.0,
            kind="vanilla",
            trial_idx=0,
            rollout_id=rid,
            branch_uid="v:0:t0",
        ),
        _s(
            group_index=0,
            index=10,
            reward=0.0,
            kind="vanilla",
            trial_idx=0,
            rollout_id=rid,
            branch_uid="v:0:t0",
        ),
        _s(
            group_index=0,
            index=11,
            reward=0.0,
            kind="vanilla",
            trial_idx=1,
            rollout_id=rid,
            branch_uid="v:0:t1",
        ),
        # Edit group A has two branches.
        _s(
            group_index=0,
            index=20,
            reward=1.0,
            kind="branch",
            step_group_key="0:0:edit:2",
            rollout_id=rid,
            branch_uid="0:0:edit:2:0",
        ),
        _s(
            group_index=0,
            index=21,
            reward=0.0,
            kind="branch",
            step_group_key="0:0:edit:2",
            rollout_id=rid,
            branch_uid="0:0:edit:2:1",
        ),
        # Edit group B has one branch.
        _s(
            group_index=0,
            index=30,
            reward=1.0,
            kind="branch",
            step_group_key="0:1:edit:5",
            rollout_id=rid,
            branch_uid="0:1:edit:5:0",
        ),
    ]

    filter(SimpleNamespace(), samples)

    # Compact segments share one group and one 1/2 episode coefficient.
    assert samples[0].loss_group_id == samples[1].loss_group_id
    assert samples[0].loss_group_id != samples[2].loss_group_id
    assert [s.loss_weight for s in samples[:3]] == [0.5, 0.5, 0.5]
    # lambda=2, two active edit groups: group A gets 1 total (1/2 each),
    # group B gets 1 total (one branch).
    assert [s.loss_weight for s in samples[3:]] == [0.5, 0.5, 1.0]


def test_zero_token_vanilla_slot_keeps_fixed_group_weight(monkeypatch):
    monkeypatch.setenv("STEP_GRPO_FILTER", "0")
    samples = [
        _s(group_index=0, index=10, reward=1.0, kind="vanilla", trial_idx=0, loss_mask=[1]),
        _s(group_index=0, index=11, reward=0.0, kind="vanilla", trial_idx=1, loss_mask=[0]),
    ]
    samples[1].remove_sample = True
    filter(SimpleNamespace(), samples)
    assert samples[0].loss_weight == 0.5
    assert samples[1].loss_weight == 0.5


def test_zero_token_vanilla_slot_defines_sibling_advantage_like_plain_grpo(monkeypatch):
    args = SimpleNamespace(advantage_estimator="grpo", reward_key=None)
    samples = [
        _s(group_index=0, index=10, reward=1.0, kind="vanilla", trial_idx=0, loss_mask=[1]),
        _s(group_index=0, index=11, reward=0.0, kind="vanilla", trial_idx=1, loss_mask=[0]),
    ]
    samples[1].remove_sample = True

    monkeypatch.setenv("STEP_GRPO_STD_NORMALIZATION", "0")
    monkeypatch.setenv("STEP_GRPO_FILTER", "0")
    _, advantages = post_process_rewards(args, samples)
    # The placeholder itself has no loss, but its reward=0 remains one member
    # of the two-trial GRPO group.
    assert advantages == [0.5, -0.5]

    monkeypatch.setenv("STEP_GRPO_FILTER", "1")
    filter(args, samples)
    assert samples[0].remove_sample is False
    assert samples[1].remove_sample is True
    assert [sample.loss_weight for sample in samples] == [0.5, 0.5]


def test_declared_stage1_group_size_rejects_missing_slot(monkeypatch):
    monkeypatch.setenv("STEP_GRPO_FILTER", "0")
    samples = [
        _s(group_index=0, index=10, reward=1.0, kind="vanilla", trial_idx=0),
    ]
    samples[0].metadata["stage1_group_size"] = 2
    with pytest.raises(ValueError, match="has 1 episode slots, expected 2"):
        filter(SimpleNamespace(), samples)
