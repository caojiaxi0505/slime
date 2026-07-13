import os
from types import SimpleNamespace

from examples.claudecode_ags.step_reconstruct.step_grpo_advantage import filter, post_process_rewards
from slime.utils.types import Sample


def _s(*, group_index, index, reward, kind, step_group_key=None, loss_mask=None, trial_idx=0):
    md = {"sample_kind": kind, "trial_idx": trial_idx}
    if step_group_key is not None:
        md["step_group_key"] = step_group_key
    return Sample(
        group_index=group_index,
        index=index,
        rollout_id=index,
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


def test_filter_std_zero():
    samples = [
        _s(group_index=0, index=10, reward=1.0, kind="vanilla", trial_idx=0),
        _s(group_index=0, index=11, reward=1.0, kind="vanilla", trial_idx=1),
    ]
    os.environ["STEP_GRPO_FILTER"] = "1"
    filter(SimpleNamespace(), samples)
    assert all(s.remove_sample for s in samples)


def test_filter_disabled():
    samples = [
        _s(group_index=0, index=10, reward=1.0, kind="vanilla", trial_idx=0),
        _s(group_index=0, index=11, reward=1.0, kind="vanilla", trial_idx=1),
    ]
    os.environ["STEP_GRPO_FILTER"] = "0"
    filter(SimpleNamespace(), samples)
    assert not any(getattr(s, "remove_sample", False) for s in samples)
