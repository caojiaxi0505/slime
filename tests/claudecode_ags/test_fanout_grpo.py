from types import SimpleNamespace

from slime.rollout.fanout_grpo import post_process_rewards
from slime.utils.types import Sample


def _s(group_index, index, reward):
    return Sample(group_index=group_index, index=index, reward=reward, prompt="p")


def test_sums_segments_then_centers_across_repeats():
    # One prompt, two repeats. Repeat0 has two segments 0.5+0.5=1; repeat1 has 0.
    samples = [
        _s(0, 10, 0.5),
        _s(0, 10, 0.5),
        _s(0, 11, 0.0),
    ]
    args = SimpleNamespace(grpo_std_normalization=False, advantage_estimator="grpo")
    raw, adv = post_process_rewards(args, samples)
    assert raw == [0.5, 0.5, 0.0]
    # episode rewards [1.0, 0.0] → mean 0.5 → advantages [0.5, -0.5]
    assert adv[0] == adv[1] == 0.5
    assert adv[2] == -0.5
