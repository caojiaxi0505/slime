from types import SimpleNamespace

import pytest

from slime.utils.loss_groups import build_loss_group_fields


def _sample(group=None, weight=None, kind=None):
    metadata = {} if kind is None else {"sample_kind": kind}
    return SimpleNamespace(loss_group_id=group, loss_weight=weight, metadata=metadata)


def test_legacy_fallback_is_identical_to_rollout_grouping():
    fields = build_loss_group_fields(
        [_sample(), _sample(), _sample()],
        [7, 7, 8],
        [[1, 1], [1], [1, 1, 1]],
    )
    assert fields == {
        "loss_group_ids": [7, 7, 8],
        "loss_group_mask_sums": [3, 3, 3],
        "loss_weights": [1.0, 1.0, 1.0],
    }


def test_independent_episode_denominators_and_weights():
    fields = build_loss_group_fields(
        [_sample("v0", 0.5, "vanilla"), _sample("v0", 0.5, "vanilla"), _sample("v1", 0.5, "branch")],
        [7, 7, 7],
        [[1, 1], [1, 0], [1, 1, 1]],
    )
    assert fields["loss_group_ids"] == ["v0", "v0", "v1"]
    assert fields["loss_group_mask_sums"] == [3, 3, 3]
    assert fields["loss_weights"] == [0.5, 0.5, 0.5]
    assert fields["loss_stage_ids"] == [0, 0, 1]


def test_compact_episode_rejects_inconsistent_weight():
    with pytest.raises(ValueError, match="inconsistent weights"):
        build_loss_group_fields(
            [_sample("v0", 0.5), _sample("v0", 1.0)],
            [7, 7],
            [[1], [1]],
        )


def test_loss_group_cannot_cross_scheduler_rollouts():
    with pytest.raises(ValueError, match="spans rollout_ids"):
        build_loss_group_fields(
            [_sample("same", 1.0), _sample("same", 1.0)],
            [7, 8],
            [[1], [1]],
        )
