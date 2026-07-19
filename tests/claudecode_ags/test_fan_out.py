import pytest

from slime.agent.segment_trajectory import (
    TokenSegment,
    fan_out_sample_segments,
    mask_to_first_assistant_turn,
    merge_turns,
)
from slime.agent.trajectory import TurnRecord
from slime.utils.types import Sample


class _Tok:
    def decode(self, ids, skip_special_tokens=False):
        return "x" * len(ids)


def test_fan_out_splits_reward_and_shares_rollout_id():
    sample = Sample(index=7, group_index=3, prompt="p", metadata={})
    segs = [
        TokenSegment(
            prompt_ids=[1],
            response_ids=[2, 3],
            loss_mask=[1, 1],
            rollout_log_probs=[0.0, 0.0],
            metadata={"segment_kind": "wipe"},
        ),
        TokenSegment(
            prompt_ids=[1],
            response_ids=[4],
            loss_mask=[1],
            rollout_log_probs=[0.0],
            metadata={"segment_kind": "final"},
        ),
    ]
    out = fan_out_sample_segments(sample, segs, reward=1.0, tokenizer=_Tok())
    assert len(out) == 2
    assert out[0].reward == 0.5
    assert out[1].reward == 0.5
    assert out[0].rollout_id == 7
    assert out[1].rollout_id == 7
    assert out[0].metadata["num_segments"] == 2


def test_first_assistant_turn_scope_uses_recorded_turn_boundaries():
    segment = merge_turns(
        [
            TurnRecord(
                prompt_ids=[10],
                output_ids=[11, 12],
                output_log_probs=[-0.1, -0.2],
                finish_reason="tool_use",
            ),
            TurnRecord(
                prompt_ids=[10, 11, 12, 20],
                output_ids=[13, 14],
                output_log_probs=[-0.3, -0.4],
                finish_reason="end_turn",
            ),
        ],
        metadata={"segment_kind": "final"},
    )
    assert segment is not None
    assert segment.response_ids == [11, 12, 20, 13, 14]
    assert segment.loss_mask == [1, 1, 0, 1, 1]
    assert segment.metadata["assistant_output_spans"] == [[0, 2], [3, 5]]

    scoped, stats = mask_to_first_assistant_turn([segment])
    assert scoped[0].response_ids == segment.response_ids
    assert scoped[0].loss_mask == [1, 1, 0, 0, 0]
    assert scoped[0].rollout_log_probs == [-0.1, -0.2, 0.0, 0.0, 0.0]
    assert stats == {
        "assistant_turns": 2,
        "total_trainable_tokens": 4,
        "kept_trainable_tokens": 2,
        "masked_trainable_tokens": 2,
        "first_turn_segment_idx": 0,
    }


def test_first_assistant_turn_scope_fails_without_boundaries():
    segment = TokenSegment(
        prompt_ids=[1],
        response_ids=[2],
        loss_mask=[1],
        rollout_log_probs=[-0.1],
    )
    with pytest.raises(ValueError, match="missing assistant_output_spans"):
        mask_to_first_assistant_turn([segment])
