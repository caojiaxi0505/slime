from slime.agent.segment_trajectory import TokenSegment, fan_out_sample_segments
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
