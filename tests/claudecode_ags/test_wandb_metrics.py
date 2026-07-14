from examples.claudecode_ags.wandb_metrics import (
    _reward_source_metrics,
    _trajectory_metrics,
)
from slime.utils.types import Sample


def _sample(
    *,
    instance_id: str,
    group_index: int,
    index: int,
    segment_idx: int,
    num_segments: int,
    segment_kind: str,
    reward: float,
    response_length: int,
    tokens_len: int,
    grading_solved: bool = False,
    abort_reason: str | None = None,
    agent_exit_code: int = 0,
    applied_cleanly: bool = True,
    status: Sample.Status = Sample.Status.COMPLETED,
) -> Sample:
    s = Sample(
        group_index=group_index,
        index=index,
        reward=reward,
        prompt=["p"],
        response_length=response_length,
        tokens=list(range(tokens_len)),
        status=status,
        metadata={
            "instance_id": instance_id,
            "segment_idx": segment_idx,
            "num_segments": num_segments,
            "segment_kind": segment_kind,
            "grading_solved": grading_solved,
            "agent_exit_code": agent_exit_code,
            "applied_cleanly": applied_cleanly,
            "test_f2p_ratio": 1.0 if grading_solved else 0.0,
            "test_p2p_ratio": 1.0,
            "test_f2p_passed": 1 if grading_solved else 0,
            "test_f2p_total": 1,
            "test_p2p_passed": 1,
            "test_p2p_total": 1,
        },
    )
    if abort_reason is not None:
        s.metadata["abort_reason"] = abort_reason
    return s


def test_trajectory_metrics_episode_aggregation():
    # Episode A: 2 segments, solved, wipe+final
    a0 = _sample(
        instance_id="a",
        group_index=0,
        index=10,
        segment_idx=0,
        num_segments=2,
        segment_kind="wipe",
        reward=0.5,
        response_length=100,
        tokens_len=1100,
        grading_solved=True,
    )
    a1 = _sample(
        instance_id="a",
        group_index=0,
        index=10,
        segment_idx=1,
        num_segments=2,
        segment_kind="final",
        reward=0.5,
        response_length=200,
        tokens_len=1300,
        grading_solved=True,
    )
    # Episode B: abort
    b0 = _sample(
        instance_id="b",
        group_index=0,
        index=11,
        segment_idx=0,
        num_segments=1,
        segment_kind="final",
        reward=0.0,
        response_length=10,
        tokens_len=50,
        abort_reason="adapter_session_empty",
        agent_exit_code=-1,
        applied_cleanly=False,
        status=Sample.Status.ABORTED,
    )

    m = _trajectory_metrics([a0, a1, b0])
    assert m["traj/num_segments/mean"] == 1.5
    assert m["traj/num_segments/max"] == 2.0
    assert m["traj/num_segments/min"] == 1.0
    assert m["traj/segment_kind/wipe_rate"] == 1 / 3
    assert m["traj/segment_kind/final_rate"] == 2 / 3
    assert m["traj/segment_kind/subagent_rate"] == 0.0
    assert m["outcome/abort_rate"] == 0.5
    assert m["outcome/abort_reason/adapter_session_empty"] == 0.5
    assert m["outcome/agent_exit_nonzero_rate"] == 0.5
    assert m["outcome/applied_cleanly_rate"] == 0.5
    assert m["rollout/episode_reward/mean"] == 0.5  # (1.0 + 0.0) / 2
    assert m["traj/episode_response_len/mean"] == 155.0  # (300 + 10) / 2
    # a0 prompt=1000, a1 prompt=1100 → episode max prompt 1100
    assert m["rollout/prompt_len/max"] == 1100.0
    assert m["rollout/total_len/max"] == 1300.0


def test_reward_source_still_episode_deduped():
    a0 = _sample(
        instance_id="a",
        group_index=0,
        index=1,
        segment_idx=0,
        num_segments=2,
        segment_kind="wipe",
        reward=0.5,
        response_length=1,
        tokens_len=2,
        grading_solved=True,
    )
    a1 = _sample(
        instance_id="a",
        group_index=0,
        index=1,
        segment_idx=1,
        num_segments=2,
        segment_kind="final",
        reward=0.5,
        response_length=1,
        tokens_len=2,
        grading_solved=True,
    )
    m = _reward_source_metrics([a0, a1])
    assert m["outcome/n_episodes"] == 1.0
    assert m["outcome/resolved_rate"] == 1.0
    assert "outcome/stage-2/resolved_rate" not in m


def test_outcome_resolved_rate_stage1_only():
    """Top-level outcome/* = Stage-1; branch under outcome/stage-2/*."""
    van = _sample(
        instance_id="a",
        group_index=0,
        index=1,
        segment_idx=0,
        num_segments=1,
        segment_kind="final",
        reward=1.0,
        response_length=1,
        tokens_len=2,
        grading_solved=True,
    )
    van.metadata["sample_kind"] = "vanilla"
    van.metadata["hybrid_stage1_wall_sec"] = 10.0
    van.metadata["hybrid_stage2_wall_sec"] = 20.0
    van.metadata["hybrid_total_wall_sec"] = 30.0
    van.metadata["agent_elapsed_sec"] = 5.0
    van.metadata["eval_elapsed_sec"] = 1.0
    van.metadata["total_elapsed_sec"] = 6.0
    br_ok = _sample(
        instance_id="a",
        group_index=0,
        index=2,
        segment_idx=0,
        num_segments=1,
        segment_kind="final",
        reward=1.0,
        response_length=1,
        tokens_len=2,
        grading_solved=True,
    )
    br_ok.metadata["sample_kind"] = "branch"
    br_ok.metadata["agent_elapsed_sec"] = 3.0
    br_ok.metadata["eval_elapsed_sec"] = 0.5
    br_ok.metadata["total_elapsed_sec"] = 3.5
    br_bad = _sample(
        instance_id="a",
        group_index=0,
        index=3,
        segment_idx=0,
        num_segments=1,
        segment_kind="final",
        reward=0.0,
        response_length=1,
        tokens_len=2,
        grading_solved=False,
        agent_exit_code=1,
        applied_cleanly=False,
    )
    br_bad.metadata["sample_kind"] = "branch"
    br_bad.metadata["agent_elapsed_sec"] = 2.0

    m = _reward_source_metrics([van, br_ok, br_bad])
    assert m["outcome/n_episodes"] == 1.0
    assert m["outcome/resolved_rate"] == 1.0
    assert m["outcome/stage-2/n_episodes"] == 2.0
    assert m["outcome/stage-2/resolved_rate"] == 0.5

    t = _trajectory_metrics([van, br_ok, br_bad])
    assert t["outcome/agent_exit_nonzero_rate"] == 0.0
    assert t["outcome/applied_cleanly_rate"] == 1.0
    assert t["outcome/stage-2/agent_exit_nonzero_rate"] == 0.5
    assert t["outcome/stage-2/applied_cleanly_rate"] == 0.5
    # Stage-1 traj counts only vanilla samples
    assert t["traj/n_samples"] == 1.0
    assert t["traj/stage-2/n_samples"] == 2.0

    from examples.claudecode_ags.wandb_metrics import _timing_metrics

    tm = _timing_metrics([van, br_ok, br_bad], rollout_time=100.0)
    assert tm["perf/rollout_time"] == 100.0
    assert tm["perf/step_grpo/rollout_time"] == 100.0
    assert tm["perf/agent_time/mean"] == 5.0
    assert tm["perf/stage-2/agent_time/mean"] == 2.5  # (3+2)/2
    assert tm["perf/step_grpo/stage1_wall/mean"] == 10.0
    assert tm["perf/step_grpo/stage2_wall/mean"] == 20.0
    assert tm["perf/step_grpo/n_stage1_episodes"] == 1.0
    assert tm["perf/step_grpo/n_stage2_episodes"] == 2.0
