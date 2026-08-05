from examples.claudecode_ags.wandb_metrics import (
    _reward_source_metrics,
    _timing_metrics,
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


def test_tool_loop_metrics_are_logged_per_stage_and_threshold() -> None:
    stage1_runs = [2, 3, 5]
    samples = []
    for index, run in enumerate(stage1_runs):
        sample = _sample(
            instance_id=f"stage1-{index}",
            group_index=0,
            index=index,
            segment_idx=0,
            num_segments=1,
            segment_kind="final",
            reward=0.0,
            response_length=1,
            tokens_len=2,
        )
        sample.metadata.update(
            {
                "tool_loop_detection_enabled": True,
                "consecutive_tool_signature_max": run,
            }
        )
        samples.append(sample)

    branch = _sample(
        instance_id="stage2-0",
        group_index=0,
        index=9,
        segment_idx=0,
        num_segments=1,
        segment_kind="final",
        reward=-1.0,
        response_length=1,
        tokens_len=2,
    )
    branch.metadata.update(
        {
            "sample_kind": "branch",
            "branch_uid": "stage2-0",
            "tool_loop_detection_enabled": True,
            "consecutive_tool_signature_max": 4,
        }
    )
    samples.append(branch)

    metrics = _trajectory_metrics(samples)
    assert metrics["behavior/tool_loop/n_checked_episodes"] == 3.0
    assert metrics["behavior/tool_loop/ge_3_count"] == 2.0
    assert metrics["behavior/tool_loop/ge_3_rate"] == 2 / 3
    assert metrics["behavior/tool_loop/ge_4_count"] == 1.0
    assert metrics["behavior/tool_loop/ge_5_count"] == 1.0
    assert metrics["behavior/stage-2/tool_loop/n_checked_episodes"] == 1.0
    assert metrics["behavior/stage-2/tool_loop/ge_4_rate"] == 1.0
    assert metrics["behavior/stage-2/tool_loop/ge_5_rate"] == 0.0


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
    van.metadata.update(
        {
            "adapter_turn_count": 4,
            "adapter_failed_request_count": 1,
            "adapter_prepare_tokenize_ms_mean": 12.0,
            "adapter_cpu_queue_ms_mean": 3.0,
        }
    )
    br_ok.metadata.update(
        {
            "adapter_turn_count": 2,
            "adapter_failed_request_count": 0,
            "adapter_prepare_tokenize_ms_mean": 20.0,
        }
    )
    br_bad.metadata.update(
        {
            "adapter_turn_count": 2,
            "adapter_failed_request_count": 1,
            "adapter_prepare_tokenize_ms_mean": 30.0,
        }
    )

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
    assert tm["perf/adapter/prepare_tokenize_ms/mean"] == 12.0
    assert tm["perf/adapter/cpu_queue_ms/mean"] == 3.0
    assert tm["perf/adapter/request_failure_rate"] == 0.25
    assert tm["perf/stage-2/adapter/prepare_tokenize_ms/mean"] == 25.0
    assert tm["perf/stage-2/adapter/request_failure_rate"] == 0.25


def test_branch_uid_prevents_cross_edit_episode_collision(monkeypatch):
    """Same legacy index at two edit points must remain two W&B episodes."""
    branches = []
    for edit_step, solved in ((2, True), (5, False)):
        s = _sample(
            instance_id="a",
            group_index=0,
            index=100,
            segment_idx=0,
            num_segments=1,
            segment_kind="final",
            reward=float(solved),
            response_length=1,
            tokens_len=2,
            grading_solved=solved,
        )
        s.rollout_id = 7
        s.metadata.update(
            {
                "sample_kind": "branch",
                "source_trial_idx": 1,
                "edit_step_i": edit_step,
                "branch_idx": 0,
                "step_group_key": f"0:1:edit:{edit_step}",
                "branch_uid": f"0:1:edit:{edit_step}:0",
            }
        )
        branches.append(s)

    m = _reward_source_metrics(branches)
    assert m["outcome/stage-2/n_episodes"] == 2.0
    assert m["outcome/stage-2/resolved_rate"] == 0.5


def test_hybrid_objective_and_queue_metrics(monkeypatch):
    monkeypatch.setenv("STEP_GRPO_BRANCH_LOSS_WEIGHT", "2")
    vanilla = _sample(
        instance_id="a",
        group_index=0,
        index=1,
        segment_idx=0,
        num_segments=1,
        segment_kind="final",
        reward=1.0,
        response_length=2,
        tokens_len=3,
    )
    vanilla.metadata.update(
        {
            "sample_kind": "vanilla",
            "branch_uid": "v:0:t0",
            "hybrid_stage1_wall_sec": 1.0,
            "transcript_valid": True,
            "agent_queue_wait_sec": 3.0,
            "hybrid_num_stage1_planned_trials": 8,
            "hybrid_num_stage1_aborted_placeholders": 2,
            "hybrid_num_stage1_cancelled_placeholders": 1,
            "hybrid_num_patch_candidates": 2,
            "hybrid_num_selected_edits": 1,
            "hybrid_num_branch_tasks": 8,
            "hybrid_num_stage2_samples_before_length_filter": 10,
            "hybrid_num_stage2_samples_after_length_filter": 8,
            "hybrid_num_stage2_samples_dropped_over_context": 2,
            "hybrid_stage2_context_limit_tokens": 131072,
            "hybrid_stage2_max_total_tokens": 140000,
            "hybrid_stage2_max_response_tokens": 20000,
            "hybrid_stage2_max_loss_tokens": 12000,
            "hybrid_num_dropped_branches": 2,
            "hybrid_num_dropped_timeout": 0,
            "hybrid_num_dropped_cancelled": 1,
            "hybrid_num_dropped_resume_tool_echo": 1,
            "hybrid_num_dropped_resume_missing_result": 0,
            "hybrid_num_dropped_resume_no_pending": 1,
            "hybrid_num_dropped_resume_tool_schema": 0,
            "hybrid_num_dropped_resume_subagent": 0,
            "hybrid_num_dropped_resume_other": 0,
            "hybrid_num_dropped_workspace_rebuild": 0,
            "hybrid_num_dropped_other": 0,
        }
    )
    vanilla.loss_mask = [1, 1]
    vanilla.loss_weight = 1.0
    branch = _sample(
        instance_id="a",
        group_index=0,
        index=2,
        segment_idx=0,
        num_segments=1,
        segment_kind="final",
        reward=0.0,
        response_length=3,
        tokens_len=4,
    )
    branch.metadata.update(
        {
            "sample_kind": "branch",
            "branch_uid": "0:0:edit:2:0",
            "step_group_key": "0:0:edit:2",
            "agent_queue_wait_sec": 5.0,
            "prefix_reseed_verified": True,
            "prompt_exact": True,
            "prompt_sha256": "checkpoint-hash",
            "tool_use_echo_mismatch_count": 2,
            "tool_use_echo_missing_count": 1,
            "tool_use_echo_payload_mismatch_count": 1,
            "runtime_tool_schema_mismatch_count": 3,
            "generated_runtime_tool_unavailable_count": 2,
            "generated_runtime_tool_input_invalid_count": 6,
            "max_tokens_continuation_count": 4,
            "post_end_turn_ack_count": 1,
            "resume_request_replay_count": 5,
            "stage2_loss_scope": "first_turn",
            "stage2_pre_scope_trainable_tokens": 7,
            "stage2_kept_trainable_tokens": 2,
            "stage2_masked_later_trainable_tokens": 5,
        }
    )
    branch.loss_mask = [1, 0, 1]
    branch.loss_weight = 2.0

    m = _timing_metrics([vanilla, branch], rollout_time=10.0)
    assert m["perf/agent_queue_wait/mean"] == 3.0
    assert m["perf/stage-2/agent_queue_wait/mean"] == 5.0
    assert m["perf/step_grpo/n_stage1_active_episodes"] == 1.0
    assert m["perf/step_grpo/n_stage2_active_episodes"] == 1.0
    assert m["perf/step_grpo/n_active_edit_groups"] == 1.0
    assert m["perf/step_grpo/stage1_active_tokens"] == 2.0
    assert m["perf/step_grpo/stage2_active_tokens"] == 2.0
    assert m["perf/step_grpo/stage1_nominal_loss_weight"] == 1.0
    assert m["perf/step_grpo/stage2_nominal_loss_weight"] == 2.0
    assert m["perf/step_grpo/n_stage2_loss_scope_audited"] == 1.0
    assert m["perf/step_grpo/n_stage2_first_turn_scoped"] == 1.0
    assert m["perf/step_grpo/stage2_pre_scope_trainable_tokens"] == 7.0
    assert m["perf/step_grpo/stage2_kept_trainable_tokens"] == 2.0
    assert m["perf/step_grpo/stage2_masked_later_trainable_tokens"] == 5.0
    assert m["perf/step_grpo/stage2_kept_token_rate"] == 2.0 / 7.0
    assert m["perf/step_grpo/branch_loss_weight"] == 2.0
    assert m["perf/step_grpo/n_stage1_planned_trials"] == 8.0
    assert m["perf/step_grpo/n_stage1_aborted_placeholders"] == 2.0
    assert m["perf/step_grpo/n_stage1_cancelled_placeholders"] == 1.0
    assert m["perf/step_grpo/stage1_aborted_placeholder_rate"] == 0.25
    assert m["perf/step_grpo/stage1_cancelled_placeholder_rate"] == 0.125
    assert m["perf/step_grpo/n_patch_candidates"] == 2.0
    assert m["perf/step_grpo/n_branch_tasks"] == 8.0
    assert m["perf/step_grpo/n_stage2_samples_before_length_filter"] == 10.0
    assert m["perf/step_grpo/n_stage2_samples_after_length_filter"] == 8.0
    assert m["perf/step_grpo/n_stage2_samples_dropped_over_context"] == 2.0
    assert m["perf/step_grpo/stage2_over_context_sample_rate"] == 0.2
    assert m["perf/step_grpo/stage2_context_limit_tokens"] == 131072.0
    assert m["perf/step_grpo/stage2_max_total_tokens_before_length_filter"] == 140000.0
    assert m["perf/step_grpo/stage2_max_response_tokens_before_length_filter"] == 20000.0
    assert m["perf/step_grpo/stage2_max_loss_tokens_before_length_filter"] == 12000.0
    assert m["perf/step_grpo/n_dropped_branches"] == 2.0
    assert m["perf/step_grpo/n_completed_branches"] == 6.0
    assert m["perf/step_grpo/branch_completion_rate"] == 0.75
    assert m["perf/step_grpo/branch_drop_rate"] == 0.25
    assert m["perf/step_grpo/n_dropped_cancelled"] == 1.0
    assert m["perf/step_grpo/n_dropped_resume_tool_echo"] == 1.0
    assert m["perf/step_grpo/n_dropped_resume_no_pending"] == 1.0
    assert m["resume/prompt_exact_rate"] == 1.0
    assert m["resume/checkpoint_hash_consistency_rate"] == 1.0
    assert m["resume/tool_use_echo_mismatch_count"] == 2.0
    assert m["resume/tool_use_echo_mismatch_branch_rate"] == 1.0
    assert m["resume/tool_use_echo_missing_count"] == 1.0
    assert m["resume/tool_use_echo_payload_mismatch_count"] == 1.0
    assert m["resume/runtime_tool_schema_mismatch_count"] == 3.0
    assert m["resume/runtime_tool_schema_mismatch_branch_rate"] == 1.0
    assert m["resume/generated_runtime_tool_unavailable_count"] == 2.0
    assert m["resume/generated_runtime_tool_unavailable_branch_rate"] == 1.0
    assert m["resume/generated_runtime_tool_input_invalid_count"] == 6.0
    assert m["resume/generated_runtime_tool_input_invalid_branch_rate"] == 1.0
    assert m["resume/max_tokens_continuation_count"] == 4.0
    assert m["resume/post_end_turn_ack_count"] == 1.0
    assert m["resume/request_replay_count"] == 5.0
    assert m["resume/request_replay_branch_rate"] == 1.0
