import asyncio
import gzip
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from examples.claudecode_ags.step_reconstruct.hybrid_generate import (
    VanillaTrialResult,
    _branch_drop_bucket,
    collect_patch_candidates,
    hybrid_generate,
    pre_checkpoint_snapshot_index,
    select_branch_turns,
)
from examples.claudecode_ags.step_reconstruct.session_capture import SessionBundle, steps_from_diff_files
from slime.utils.types import Sample


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RuntimeError("expected one echo for tool_use toolu_x, got 0"), "resume_tool_echo"),
        (RuntimeError("expected one result for tool_use toolu_x, got 0"), "resume_missing_result"),
        (
            RuntimeError("received resumed request without pending tool calls after stop_reason=unknown"),
            "resume_no_pending",
        ),
        (RuntimeError("tool schema differs from the Stage-1 checkpoint"), "resume_tool_schema"),
        (
            RuntimeError("tool input incompatible with Claude Code runtime tool schema"),
            "resume_tool_schema",
        ),
        (
            RuntimeError("Task/Agent subagent dispatch is not supported in a token-exact resumed branch"),
            "resume_subagent",
        ),
        (RuntimeError("token_exact_resume_not_verified"), "resume_other"),
        (
            RuntimeError("teacher_context_integrity:token_prefix_mismatch"),
            "context_integrity",
        ),
        (RuntimeError("rebuild apply/verify failed"), "workspace_rebuild"),
        (TimeoutError(), "timeout_other"),
        (asyncio.CancelledError(), "cancelled"),
        (RuntimeError("boom"), "other"),
    ],
)
def test_branch_drop_bucket(error, expected):
    assert _branch_drop_bucket(error) == expected


def _bundle(tmp_path, diffs):
    d = tmp_path / "b"
    d.mkdir(parents=True, exist_ok=True)
    steps = steps_from_diff_files(str(d), diffs)
    tool_ids = [f"toolu_{i}" for i in range(len(steps))]
    metadata_payload = '{"version":1,"records":[],"unsupported_paths":[]}'
    snapshots = d / ".cagent_snapshots"
    (snapshots / "initial.metadata.json").write_text(metadata_payload)
    for i, (step, tool_id) in enumerate(zip(steps, tool_ids, strict=True)):
        step.tool_use_id = tool_id
        metadata_rel = f".cagent_snapshots/step_{i:04d}.metadata.json"
        (d / metadata_rel).write_text(metadata_payload)
        step.metadata_file = metadata_rel
    b = SessionBundle(
        instance_id="x",
        session_id="s",
        cc_session_id="",
        task_metadata={"image": "img", "workdir": "/testbed"},
        steps=steps,
        prompt_checkpoints_valid=True,
        native_session_valid=True,
        workspace_metadata_valid=True,
        dir=str(d),
    )
    events = []
    for tool_id in tool_ids:
        events.extend(
            [
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": tool_id, "name": "Read", "input": {}}],
                    },
                },
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": "ok"}],
                    },
                },
            ]
        )
    (d / "transcript.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")
    with gzip.open(d / "prompt_checkpoints.json.gz", "wt", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "checkpoint_id": f"main-{i}",
                    "generated_tool_use_ids": [tool_id],
                    "generated_tool_use_names": {tool_id: "Read"},
                    "chain_kind": "main",
                    "request_kind": "new" if i == 0 else "append",
                }
                for i, tool_id in enumerate(tool_ids)
            ],
            f,
        )
    b.save(str(d))
    return SessionBundle.load(str(d))


def _sample(reward=0.0, index=0):
    return Sample(prompt="p", index=index, group_index=0, reward=reward, metadata={}, loss_mask=[1])


def test_collect_and_select(tmp_path):
    b0 = _bundle(tmp_path / "t0", ["", "diff --git a/a b/a\n+1\n"])
    b1 = _bundle(tmp_path / "t1", ["", "diff --git a/a b/a\n+2\n"])
    trials = [
        VanillaTrialResult(0, b0, [_sample()], False, [[], [-2.0, -2.0]]),
        VanillaTrialResult(1, b1, [_sample()], False, [[], [-5.0, -5.0]]),
        VanillaTrialResult(2, b0, [_sample(1.0)], True, [[], [-9.0]]),
    ]
    cands = collect_patch_candidates(trials)
    assert len(cands) == 2
    sel = select_branch_turns(trials, k=1)
    assert sel[0].source_trial_idx == 1
    assert sel[0].edit_step_i == 1
    assert sel[0].branch_step_t == 0


def test_collect_candidates_without_patch_only_keeps_read_turns(tmp_path):
    # Step 0 is a read (diff unchanged); teacher relabeling can resume it too.
    b0 = _bundle(tmp_path / "t0", ["", "diff --git a/a b/a\n+1\n"])
    trials = [VanillaTrialResult(0, b0, [_sample()], False, [[-1.0], [-2.0, -2.0]])]
    assert [c.edit_step_i for c in collect_patch_candidates(trials)] == [1]
    all_turns = collect_patch_candidates(trials, patch_only=False)
    assert [(c.edit_step_i, c.branch_step_t) for c in all_turns] == [(0, -1), (1, 0)]


def test_checkpoint_group_sets_pre_turn_without_transcript_order():
    # The two tools belong to one model turn but completed in reverse order.
    assert pre_checkpoint_snapshot_index(
        ["toolu_previous", "toolu_b", "toolu_a"],
        ["toolu_a", "toolu_b"],
        "toolu_a",
    ) == 0


def test_checkpoint_group_rejects_interleaved_unknown_snapshot():
    with pytest.raises(ValueError, match="not_contiguous"):
        pre_checkpoint_snapshot_index(
            ["toolu_a", "toolu_nested", "toolu_b"],
            ["toolu_a", "toolu_b"],
            "toolu_a",
        )


def test_invalid_transcript_is_audit_only_for_token_exact_candidate(tmp_path):
    bundle = _bundle(tmp_path, ["", "diff --git a/a b/a\n+1\n"])
    bundle.transcript_valid = False
    bundle.transcript_error = "split parallel stream rows"
    bundle.save(bundle.dir)
    candidates = collect_patch_candidates(
        [VanillaTrialResult(0, bundle, [_sample()], False, [[], [-3.0]])]
    )
    assert len(candidates) == 1
    assert candidates[0].branch_step_t == 0


def test_hybrid_all_solved_no_branch(tmp_path):
    os.environ["STEP_GRPO_HYBRID_K"] = "2"
    b = _bundle(tmp_path, ["", "diff --git a/a b/a\n+1\n"])

    async def vanilla_runner(**kwargs):
        i = kwargs["trial_idx"]
        s = _sample(1.0, index=i)
        s.metadata = {"sample_kind": "vanilla", "trial_idx": i}
        return b, [s], True, [[], [-1.0]]

    async def branch_runner(**kwargs):
        raise AssertionError("should not branch")

    out = asyncio.run(
        hybrid_generate(
            SimpleNamespace(),
            _sample(),
            {},
            evaluation=False,
            vanilla_runner=vanilla_runner,
            branch_runner=branch_runner,
        )
    )
    assert len(out) == 2
    assert all(s.metadata["sample_kind"] == "vanilla" for s in out)
    assert len({s.loss_group_id for s in out}) == 2
    assert len({s.rollout_id for s in out}) == 1


def test_hybrid_branches_k_times_k(tmp_path):
    os.environ["STEP_GRPO_HYBRID_K"] = "2"
    b0 = _bundle(tmp_path / "a", ["", "diff --git a/a b/a\n+1\n"])
    b1 = _bundle(tmp_path / "b", ["", "diff --git a/a b/a\n+2\n"])

    async def vanilla_runner(**kwargs):
        i = kwargs["trial_idx"]
        bundle = b0 if i == 0 else b1
        lps = [[], [-1.0]] if i == 0 else [[], [-5.0]]
        s = _sample(0.0, index=i)
        return bundle, [s], False, lps

    calls = []

    async def branch_runner(**kwargs):
        calls.append(
            (
                kwargs["source_trial_idx"],
                kwargs["edit_step_i"],
                kwargs["branch_step_t"],
                kwargs["branch_idx"],
            )
        )
        s = _sample(0.0, index=100 + len(calls))
        s.metadata = {
            "sample_kind": "branch",
            "step_group_key": f"0:{kwargs['source_trial_idx']}:edit:{kwargs['edit_step_i']}",
            "source_trial_idx": kwargs["source_trial_idx"],
            "edit_step_i": kwargs["edit_step_i"],
            "branch_step_t": kwargs["branch_step_t"],
            "branch_idx": kwargs["branch_idx"],
            "edit_ppl": kwargs["edit_ppl"],
        }
        s.loss_mask = [1, 1, 1]
        return [s]

    out = asyncio.run(
        hybrid_generate(
            SimpleNamespace(),
            _sample(),
            {},
            vanilla_runner=vanilla_runner,
            branch_runner=branch_runner,
        )
    )
    assert len([s for s in out if s.metadata["sample_kind"] == "vanilla"]) == 2
    branches = [s for s in out if s.metadata["sample_kind"] == "branch"]
    assert len(branches) == 4
    assert len(calls) == 4
    # Six independent episodes, still one outer scheduling rollout.
    assert len({s.loss_group_id for s in out}) == 6
    assert len({s.rollout_id for s in out}) == 1


def test_hybrid_filters_over_context_stage2_samples(tmp_path):
    os.environ["STEP_GRPO_HYBRID_K"] = "2"
    b0 = _bundle(tmp_path / "a", ["", "diff --git a/a b/a\n+1\n"])
    b1 = _bundle(tmp_path / "b", ["", "diff --git a/a b/a\n+2\n"])

    async def vanilla_runner(**kwargs):
        i = kwargs["trial_idx"]
        bundle = b0 if i == 0 else b1
        lps = [[], [-1.0]] if i == 0 else [[], [-5.0]]
        s = _sample(0.0, index=i)
        return bundle, [s], False, lps

    calls = []

    async def branch_runner(**kwargs):
        calls.append(kwargs["branch_idx"])
        s = _sample(0.0, index=100 + len(calls))
        s.metadata = {
            "sample_kind": "branch",
            "step_group_key": f"0:{kwargs['source_trial_idx']}:edit:{kwargs['edit_step_i']}",
            "source_trial_idx": kwargs["source_trial_idx"],
            "edit_step_i": kwargs["edit_step_i"],
            "branch_step_t": kwargs["branch_step_t"],
            "branch_idx": kwargs["branch_idx"],
            "edit_ppl": kwargs["edit_ppl"],
        }
        if len(calls) % 2:
            s.tokens = list(range(12))
            s.response_length = 6
            s.loss_mask = [1, 1, 0, 0, 1, 1]
        else:
            s.tokens = list(range(8))
            s.response_length = 4
            s.loss_mask = [1, 0, 1, 0]
        return [s]

    out = asyncio.run(
        hybrid_generate(
            SimpleNamespace(rollout_max_context_len=10),
            _sample(),
            {},
            vanilla_runner=vanilla_runner,
            branch_runner=branch_runner,
        )
    )
    branches = [s for s in out if s.metadata["sample_kind"] == "branch"]
    assert len(branches) == 2
    assert all(len(s.tokens) <= 10 for s in branches)
    assert len(calls) == 4
    assert all(s.metadata["hybrid_num_stage2_samples_before_length_filter"] == 4 for s in out)
    assert all(s.metadata["hybrid_num_stage2_samples_after_length_filter"] == 2 for s in out)
    assert all(s.metadata["hybrid_num_stage2_samples_dropped_over_context"] == 2 for s in out)
    assert all(s.metadata["hybrid_stage2_context_limit_tokens"] == 10 for s in out)
    assert all(s.metadata["hybrid_stage2_max_total_tokens"] == 12 for s in out)
    assert all(s.metadata["hybrid_stage2_max_response_tokens"] == 6 for s in out)
    assert all(s.metadata["hybrid_stage2_max_loss_tokens"] == 4 for s in out)


def test_hybrid_partial_trial_failure_keeps_grpo_slot(tmp_path):
    os.environ["STEP_GRPO_HYBRID_K"] = "2"
    bundle = _bundle(tmp_path, ["", "diff --git a/a b/a\n+1\n"])

    async def vanilla_runner(**kwargs):
        i = kwargs["trial_idx"]
        if i == 0:
            raise ConnectionError("ags unavailable")
        return bundle, [_sample(1.0, index=i)], True, [[], [-1.0]]

    out = asyncio.run(
        hybrid_generate(
            SimpleNamespace(),
            _sample(index=7),
            {},
            vanilla_runner=vanilla_runner,
            branch_runner=AsyncMock(),
        )
    )

    assert len(out) == 2
    by_trial = {s.metadata["trial_idx"]: s for s in out}
    aborted = by_trial[0]
    assert aborted.status == Sample.Status.ABORTED
    assert aborted.remove_sample is True
    assert aborted.reward == 0.0
    assert aborted.loss_mask == [0]
    assert aborted.index == 7 * 4096
    assert aborted.metadata["branch_uid"] == "v:0:t0"
    assert aborted.metadata["stage1_group_size"] == 2
    assert by_trial[1].metadata["stage1_group_size"] == 2
    assert len({s.loss_group_id for s in out}) == 2
    assert all(s.metadata["hybrid_num_stage1_planned_trials"] == 2 for s in out)
    assert all(s.metadata["hybrid_num_stage1_aborted_placeholders"] == 1 for s in out)


def test_hybrid_cancelled_trial_keeps_grpo_slot(tmp_path):
    os.environ["STEP_GRPO_HYBRID_K"] = "2"
    bundle = _bundle(tmp_path, ["", "diff --git a/a b/a\n+1\n"])

    async def vanilla_runner(**kwargs):
        i = kwargs["trial_idx"]
        if i == 0:
            raise asyncio.CancelledError()
        return bundle, [_sample(1.0, index=i)], True, [[], [-1.0]]

    out = asyncio.run(
        hybrid_generate(
            SimpleNamespace(),
            _sample(index=7),
            {},
            vanilla_runner=vanilla_runner,
            branch_runner=AsyncMock(),
        )
    )

    assert len(out) == 2
    by_trial = {s.metadata["trial_idx"]: s for s in out}
    aborted = by_trial[0]
    assert aborted.status == Sample.Status.ABORTED
    assert aborted.remove_sample is True
    assert aborted.reward == 0.0
    assert aborted.loss_mask == [0]
    assert aborted.metadata["abort_reason"] == "vanilla_trial_exception:CancelledError"
    assert aborted.metadata["branch_uid"] == "v:0:t0"
    assert all(s.metadata["hybrid_num_stage1_aborted_placeholders"] == 1 for s in out)
    assert all(s.metadata["hybrid_num_stage1_cancelled_placeholders"] == 1 for s in out)


def test_hybrid_all_trials_fail_returns_all_abort_slots():
    os.environ["STEP_GRPO_HYBRID_K"] = "2"

    async def vanilla_runner(**kwargs):
        raise RuntimeError("ags down")

    out = asyncio.run(
        hybrid_generate(
            SimpleNamespace(),
            _sample(),
            {},
            vanilla_runner=vanilla_runner,
            branch_runner=AsyncMock(),
        )
    )
    assert len(out) == 2
    assert {s.metadata["trial_idx"] for s in out} == {0, 1}
    assert len({s.loss_group_id for s in out}) == 2
    assert all(s.status == Sample.Status.ABORTED for s in out)
    assert all(s.remove_sample is True for s in out)
    assert all(s.tokens == [0, 0] for s in out)
    assert all(s.response_length == 1 for s in out)
    assert all(s.loss_mask == [0] for s in out)
    assert all(s.metadata["abort_reason"] == "vanilla_trial_exception:RuntimeError" for s in out)
    assert all(s.metadata["stage1_group_size"] == 2 for s in out)
    assert all(s.metadata["hybrid_num_stage1_aborted_placeholders"] == 2 for s in out)


def test_hybrid_cancelled_branch_is_dropped(tmp_path):
    os.environ["STEP_GRPO_HYBRID_K"] = "2"
    b0 = _bundle(tmp_path / "a", ["", "diff --git a/a b/a\n+1\n"])
    b1 = _bundle(tmp_path / "b", ["", "diff --git a/a b/a\n+2\n"])

    async def vanilla_runner(**kwargs):
        i = kwargs["trial_idx"]
        bundle = b0 if i == 0 else b1
        lps = [[], [-1.0]] if i == 0 else [[], [-5.0]]
        return bundle, [_sample(0.0, index=i)], False, lps

    calls = []

    async def branch_runner(**kwargs):
        calls.append(kwargs["branch_idx"])
        if len(calls) == 1:
            raise asyncio.CancelledError()
        s = _sample(0.0, index=100 + len(calls))
        s.metadata = {
            "sample_kind": "branch",
            "step_group_key": f"0:{kwargs['source_trial_idx']}:edit:{kwargs['edit_step_i']}",
            "source_trial_idx": kwargs["source_trial_idx"],
            "edit_step_i": kwargs["edit_step_i"],
            "branch_step_t": kwargs["branch_step_t"],
            "branch_idx": kwargs["branch_idx"],
            "edit_ppl": kwargs["edit_ppl"],
        }
        s.loss_mask = [1]
        return [s]

    out = asyncio.run(
        hybrid_generate(
            SimpleNamespace(),
            _sample(),
            {},
            vanilla_runner=vanilla_runner,
            branch_runner=branch_runner,
        )
    )

    branches = [s for s in out if s.metadata["sample_kind"] == "branch"]
    assert len(calls) == 4
    assert len(branches) == 3
    assert all(s.metadata["hybrid_num_dropped_branches"] == 1 for s in out)
    assert all(s.metadata["hybrid_num_dropped_cancelled"] == 1 for s in out)


def test_hybrid_stage2_fn_replaces_selection_and_branching(tmp_path):
    os.environ["STEP_GRPO_HYBRID_K"] = "2"
    b0 = _bundle(tmp_path / "a", ["", "diff --git a/a b/a\n+1\n"])
    b1 = _bundle(tmp_path / "b", ["", "diff --git a/a b/a\n+2\n"])

    async def vanilla_runner(**kwargs):
        i = kwargs["trial_idx"]
        bundle = b0 if i == 0 else b1
        s = _sample(0.0, index=i)
        s.metadata = {"sample_kind": "vanilla", "trial_idx": i}
        return bundle, [s], False, [[], [-1.0]]

    async def branch_runner(**kwargs):
        raise AssertionError("stage2_fn must replace branching")

    seen = {}

    async def stage2_fn(**kwargs):
        seen.update(kwargs)
        s = _sample(0.0, index=99)
        s.metadata = {"sample_kind": "teacher_sft", "branch_uid": "tt:0:t0:c0"}
        s.loss_mask = [1]
        return [s]

    out = asyncio.run(
        hybrid_generate(
            SimpleNamespace(),
            _sample(),
            {},
            vanilla_runner=vanilla_runner,
            branch_runner=branch_runner,
            stage2_fn=stage2_fn,
        )
    )

    assert [t.trial_idx for t in seen["trials"]] == [0, 1]
    assert seen["sampling_params"] == {}
    assert "hybrid_stage2_context_limit_tokens" in seen["hybrid_stats"]
    kinds = [s.metadata["sample_kind"] for s in out]
    assert kinds.count("vanilla") == 2
    assert kinds.count("teacher_sft") == 1
    assert len({s.rollout_id for s in out}) == 1
    assert len({s.loss_group_id for s in out}) == 3
    assert all(s.metadata["hybrid_stage2_wall_sec"] >= 0.0 for s in out)


def test_hybrid_step_turn_mismatch_raises():
    """Alignment bugs must not silently degrade to vanilla-only."""
    from examples.claudecode_ags.step_reconstruct.edit_ppl import StepTurnAlignmentError

    os.environ["STEP_GRPO_HYBRID_K"] = "2"

    async def vanilla_runner(**kwargs):
        raise StepTurnAlignmentError("step_turn_mismatch: aligned_tool_turns=1 num_steps=2")

    with pytest.raises(StepTurnAlignmentError, match="step_turn_mismatch"):
        asyncio.run(
            hybrid_generate(
                SimpleNamespace(),
                _sample(),
                {},
                vanilla_runner=vanilla_runner,
                branch_runner=AsyncMock(),
            )
        )


def test_eval_delegates_to_path_a_generate():
    with patch(
        "examples.claudecode_ags.generate.generate",
        new_callable=AsyncMock,
    ) as m:
        m.return_value = _sample(0.0)
        out = asyncio.run(hybrid_generate(SimpleNamespace(), _sample(), {}, evaluation=True))
        assert m.await_count == 1
        assert out.reward == 0.0
