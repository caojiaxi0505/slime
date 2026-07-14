import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from examples.claudecode_ags.step_reconstruct.hybrid_generate import (
    VanillaTrialResult,
    collect_patch_candidates,
    hybrid_generate,
    select_branch_turns,
)
from examples.claudecode_ags.step_reconstruct.session_capture import SessionBundle, steps_from_diff_files
from slime.utils.types import Sample


def _bundle(tmp_path, diffs):
    d = tmp_path / "b"
    d.mkdir(parents=True, exist_ok=True)
    steps = steps_from_diff_files(str(d), diffs)
    b = SessionBundle(
        instance_id="x",
        session_id="s",
        cc_session_id="",
        task_metadata={"image": "img", "workdir": "/testbed"},
        steps=steps,
        dir=str(d),
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
    assert sel[0].step_t == 1


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
        calls.append((kwargs["source_trial_idx"], kwargs["step_t"], kwargs["branch_idx"]))
        s = _sample(0.0, index=100 + len(calls))
        s.metadata = {
            "sample_kind": "branch",
            "step_group_key": f"0:{kwargs['source_trial_idx']}:{kwargs['step_t']}",
            "source_trial_idx": kwargs["source_trial_idx"],
            "step_t": kwargs["step_t"],
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


def test_hybrid_all_trials_fail_returns_abort_sample():
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
    assert len(out) == 1
    s = out[0]
    assert s.status == Sample.Status.ABORTED
    assert s.remove_sample is True
    assert s.tokens == [0, 0]
    assert s.response_length == 1
    assert s.loss_mask == [0]
    assert s.metadata["abort_reason"] == "all_vanilla_trials_failed"


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
