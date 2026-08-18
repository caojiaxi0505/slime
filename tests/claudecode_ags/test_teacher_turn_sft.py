"""Unit tests for Stage-2 teacher turn relabeling (sandbox resume, capped teacher)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from examples.claudecode_ags.step_reconstruct import teacher_turn_sft as mod
from examples.claudecode_ags.step_reconstruct.selection import PatchTurnCandidate
from examples.claudecode_ags.step_reconstruct.step_grpo_advantage import _assign_loss_weights
from examples.claudecode_ags.step_reconstruct.teacher_turn_sft import (
    relabel_turns,
    select_relabel_turns,
    teacher_turn_samples,
    teacher_turn_select,
)

from slime.utils.types import Sample


def _candidate(trial_idx: int, step: int, ppl: float = 1.0) -> PatchTurnCandidate:
    return PatchTurnCandidate(
        source_trial_idx=trial_idx,
        edit_step_i=step,
        branch_step_t=step - 1,
        edit_ppl=ppl,
    )


def _trial(trial_idx: int, *, solved: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        trial_idx=trial_idx,
        is_solved=solved,
        bundle=SimpleNamespace(dir=f"/bundle/{trial_idx}"),
        samples=[],
        turn_logprobs=[],
    )


def _teacher_row(index: int, branch_uid: str) -> Sample:
    s = Sample(prompt="p", index=index, group_index=0, reward=0.0, metadata={"branch_uid": branch_uid})
    s.tokens = [1, 2, 3]
    s.loss_mask = [0, 1, 1]
    return s


def test_teacher_turn_select_validation(monkeypatch):
    monkeypatch.delenv("STEP_GRPO_TEACHER_TURN_SELECT", raising=False)
    assert teacher_turn_select() == "all"
    monkeypatch.setenv("STEP_GRPO_TEACHER_TURN_SELECT", "patch_ppl")
    assert teacher_turn_select() == "patch_ppl"
    monkeypatch.setenv("STEP_GRPO_TEACHER_TURN_SELECT", "nope")
    with pytest.raises(ValueError, match="STEP_GRPO_TEACHER_TURN_SELECT"):
        teacher_turn_select()


def test_relabel_turns_subsamples_each_trial_evenly():
    candidates = [_candidate(0, i) for i in range(1, 11)] + [_candidate(1, 1), _candidate(1, 2)]
    picked = relabel_turns(candidates, max_per_trial=3)
    assert [(s.source_trial_idx, s.edit_step_i) for s in picked] == [
        (0, 1),
        (0, 4),
        (0, 7),
        (1, 1),
        (1, 2),
    ]
    assert len(relabel_turns(candidates, max_per_trial=0)) == len(candidates)


def test_select_relabel_turns_modes(monkeypatch):
    all_turns = [_candidate(0, 1, ppl=1.0), _candidate(0, 2, ppl=9.0), _candidate(0, 3, ppl=5.0)]
    patch_turns = [all_turns[1], all_turns[2]]
    seen: list[bool] = []

    def _collect(trials, *, patch_only=True):
        del trials
        seen.append(patch_only)
        return patch_turns if patch_only else all_turns

    monkeypatch.setattr(mod, "collect_patch_candidates", _collect)
    monkeypatch.setenv("STEP_GRPO_TEACHER_TURN_MAX_PER_TRIAL", "0")

    monkeypatch.setenv("STEP_GRPO_TEACHER_TURN_SELECT", "all")
    _, selected = select_relabel_turns([])
    assert [s.edit_step_i for s in selected] == [1, 2, 3]
    assert seen[-1] is False

    monkeypatch.setenv("STEP_GRPO_TEACHER_TURN_SELECT", "patch")
    _, selected = select_relabel_turns([])
    assert [s.edit_step_i for s in selected] == [2, 3]
    assert seen[-1] is True

    monkeypatch.setenv("STEP_GRPO_TEACHER_TURN_SELECT", "patch_ppl")
    monkeypatch.setenv("STEP_GRPO_HYBRID_K", "1")
    _, selected = select_relabel_turns([])
    assert [s.edit_step_i for s in selected] == [2]


def test_teacher_turn_samples_relabels_every_selected_turn(monkeypatch):
    trials = [_trial(0, solved=True), _trial(1)]
    monkeypatch.setattr(
        mod,
        "collect_patch_candidates",
        lambda trials, patch_only=True: [_candidate(1, 1), _candidate(1, 2)],
    )
    monkeypatch.setenv("STEP_GRPO_TEACHER_TURN_SELECT", "all")
    monkeypatch.setenv("STEP_GRPO_TEACHER_TURN_MAX_PER_TRIAL", "0")
    calls: list[dict] = []

    async def _runner(**kwargs):
        calls.append(kwargs)
        step = kwargs["edit_step_i"]
        return [_teacher_row(100 + step, f"teach:1:edit:{step}")]

    stats: dict[str, object] = {"hybrid_stage2_context_limit_tokens": 0}
    out = asyncio.run(
        teacher_turn_samples(
            args=SimpleNamespace(),
            sample=Sample(prompt="p", index=2, group_index=0, metadata={"instance_id": "inst-1"}),
            sampling_params={"temperature": 1.0},
            trials=trials,
            group_index=0,
            hybrid_stats=stats,
            branch_runner=_runner,
        )
    )

    assert [c["edit_step_i"] for c in calls] == [1, 2]
    assert {c["source_trial_idx"] for c in calls} == {1}
    assert {c["branch_idx"] for c in calls} == {0}
    assert {c["bundle"].dir for c in calls} == {"/bundle/1"}
    assert calls[0]["sampling_params"] == {"temperature": 1.0}
    assert [s.metadata["sample_kind"] for s in out] == ["teacher_sft", "teacher_sft"]
    assert stats["hybrid_num_selected_edits"] == 2
    assert stats["hybrid_num_branch_tasks"] == 2
    assert stats["hybrid_num_stage2_samples_after_length_filter"] == 2


def test_teacher_turn_samples_buckets_runner_failures(monkeypatch):
    monkeypatch.setattr(
        mod,
        "collect_patch_candidates",
        lambda trials, patch_only=True: [_candidate(0, 1), _candidate(0, 2)],
    )
    monkeypatch.setenv("STEP_GRPO_TEACHER_TURN_MAX_PER_TRIAL", "0")

    async def _runner(**kwargs):
        if kwargs["edit_step_i"] == 1:
            raise RuntimeError("rebuild apply/verify failed t=0")
        return [_teacher_row(101, "teach:0:edit:2")]

    stats: dict[str, object] = {"hybrid_stage2_context_limit_tokens": 0}
    out = asyncio.run(
        teacher_turn_samples(
            args=SimpleNamespace(),
            sample=Sample(prompt="p", index=0, group_index=0, metadata={}),
            sampling_params={},
            trials=[_trial(0)],
            group_index=0,
            hybrid_stats=stats,
            branch_runner=_runner,
        )
    )
    assert len(out) == 1
    assert stats["hybrid_num_dropped_branches"] == 1


def test_teacher_turn_samples_returns_nothing_without_candidates(monkeypatch):
    monkeypatch.setattr(mod, "collect_patch_candidates", lambda trials, patch_only=True: [])

    async def _runner(**kwargs):
        raise AssertionError("no candidates must not launch relabels")

    out = asyncio.run(
        teacher_turn_samples(
            args=SimpleNamespace(),
            sample=Sample(prompt="p", index=0, group_index=0, metadata={}),
            sampling_params={},
            trials=[_trial(0, solved=True)],
            group_index=0,
            branch_runner=_runner,
        )
    )
    assert out == []


def test_capped_teacher_skips_grading(monkeypatch):
    # teacher_branch_runner imports the live generate module; skip where its
    # optional deps are unavailable.
    runner = pytest.importorskip("examples.claudecode_ags.step_reconstruct.teacher_branch_runner")
    monkeypatch.delenv("STEP_GRPO_TEACHER_MAX_STEPS", raising=False)
    monkeypatch.delenv("STEP_GRPO_TEACHER_SFT_RESOLVED_ONLY", raising=False)
    assert runner.teacher_max_steps() == 2
    assert runner.teacher_grades_continuation() is False

    monkeypatch.setenv("STEP_GRPO_TEACHER_MAX_STEPS", "0")
    assert runner.teacher_grades_continuation() is True
    monkeypatch.setenv("STEP_GRPO_TEACHER_SFT_RESOLVED_ONLY", "0")
    assert runner.teacher_grades_continuation() is False

    monkeypatch.setenv("STEP_GRPO_TEACHER_MAX_STEPS", "-1")
    with pytest.raises(ValueError, match="STEP_GRPO_TEACHER_MAX_STEPS"):
        runner.teacher_max_steps()


def test_teacher_and_student_sft_logs_must_be_isolated(monkeypatch, tmp_path):
    runner = pytest.importorskip("examples.claudecode_ags.step_reconstruct.teacher_branch_runner")
    student = tmp_path / "student"
    teacher = tmp_path / "teacher"
    monkeypatch.setenv("SLIME_AGENT_SFT_LOG_DIR", str(student))
    monkeypatch.setenv("SLIME_TEACHER_SFT_LOG_DIR", str(teacher))
    assert runner._teacher_sft_log_dir(SimpleNamespace()) == str(teacher)

    monkeypatch.setenv("SLIME_TEACHER_SFT_LOG_DIR", str(student))
    with pytest.raises(RuntimeError, match="teacher_sft_log_isolation"):
        runner._teacher_sft_log_dir(SimpleNamespace())


def _row(kind: str, *, index: int, branch_uid: str, trial_idx: int = 0) -> Sample:
    return Sample(
        prompt="p",
        index=index,
        group_index=0,
        reward=0.0,
        loss_mask=[1, 1],
        metadata={
            "sample_kind": kind,
            "trial_idx": trial_idx,
            "stage1_group_size": 2,
            "branch_uid": branch_uid,
        },
        loss_group_id=f"g-{kind}-{index}",
    )


def test_assign_loss_weights_normalizes_teacher_rows_per_prompt():
    vanilla = [
        _row("vanilla", index=0, branch_uid="v:0:t0", trial_idx=0),
        _row("vanilla", index=1, branch_uid="v:0:t1", trial_idx=1),
    ]
    teacher = [_row("teacher_sft", index=10 + i, branch_uid=f"teach:0:t1:c{i}") for i in range(4)]
    active_v, active_bg, active_b, active_t = _assign_loss_weights(vanilla + teacher)

    assert (active_v, active_bg, active_b, active_t) == (2, 0, 0, 4)
    assert [s.loss_weight for s in vanilla] == [0.5, 0.5]
    assert [s.loss_weight for s in teacher] == [0.25] * 4
    assert sum(s.loss_weight for s in teacher) == pytest.approx(1.0)
