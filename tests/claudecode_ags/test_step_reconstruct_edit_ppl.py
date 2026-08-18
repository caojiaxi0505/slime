import pytest

from examples.claudecode_ags.step_reconstruct.edit_ppl import (
    PatchTurnPPL,
    StepTurnAlignmentError,
    align_logprobs_to_steps,
    iter_turn_ppls,
)


def test_iter_turn_ppls_detects_diff_changes():
    # step0 empty, step1 adds file, step2 same → only step1 is patch turn
    diffs = ["", "diff --git a/a.py b/a.py\n+x\n", "diff --git a/a.py b/a.py\n+x\n"]
    turns = [[-1.0, -1.0], [-2.0, -2.0], [-3.0]]
    got = iter_turn_ppls(diffs, turns, clip=20.0)
    assert len(got) == 1
    assert got[0] == PatchTurnPPL(step_t=1, edit_ppl=2.0, n_tokens=2)


def test_iter_turn_ppls_without_patch_only_keeps_every_tool_turn():
    diffs = ["", "diff --git a/a.py b/a.py\n+x\n", "diff --git a/a.py b/a.py\n+x\n"]
    turns = [[-1.0, -1.0], [-2.0, -2.0], [-3.0]]
    got = iter_turn_ppls(diffs, turns, clip=20.0, patch_only=False)
    assert [p.step_t for p in got] == [0, 1, 2]
    assert got[0] == PatchTurnPPL(step_t=0, edit_ppl=1.0, n_tokens=2)


def test_iter_turn_ppls_clips():
    diffs = ["", "diff --git a/a.py b/a.py\n+x\n"]
    turns = [[], [-100.0]]  # -logp would be 100 → clip 5
    got = iter_turn_ppls(diffs, turns, clip=5.0)
    assert len(got) == 1
    assert got[0].edit_ppl == 5.0


def test_iter_turn_ppls_mismatch_raises():
    with pytest.raises(StepTurnAlignmentError, match="step_turn_mismatch"):
        iter_turn_ppls(["a", "b"], [[-1.0]], clip=20.0)


def test_align_by_tool_use_id_joins_and_drops_extras():
    # text-only, multi-tool turn, single-tool; only snapshotted ids must align
    entries = [
        ([-0.1], []),
        ([-1.0, -1.0], ["toolu_a", "toolu_b"]),
        ([-2.0], ["toolu_c"]),
        ([-9.0], ["toolu_dropped"]),  # emitted but no PostToolUse snapshot
    ]
    got = align_logprobs_to_steps(entries, ["toolu_a", "toolu_b", "toolu_c"])
    assert got == [[-1.0, -1.0], [-1.0, -1.0], [-2.0]]


def test_align_missing_step_id_raises():
    with pytest.raises(StepTurnAlignmentError, match="missing_turn_for_steps"):
        align_logprobs_to_steps([([-1.0], ["toolu_a"])], ["toolu_a", "toolu_missing"])


def test_align_zero_steps_ok():
    assert align_logprobs_to_steps([([-0.1], [])], []) == []
