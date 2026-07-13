from examples.claudecode_ags.step_reconstruct.edit_ppl import PatchTurnPPL, iter_patch_turn_ppls


def test_iter_patch_turn_ppls_detects_diff_changes():
    # step0 empty, step1 adds file, step2 same → only step1 is patch turn
    diffs = ["", "diff --git a/a.py b/a.py\n+x\n", "diff --git a/a.py b/a.py\n+x\n"]
    turns = [[-1.0, -1.0], [-2.0, -2.0], [-3.0]]
    got = iter_patch_turn_ppls(diffs, turns, clip=20.0)
    assert len(got) == 1
    assert got[0] == PatchTurnPPL(step_t=1, edit_ppl=2.0, n_tokens=2)


def test_iter_patch_turn_ppls_clips():
    diffs = ["", "diff --git a/a.py b/a.py\n+x\n"]
    turns = [[], [-100.0]]  # -logp would be 100 → clip 5
    got = iter_patch_turn_ppls(diffs, turns, clip=5.0)
    assert len(got) == 1
    assert got[0].edit_ppl == 5.0


def test_iter_patch_turn_ppls_mismatch_returns_empty():
    assert iter_patch_turn_ppls(["a", "b"], [[-1.0]], clip=20.0) == []
