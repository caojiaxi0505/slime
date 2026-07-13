from examples.claudecode_ags.step_reconstruct.selection import (
    PatchTurnCandidate,
    SelectedTurn,
    select_top_patch_turns,
)


def test_select_top_across_trials():
    cands = [
        PatchTurnCandidate(0, 1, 1.0),
        PatchTurnCandidate(1, 2, 5.0),
        PatchTurnCandidate(0, 3, 3.0),
        PatchTurnCandidate(2, 0, 4.0),
    ]
    got = select_top_patch_turns(cands, k=3)
    assert [ (s.source_trial_idx, s.step_t, s.edit_ppl) for s in got ] == [
        (1, 2, 5.0),
        (2, 0, 4.0),
        (0, 3, 3.0),
    ]


def test_undersubscribe():
    cands = [PatchTurnCandidate(0, 1, 2.0)]
    got = select_top_patch_turns(cands, k=8)
    assert got == [SelectedTurn(0, 1, 2.0)]


def test_empty_pool():
    assert select_top_patch_turns([], k=8) == []
