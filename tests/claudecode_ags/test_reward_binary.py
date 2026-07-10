from examples.claudecode_ags.rewards.default import compose


def test_binary_resolved_true():
    reward, details = compose(base_eval={"resolved": True}, sample=None)
    assert reward == 1.0
    assert details["mode"] == "binary"


def test_binary_resolved_false():
    reward, details = compose(base_eval={"resolved": False}, sample=None)
    assert reward == 0.0
    assert details["mode"] == "binary"
