"""Unit tests for segmented Anthropic chain routing (no HTTP)."""

from slime.agent.adapters.anthropic_segmented import (
    Session,
    append_turn,
    commit_fingerprint,
    select_chain,
    start_sub_chain,
)
from slime.agent.trajectory import TurnRecord


def _turn(prompt=(1,), output=(2,)) -> TurnRecord:
    return TurnRecord(prompt_ids=list(prompt), output_ids=list(output), finish_reason="stop")


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _tool_result(tool_use_id: str, content: str = "done") -> dict:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": content}],
    }


def test_empty_main_first_request_is_new():
    s = Session()
    body = {"messages": [_user("hello")], "system": "sys"}
    target, is_sub, kind = select_chain(s, body)
    assert target is s.main
    assert is_sub is False
    assert kind == "new"


def test_prefix_continuing_messages_are_append():
    s = Session()
    first = {"messages": [_user("hello")], "system": "sys"}
    target, _, kind = select_chain(s, first)
    assert kind == "new"
    commit_fingerprint(target, first, kind)
    append_turn(target, _turn())

    cont = {
        "messages": [_user("hello"), _assistant("hi"), _user("next")],
        "system": "sys",
    }
    target2, is_sub, kind2 = select_chain(s, cont)
    assert target2 is s.main
    assert is_sub is False
    assert kind2 == "append"


def test_divergent_main_messages_wipe_and_record_segment():
    s = Session()
    first = {"messages": [_user("hello")], "system": "sys"}
    target, _, kind = select_chain(s, first)
    commit_fingerprint(target, first, kind)
    append_turn(target, _turn(prompt=(10,), output=(11, 12)))

    wipe_body = {"messages": [_user("compacted")], "system": "sys"}
    target2, is_sub, kind2 = select_chain(s, wipe_body)
    assert target2 is s.main
    assert is_sub is False
    assert kind2 == "wipe"
    assert len(s.segments) == 1
    assert s.segments[0].metadata.get("segment_kind") == "wipe"
    assert len(s.segments[0].turns) == 1


def test_after_task_dispatch_non_continuing_routes_to_sub():
    s = Session()
    main_msgs = [_user("do work")]
    first = {"messages": main_msgs, "system": "sys"}
    target, _, kind = select_chain(s, first)
    commit_fingerprint(target, first, kind)
    append_turn(target, _turn())
    start_sub_chain(s, "toolu_task_1")
    assert s.active_sub is not None
    assert s.pending_dispatch_id == "toolu_task_1"

    # Subagent conversation does not continue main's prefix.
    sub_body = {
        "messages": [_user("sub prompt")],
        "system": "sub-sys",
    }
    target2, is_sub, kind2 = select_chain(s, sub_body)
    assert is_sub is True
    assert target2 is s.active_sub
    assert kind2 == "new"


def test_tool_result_for_pending_dispatch_closes_sub_as_subagent():
    s = Session()
    first = {"messages": [_user("do work")], "system": "sys"}
    target, _, kind = select_chain(s, first)
    commit_fingerprint(target, first, kind)
    append_turn(target, _turn())
    start_sub_chain(s, "toolu_task_1")

    sub_body = {"messages": [_user("sub prompt")], "system": "sub-sys"}
    sub_target, is_sub, sub_kind = select_chain(s, sub_body)
    assert is_sub and sub_kind == "new"
    commit_fingerprint(sub_target, sub_body, sub_kind)
    append_turn(sub_target, _turn(prompt=(20,), output=(21,)))

    # Main continues with the Task tool_result → close sub.
    close_body = {
        "messages": [
            _user("do work"),
            _assistant("calling task"),
            _tool_result("toolu_task_1"),
        ],
        "system": "sys",
    }
    target2, is_sub2, kind2 = select_chain(s, close_body)
    assert s.active_sub is None
    assert s.pending_dispatch_id == ""
    assert any(seg.metadata.get("segment_kind") == "subagent" for seg in s.segments)
    assert target2 is s.main
    assert is_sub2 is False
    assert kind2 == "append"
