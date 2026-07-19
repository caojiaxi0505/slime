import json

import pytest

from examples.claudecode_ags.step_reconstruct.native_session import (
    parse_native_session,
    truncate_native_session_before_tools,
)


def _row(row_type, uuid, parent, *, session="session-1", message=None, side=False):
    value = {
        "type": row_type,
        "uuid": uuid,
        "parentUuid": parent,
        "isSidechain": side,
        "sessionId": session,
        "cwd": "/testbed",
        "version": "2.1.104",
        "gitBranch": "buggy",
    }
    if message is not None:
        value["message"] = message
    return value


def _jsonl(rows):
    return "".join(json.dumps(row) + "\n" for row in rows)


def test_truncate_native_session_cuts_before_whole_parallel_assistant_turn():
    rows = [
        _row("user", "user-1", None, message={"role": "user", "content": "fix"}),
        _row("attachment", "attachment-1", "user-1"),
        _row(
            "assistant",
            "assistant-text",
            "attachment-1",
            message={
                "id": "msg-parallel",
                "content": [{"type": "text", "text": "checking"}],
            },
        ),
        _row(
            "assistant",
            "assistant-a",
            "assistant-text",
            message={
                "id": "msg-parallel",
                "content": [{"type": "tool_use", "id": "toolu_a", "name": "Read", "input": {}}],
            },
        ),
        _row(
            "assistant",
            "assistant-b",
            "assistant-a",
            message={
                "id": "msg-parallel",
                "content": [{"type": "tool_use", "id": "toolu_b", "name": "Bash", "input": {}}],
            },
        ),
        _row("user", "result-a", "assistant-a", message={"role": "user", "content": []}),
    ]
    prefix = truncate_native_session_before_tools(_jsonl(rows), ["toolu_a", "toolu_b"])
    out = parse_native_session(prefix.jsonl)
    assert prefix.target_row_index == 2
    assert prefix.leaf_uuid == "attachment-1"
    assert len(out) == 3
    assert out[-1]["type"] == "assistant"
    assert out[-1]["parentUuid"] == "attachment-1"
    assert all("toolu_a" not in json.dumps(row) and "toolu_b" not in json.dumps(row) for row in out)


def test_truncate_native_session_rejects_ids_from_different_assistant_turns():
    rows = [
        _row("user", "u", None, message={"role": "user", "content": "fix"}),
        _row(
            "assistant",
            "a",
            "u",
            message={"id": "msg-a", "content": [{"type": "tool_use", "id": "toolu_a"}]},
        ),
        _row(
            "assistant",
            "b",
            "a",
            message={"id": "msg-b", "content": [{"type": "tool_use", "id": "toolu_b"}]},
        ),
    ]
    with pytest.raises(ValueError, match="one native Claude Code assistant turn"):
        truncate_native_session_before_tools(_jsonl(rows), ["toolu_a", "toolu_b"])


def test_truncate_native_session_rejects_missing_target_id():
    rows = [_row("user", "u", None, message={"role": "user", "content": "fix"})]
    with pytest.raises(ValueError, match="expected one native assistant row"):
        truncate_native_session_before_tools(_jsonl(rows), ["toolu_missing"])
