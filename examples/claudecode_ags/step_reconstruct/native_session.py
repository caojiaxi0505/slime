"""Strict Claude Code native-session truncation for Stage-2 branches."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NativeSessionPrefix:
    jsonl: str
    target_row_index: int
    source_session_id: str
    target_tool_use_ids: list[str]
    leaf_uuid: str


def _tool_use_ids(row: dict[str, Any]) -> list[str]:
    if row.get("type") != "assistant":
        return []
    content = (row.get("message") or {}).get("content")
    if not isinstance(content, list):
        return []
    return [
        str(block["id"])
        for block in content
        if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id")
    ]


def parse_native_session(session_jsonl: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate((session_jsonl or "").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid native session JSON at line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"native session line {line_number} is not an object")
        rows.append(row)
    if not rows:
        raise ValueError("native Claude Code session is empty")
    return rows


def _synthetic_terminal(parent: dict[str, Any], session_id: str) -> dict[str, Any]:
    """Create the minimal terminal assistant record accepted by CC 2.1.104."""
    return {
        "parentUuid": parent["uuid"],
        "isSidechain": False,
        "message": {
            "id": "msg_slime_resume_terminal",
            "type": "message",
            "role": "assistant",
            "model": "slime-actor",
            "content": [{"type": "text", "text": "Stage-2 resume checkpoint."}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 1,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 1,
            },
        },
        "type": "assistant",
        "uuid": str(uuid.uuid4()),
        "timestamp": parent.get("timestamp", "1970-01-01T00:00:00.000Z"),
        "userType": parent.get("userType", "external"),
        "entrypoint": parent.get("entrypoint", "sdk-cli"),
        "cwd": parent.get("cwd", "/testbed"),
        "sessionId": session_id,
        "version": parent.get("version", "2.1.104"),
        "gitBranch": parent.get("gitBranch", ""),
    }


def truncate_native_session_before_tools(
    session_jsonl: str,
    target_tool_use_ids: list[str],
) -> NativeSessionPrefix:
    """Cut before the whole assistant turn that generated ``target_tool_use_ids``.

    Claude Code 2.1.104 may persist parallel tools as adjacent assistant rows.
    They share one API message id and one prompt checkpoint, so the cut is made
    before the earliest matching row and all ids must belong to that same turn.
    """
    targets = [str(value) for value in target_tool_use_ids if str(value)]
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("target tool_use ids must be non-empty and unique")
    rows = parse_native_session(session_jsonl)

    matches: dict[str, list[int]] = {tool_id: [] for tool_id in targets}
    row_message_ids: dict[int, str] = {}
    for index, row in enumerate(rows):
        row_ids = _tool_use_ids(row)
        if row_ids:
            row_message_ids[index] = str((row.get("message") or {}).get("id") or "")
        for tool_id in row_ids:
            if tool_id in matches:
                matches[tool_id].append(index)

    for tool_id, indices in matches.items():
        if len(indices) != 1:
            raise ValueError(f"expected one native assistant row for {tool_id}, got {len(indices)}")
    target_indices = sorted({indices[0] for indices in matches.values()})
    message_ids = {row_message_ids.get(index, "") for index in target_indices}
    if "" in message_ids or len(message_ids) != 1:
        raise ValueError(
            "checkpoint tool ids do not belong to one native Claude Code assistant turn"
        )

    target_message_id = next(iter(message_ids))
    whole_turn_indices = [
        index
        for index, row in enumerate(rows)
        if row.get("type") == "assistant"
        and str((row.get("message") or {}).get("id") or "") == target_message_id
    ]
    if not whole_turn_indices:
        raise ValueError("native session target assistant turn disappeared")
    target_index = min(whole_turn_indices)
    prefix_rows = rows[:target_index]
    leaf = next(
        (
            row
            for row in reversed(prefix_rows)
            if row.get("uuid") and not bool(row.get("isSidechain", False))
        ),
        None,
    )
    if leaf is None:
        raise ValueError("native session has no resumable leaf before target assistant turn")

    session_ids = {
        str(row.get("sessionId"))
        for row in rows
        if row.get("sessionId")
    }
    if len(session_ids) != 1:
        raise ValueError(f"native session must contain exactly one sessionId, got {sorted(session_ids)}")
    source_session_id = next(iter(session_ids))
    terminal = _synthetic_terminal(leaf, source_session_id)
    output_rows = prefix_rows + [terminal]
    return NativeSessionPrefix(
        jsonl="".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in output_rows),
        target_row_index=target_index,
        source_session_id=source_session_id,
        target_tool_use_ids=targets,
        leaf_uuid=str(leaf["uuid"]),
    )


__all__ = [
    "NativeSessionPrefix",
    "parse_native_session",
    "truncate_native_session_before_tools",
]
