"""Claude Code stream-json validation and prefix reconstruction.

The output trajectory contains many diagnostic ``stream_event`` rows.  Stage-2
must feed only canonical ``user``/``assistant`` conversation rows back through
``--input-format stream-json`` and must stop at a tool boundary that matches the
workspace snapshot being rebuilt.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from examples.claudecode_ags.claude_stream_input import encode_stream_events, initial_user_event


class TranscriptValidationError(ValueError):
    """Raised when a transcript cannot safely seed a Stage-2 continuation."""


@dataclass(frozen=True)
class TranscriptValidation:
    valid: bool
    error: str = ""
    event_count: int = 0
    conversation_event_count: int = 0
    snapshot_count: int = 0
    tool_use_count: int = 0
    tool_result_count: int = 0
    last_aligned_tool_use_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_events(transcript_jsonl: str) -> list[dict[str, Any]]:
    if not (transcript_jsonl or "").strip():
        raise TranscriptValidationError("transcript_empty")

    events: list[dict[str, Any]] = []
    for line_no, raw in enumerate(transcript_jsonl.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TranscriptValidationError(
                f"transcript_invalid_json:line={line_no}:{exc.msg}"
            ) from exc
        if not isinstance(event, dict):
            raise TranscriptValidationError(f"transcript_event_not_object:line={line_no}")
        events.append(event)
    if not events:
        raise TranscriptValidationError("transcript_empty")
    return events


def _content_blocks(event: dict[str, Any]) -> list[dict[str, Any]]:
    message = event.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if isinstance(content, list):
        return [block for block in content if isinstance(block, dict)]
    return []


def _tool_use_ids(event: dict[str, Any]) -> list[str]:
    if event.get("type") != "assistant":
        return []
    return [
        str(block.get("id") or "")
        for block in _content_blocks(event)
        if block.get("type") == "tool_use" and str(block.get("id") or "").strip()
    ]


def _tool_result_ids(event: dict[str, Any]) -> list[str]:
    if event.get("type") != "user":
        return []
    return [
        str(block.get("tool_use_id") or "")
        for block in _content_blocks(event)
        if block.get("type") == "tool_result" and str(block.get("tool_use_id") or "").strip()
    ]


def _message_id(event: dict[str, Any]) -> str:
    message = event.get("message")
    if not isinstance(message, dict):
        return ""
    return str(message.get("id") or "")


def _tool_result_only(event: dict[str, Any]) -> bool:
    blocks = _content_blocks(event)
    return bool(blocks) and all(block.get("type") == "tool_result" for block in blocks)


def _append_content(target: dict[str, Any], source: dict[str, Any]) -> None:
    target_message = target.get("message")
    source_message = source.get("message")
    if not isinstance(target_message, dict) or not isinstance(source_message, dict):
        return
    target_content = target_message.get("content")
    source_content = source_message.get("content")
    if not isinstance(target_content, list) or not isinstance(source_content, list):
        return
    target_content.extend(copy.deepcopy(source_content))


def _conversation_events(events: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Coalesce Claude Code's split rows into logical Messages API turns.

    Claude Code 2.1.104 emits thinking/text/each parallel ``tool_use`` as
    adjacent assistant rows sharing one ``message.id``.  It likewise emits
    parallel tool results as adjacent user rows.  Treating each row as a new
    turn falsely rejects valid transcripts, so validation first reconstructs
    the logical assistant/user pair.
    """
    out: list[dict[str, Any]] = []
    for raw in events:
        if raw.get("type") not in {"user", "assistant"}:
            continue
        event = copy.deepcopy(raw)
        if out and event.get("type") == "assistant":
            message_id = _message_id(event)
            if (
                message_id
                and out[-1].get("type") == "assistant"
                and _message_id(out[-1]) == message_id
            ):
                _append_content(out[-1], event)
                continue
        if (
            out
            and event.get("type") == "user"
            and out[-1].get("type") == "user"
            and _tool_result_only(out[-1])
            and _tool_result_only(event)
        ):
            _append_content(out[-1], event)
            continue
        out.append(event)
    return out


def _positions_by_id(
    conversation: Sequence[dict[str, Any]], extractor
) -> dict[str, int]:
    positions: dict[str, int] = {}
    duplicates: set[str] = set()
    for pos, event in enumerate(conversation):
        for tool_id in extractor(event):
            if tool_id in positions:
                duplicates.add(tool_id)
            else:
                positions[tool_id] = pos
    if duplicates:
        example = sorted(duplicates)[:3]
        raise TranscriptValidationError(
            f"transcript_duplicate_tool_ids:n={len(duplicates)}:example={example}"
        )
    return positions


def _validated_layout(
    transcript_jsonl: str, snapshot_tool_use_ids: Sequence[str]
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
    dict[str, int],
    dict[str, int],
]:
    events = _parse_events(transcript_jsonl)
    conversation = _conversation_events(events)
    snapshot_ids = [str(tool_id or "").strip() for tool_id in snapshot_tool_use_ids]

    empty = [i for i, tool_id in enumerate(snapshot_ids) if not tool_id]
    if empty:
        raise TranscriptValidationError(
            f"snapshot_tool_use_id_missing:n={len(empty)}:example={empty[:3]}"
        )
    if len(set(snapshot_ids)) != len(snapshot_ids):
        raise TranscriptValidationError("snapshot_tool_use_id_duplicate")

    use_positions = _positions_by_id(conversation, _tool_use_ids)
    result_positions = _positions_by_id(conversation, _tool_result_ids)

    # A Stage-2 prefix is only safe when the conversation and workspace have
    # the same complete sequence of tool boundaries.  PostToolUse and
    # PostToolUseFailure hooks provide one snapshot for successful and failed
    # tools respectively; anything missing on either side is therefore a hard
    # alignment failure, not a reason to carry a later diff backwards.
    use_ids = set(use_positions)
    result_ids = set(result_positions)
    missing_result = sorted(use_ids - result_ids)
    orphan_result = sorted(result_ids - use_ids)
    if missing_result:
        raise TranscriptValidationError(
            f"transcript_missing_tool_result:n={len(missing_result)}:example={missing_result[:3]}"
        )
    if orphan_result:
        raise TranscriptValidationError(
            f"transcript_missing_tool_use:n={len(orphan_result)}:example={orphan_result[:3]}"
        )

    snapshot_set = set(snapshot_ids)
    missing_snapshot = sorted(use_ids - snapshot_set)
    orphan_snapshot = sorted(snapshot_set - use_ids)
    if missing_snapshot:
        raise TranscriptValidationError(
            f"snapshot_missing_for_transcript_tool:n={len(missing_snapshot)}:example={missing_snapshot[:3]}"
        )
    if orphan_snapshot:
        raise TranscriptValidationError(
            f"snapshot_tool_missing_from_transcript:n={len(orphan_snapshot)}:example={orphan_snapshot[:3]}"
        )

    missing_use = [tool_id for tool_id in snapshot_ids if tool_id not in use_positions]
    missing_result = [tool_id for tool_id in snapshot_ids if tool_id not in result_positions]
    if missing_use:
        raise TranscriptValidationError(
            f"transcript_missing_tool_use:n={len(missing_use)}:example={missing_use[:3]}"
        )
    if missing_result:
        raise TranscriptValidationError(
            f"transcript_missing_tool_result:n={len(missing_result)}:example={missing_result[:3]}"
        )

    previous_use = -1
    previous_result = -1
    for tool_id in snapshot_ids:
        use_pos = use_positions[tool_id]
        result_pos = result_positions[tool_id]
        if use_pos >= result_pos:
            raise TranscriptValidationError(f"transcript_tool_order_invalid:{tool_id}")
        # Parallel tools may share one assistant event and one user-result event,
        # so equality is valid.  Moving backwards across turns is not.
        if use_pos < previous_use or result_pos < previous_result:
            raise TranscriptValidationError(f"snapshot_transcript_order_mismatch:{tool_id}")
        previous_use = use_pos
        previous_result = result_pos

    for tool_id in use_positions:
        if use_positions[tool_id] >= result_positions[tool_id]:
            raise TranscriptValidationError(f"transcript_tool_order_invalid:{tool_id}")

    # Stream-json replay follows the Messages API conversation shape: every
    # assistant tool-use message must be followed by the matching user
    # tool-result message. Accept parallel tools in the same pair of events.
    for pos, event in enumerate(conversation):
        event_use_ids = set(_tool_use_ids(event))
        if not event_use_ids:
            continue
        if pos + 1 >= len(conversation):
            raise TranscriptValidationError("transcript_tool_result_event_missing_at_end")
        next_result_ids = set(_tool_result_ids(conversation[pos + 1]))
        if event_use_ids != next_result_ids:
            raise TranscriptValidationError(
                "transcript_tool_result_not_adjacent:"
                f"use={sorted(event_use_ids)[:3]}:result={sorted(next_result_ids)[:3]}"
            )

    return events, conversation, snapshot_ids, use_positions, result_positions


def validate_transcript(
    transcript_jsonl: str, snapshot_tool_use_ids: Sequence[str]
) -> TranscriptValidation:
    """Validate JSON and the snapshot/tool-use/tool-result alignment."""
    try:
        events, conversation, snapshot_ids, use_positions, result_positions = _validated_layout(
            transcript_jsonl, snapshot_tool_use_ids
        )
    except TranscriptValidationError as exc:
        event_count = sum(1 for line in (transcript_jsonl or "").splitlines() if line.strip())
        return TranscriptValidation(
            valid=False,
            error=str(exc),
            event_count=event_count,
            snapshot_count=len(snapshot_tool_use_ids),
        )
    return TranscriptValidation(
        valid=True,
        event_count=len(events),
        conversation_event_count=len(conversation),
        snapshot_count=len(snapshot_ids),
        tool_use_count=len(use_positions),
        tool_result_count=len(result_positions),
        last_aligned_tool_use_id=snapshot_ids[-1] if snapshot_ids else "",
    )


def pre_tool_snapshot_index(
    transcript_jsonl: str,
    snapshot_tool_use_ids: Sequence[str],
    edit_step_i: int,
) -> int:
    """Return the latest completed snapshot before the target tool's turn.

    In the usual one-tool-per-turn case this is ``edit_step_i - 1``.  For a
    parallel tool-use turn it may be earlier, because the whole assistant turn
    must be sampled again as one action.
    """
    _, _, snapshot_ids, use_positions, result_positions = _validated_layout(
        transcript_jsonl, snapshot_tool_use_ids
    )
    if edit_step_i < 0 or edit_step_i >= len(snapshot_ids):
        raise TranscriptValidationError(
            f"edit_step_out_of_range:{edit_step_i}:num_steps={len(snapshot_ids)}"
        )
    target_use_pos = use_positions[snapshot_ids[edit_step_i]]
    eligible = [
        (result_positions[tool_id], i)
        for i, tool_id in enumerate(snapshot_ids)
        if result_positions[tool_id] < target_use_pos
    ]
    if not eligible:
        return -1
    return max(eligible)[1]


def build_transcript_prefix(
    transcript_jsonl: str,
    snapshot_tool_use_ids: Sequence[str],
    branch_step_t: int,
    *,
    initial_prompt: str,
) -> str:
    """Build valid stream-json input ending at ``branch_step_t``.

    ``branch_step_t=-1`` is the logical pre-tool state: only the original user
    prompt is present.  Non-negative values include conversation events through
    the matching tool result.  Diagnostic stream events are deliberately
    omitted.
    """
    _, conversation, snapshot_ids, _, _ = _validated_layout(
        transcript_jsonl, snapshot_tool_use_ids
    )
    if branch_step_t < -1 or branch_step_t >= len(snapshot_ids):
        raise TranscriptValidationError(
            f"branch_step_out_of_range:{branch_step_t}:num_steps={len(snapshot_ids)}"
        )

    prefix: list[dict[str, Any]] = [initial_user_event(initial_prompt)]
    if branch_step_t == -1:
        return encode_stream_events(prefix)

    target_result_id = snapshot_ids[branch_step_t]
    found = False
    saw_transcript_action = False
    for event in conversation:
        # Claude Code's output stream does not normally echo the initial input
        # prompt. If a version does, use our canonical copy only once.
        # Preserve any later user-text event instead of silently deleting it.
        if event.get("type") == "user" and not _tool_result_ids(event):
            if not saw_transcript_action:
                continue
        prefix.append(event)
        saw_transcript_action = True
        if target_result_id in _tool_result_ids(event):
            found = True
            break
    if not found:
        raise TranscriptValidationError(f"prefix_target_result_missing:{target_result_id}")
    return encode_stream_events(prefix)
