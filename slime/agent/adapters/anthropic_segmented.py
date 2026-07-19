"""Segmented Anthropic adapter for Claude Code subagent / compact rollouts.

Tracks an explicit ``main`` chain plus at most one ``active_sub``. Turns are
classified as ``new`` / ``append`` / ``wipe``; wipe and subagent close freeze
prior turns into training segments. ``finish_session`` drains remaining work as
``subagent`` (if any) then ``final``.

Routing helpers are pure enough to unit-test without HTTP. The HTTP
``/v1/messages`` path reuses official Anthropic wire translation and the shared
SGLang generate helper, without TrajectoryManager fork semantics.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import dataclasses
import functools
import hashlib
import json
import logging
import os
import struct
import time
from collections.abc import Callable
from typing import Any

from jsonschema import exceptions as jsonschema_exceptions
from jsonschema import validators as jsonschema_validators
from aiohttp import web

from slime.agent.adapters import anthropic as anth
from slime.agent.adapters.common import call_sglang_generate, close_sglang_client, sid_from_bearer
from slime.agent.parsing import parse_model_output
from slime.agent.segment_trajectory import (
    TokenSegment,
    TurnSegment,
    make_turn_segment,
    merge_turn_segments,
)
from slime.agent.trajectory import TurnRecord

logger = logging.getLogger(__name__)

# Claude Code dispatches sub-agents via these tool names.
SUBAGENT_TOOLS = frozenset({"Task", "Agent"})

Kind = str  # "new" | "append" | "wipe"


# Only fields that can change the sampled continuation belong here.  Wire-only
# fields such as ``stream`` and ``model`` do not affect the rendered prompt or
# the SGLang sampling request.
_GENERATION_BODY_KEYS = (
    "max_tokens",
    "stop_sequences",
    "temperature",
    "top_p",
    "top_k",
)


def _json_safe(value: Any) -> Any:
    """Return a detached JSON-compatible value with deterministic fallbacks."""
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def canonical_sha256(value: Any) -> str:
    """SHA-256 of canonical JSON (stable across dict insertion order)."""
    payload = json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prompt_ids_sha256(prompt_ids: list[int]) -> str:
    """Hash token ids as little-endian uint32 values, independent of Python."""
    digest = hashlib.sha256()
    for token_id in prompt_ids:
        value = int(token_id)
        if value < 0 or value > 0xFFFFFFFF:
            raise ValueError(f"token id outside uint32 range: {value}")
        digest.update(struct.pack("<I", value))
    return digest.hexdigest()


def tokenizer_fingerprint(tokenizer: Any) -> dict[str, Any]:
    """Small, serializable identity for the tokenizer and its chat template."""
    try:
        vocab_size = len(tokenizer)
    except (TypeError, AttributeError):
        vocab_size = getattr(tokenizer, "vocab_size", None)
    payload = {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "vocab_size": vocab_size,
        "chat_template": getattr(tokenizer, "chat_template", None),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", None),
    }
    safe = _json_safe(payload)
    safe["sha256"] = canonical_sha256(safe)
    return safe


def _wire_tool_ids(body: dict) -> tuple[list[str], list[str]]:
    """Return tool_use and tool_result ids in wire order."""
    use_ids: list[str] = []
    result_ids: list[str] = []
    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and block.get("id"):
                use_ids.append(str(block["id"]))
            elif block.get("type") == "tool_result" and block.get("tool_use_id"):
                result_ids.append(str(block["tool_use_id"]))
    return use_ids, result_ids


def _generation_config(body: dict, session: Session) -> dict[str, Any]:
    return _generation_config_values(
        body,
        sampling_defaults=session.sampling_defaults,
        max_context_tokens=session.max_context_tokens,
    )


def _generation_config_values(
    body: dict,
    *,
    sampling_defaults: dict[str, Any],
    max_context_tokens: int,
) -> dict[str, Any]:
    return _json_safe(
        {
            "request": {key: body[key] for key in _GENERATION_BODY_KEYS if key in body},
            "sampling_defaults": sampling_defaults,
            "max_context_tokens": max_context_tokens,
        }
    )


@dataclasses.dataclass
class PromptCheckpoint:
    """Authoritative model-input state immediately before one generation.

    ``prompt_ids`` is the source of truth for the first resumed request.  The
    canonical messages and tool schema are retained so later requests can be
    extended without trusting Claude Code's replayed HTTP history.
    """

    checkpoint_id: str
    prompt_ids: list[int]
    prompt_sha256: str
    chat_messages: list[dict]
    tools_schema: list[dict] | None
    tools_sha256: str
    generation_config: dict[str, Any]
    tokenizer_fingerprint: dict[str, Any]
    chain_kind: str
    request_kind: Kind
    request_index: int
    source_tool_use_ids: list[str] = dataclasses.field(default_factory=list)
    source_tool_result_ids: list[str] = dataclasses.field(default_factory=list)
    generated_tool_use_ids: list[str] = dataclasses.field(default_factory=list)
    generated_tool_use_names: dict[str, str] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, value: PromptCheckpoint | dict[str, Any]) -> PromptCheckpoint:
        if isinstance(value, cls):
            return copy.deepcopy(value)
        if not isinstance(value, dict):
            raise TypeError("resume_checkpoint must be PromptCheckpoint or dict")
        fields = {field.name for field in dataclasses.fields(cls)}
        unknown = set(value) - fields
        if unknown:
            raise ValueError(f"unknown PromptCheckpoint fields: {sorted(unknown)}")
        checkpoint = cls(**copy.deepcopy(value))
        checkpoint.prompt_ids = [int(x) for x in checkpoint.prompt_ids]
        return checkpoint


@dataclasses.dataclass(frozen=True)
class _ResumeResponseCache:
    request_sha256: str
    wire_sha256: str
    response_body: dict[str, Any]


@dataclasses.dataclass
class ResumeState:
    checkpoint: PromptCheckpoint
    handshake_pending: bool = True
    handshake_validated: bool = False
    first_prompt_sha256: str = ""
    first_prompt_exact: bool = False
    exact_request_count: int = 0
    pending_tool_uses: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    completed_tool_use_ids: list[str] = dataclasses.field(default_factory=list)
    last_stop_reason: str = ""
    tool_use_echo_mismatch_count: int = 0
    tool_use_echo_missing_count: int = 0
    tool_use_echo_payload_mismatch_count: int = 0
    runtime_tool_schema_mismatch_count: int = 0
    generated_runtime_tool_unavailable_count: int = 0
    generated_runtime_tool_input_invalid_count: int = 0
    max_tokens_continuation_count: int = 0
    post_end_turn_ack_count: int = 0
    request_replay_count: int = 0
    cached_response: _ResumeResponseCache | None = None
    error: str = ""


# ---------------------------------------------------------------------------
# Session / chain state
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Chain:
    """One conversation chain (main or sub) with prefix fingerprints + turns."""

    system_hash: str = ""
    chat_messages: list[dict] = dataclasses.field(default_factory=list)
    tools_schema: list[dict] | None = None
    seen_msgs: int = 0
    msg_hashes: list[str] = dataclasses.field(default_factory=list)
    turns: list[TurnRecord] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Session:
    """Per-sid segmented state: main chain, optional sub, frozen segments."""

    main: Chain = dataclasses.field(default_factory=Chain)
    active_sub: Chain | None = None
    pending_dispatch_id: str = ""
    sampling_defaults: dict = dataclasses.field(default_factory=dict)
    max_context_tokens: int = 0
    capture_prompt_checkpoints: bool = False
    lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock)
    segments: list[TurnSegment] = dataclasses.field(default_factory=list)
    # Chronological (turn, tool_use_ids) log; survives wipe clears of chain.turns.
    turn_log: list[tuple[TurnRecord, list[str]]] = dataclasses.field(default_factory=list)
    checkpoints: list[PromptCheckpoint] = dataclasses.field(default_factory=list)
    checkpoint_by_tool_use_id: dict[str, str] = dataclasses.field(default_factory=dict)
    request_index: int = 0
    resume: ResumeState | None = None
    adapter_cpu_workers: int = 0
    adapter_timings: list[dict[str, Any]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class _CpuCallResult:
    value: Any
    queue_ms: float
    run_ms: float
    error: BaseException | None = None


@dataclasses.dataclass(frozen=True)
class _DecodedRequest:
    body: dict[str, Any]
    # Stable identity of the model-relevant request. Claude Code may change
    # wire-only fields such as model/stream/metadata while retrying the same
    # logical request, so the full wire hash is diagnostic only.
    sha256: str
    wire_sha256: str
    wire_tool_use_ids: tuple[str, ...]
    wire_tool_result_ids: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class _PreparedFreshRequest:
    chat_messages: list[dict]
    tools_schema: list[dict] | None
    system_hash: str
    msg_hashes: list[str]
    prompt_ids: list[int]
    checkpoint: PromptCheckpoint | None = None


@dataclasses.dataclass(frozen=True)
class _ResumeToolResultUpdate:
    translated_messages: list[dict]
    completed_tool_use_ids: list[str]
    tool_use_echo_missing_ids: list[str] = dataclasses.field(default_factory=list)
    tool_use_echo_mismatches: list[dict[str, str]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class _PreparedResumeRequest:
    prompt_ids: list[int]
    handshake: bool
    first_prompt_sha256: str = ""
    first_prompt_exact: bool = False
    chat_messages: list[dict] | None = None
    completed_tool_use_ids: list[str] = dataclasses.field(default_factory=list)
    tool_use_echo_missing_ids: list[str] = dataclasses.field(default_factory=list)
    tool_use_echo_mismatches: list[dict[str, str]] = dataclasses.field(default_factory=list)
    runtime_tool_names: tuple[str, ...] = ()
    runtime_tool_schemas: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)
    runtime_tool_schema_diff: dict[str, Any] | None = None
    max_tokens_continuation: bool = False
    terminal_ack: bool = False


# ---------------------------------------------------------------------------
# Pure hashing / routing
# ---------------------------------------------------------------------------


def _strip_cache_control(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_cache_control(v) for k, v in obj.items() if k != "cache_control"}
    if isinstance(obj, list):
        return [_strip_cache_control(x) for x in obj]
    return obj


def message_hash(obj: Any) -> str:
    """Stable short hash of a message / system payload (ignores cache_control)."""
    payload = json.dumps(_strip_cache_control(obj), sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def _request_msg_hashes(body: dict) -> list[str]:
    return [message_hash(m) for m in (body.get("messages") or [])]


def _request_system_hash(body: dict, session: Session) -> str:
    if "system" in body:
        return message_hash(body.get("system"))
    return session.main.system_hash


def _request_fingerprints(body: dict, fallback_system_hash: str) -> tuple[list[str], str]:
    """Hash one full wire request exactly once, outside mutable Session state."""
    msg_hashes = _request_msg_hashes(body)
    system_hash = message_hash(body.get("system")) if "system" in body else fallback_system_hash
    return msg_hashes, system_hash


def _continues_prefix(msg_hashes: list[str], system_hash: str, chain: Chain) -> bool:
    return (
        system_hash == chain.system_hash
        and len(msg_hashes) >= chain.seen_msgs
        and msg_hashes[: chain.seen_msgs] == chain.msg_hashes[: chain.seen_msgs]
    )


def _body_has_tool_result(body: dict, tool_use_id: str) -> bool:
    for m in body.get("messages") or []:
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for b in content:
            if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id") == tool_use_id:
                return True
    return False


def freeze_chain(session: Session, chain: Chain, *, kind: str) -> None:
    """Append a TurnSegment from ``chain.turns`` when non-empty; does not clear turns."""
    if chain.turns:
        session.segments.append(make_turn_segment(chain.turns, kind=kind))


def close_subagent_if_done(session: Session, body: dict) -> bool:
    """If main carries the pending dispatch tool_result, freeze sub as ``subagent``.

    Returns True when a sub was closed.
    """
    if not session.pending_dispatch_id or session.active_sub is None:
        return False
    if not _body_has_tool_result(body, session.pending_dispatch_id):
        return False
    freeze_chain(session, session.active_sub, kind="subagent")
    session.active_sub = None
    session.pending_dispatch_id = ""
    return True


def select_chain(
    session: Session,
    body: dict,
    *,
    msg_hashes: list[str] | None = None,
    req_system_hash: str | None = None,
) -> tuple[Chain, bool, Kind]:
    """Decide which chain this turn operates on and classify the request.

    Side effects:
    - Closing a finished sub freezes a ``subagent`` segment and clears ``active_sub``.
    - A ``wipe`` against a chain that already has turns freezes a ``wipe`` segment
      (turns stay on the chain until :func:`commit_request` clears them).

    Returns ``(target, is_sub, kind)`` where kind is ``new`` | ``append`` | ``wipe``.
    """
    close_subagent_if_done(session, body)

    if msg_hashes is None:
        msg_hashes = _request_msg_hashes(body)
    if req_system_hash is None:
        req_system_hash = _request_system_hash(body, session)

    if session.active_sub is None:
        target, is_sub = session.main, False
    else:
        main_continues = _continues_prefix(msg_hashes, req_system_hash, session.main)
        target, is_sub = (session.main, False) if main_continues else (session.active_sub, True)

    if target.seen_msgs == 0:
        kind: Kind = "new"
    elif _continues_prefix(msg_hashes, req_system_hash, target):
        kind = "append"
    else:
        freeze_chain(session, target, kind="wipe")
        kind = "wipe"

    return target, is_sub, kind


def start_sub_chain(session: Session, dispatch_id: str) -> None:
    """Arm a fresh sub chain and remember the tool_use_id that will close it."""
    session.pending_dispatch_id = dispatch_id
    if session.active_sub is None:
        session.active_sub = Chain()


def commit_request(target: Chain, body: dict, kind: Kind) -> None:
    """Update chain fingerprints (and chat_messages) after accepting a request.

    ``new`` / ``wipe`` replace chat state and clear turns; ``append`` extends.
    """
    all_msgs = body.get("messages") or []
    if kind == "append":
        translated = anth._translate_messages(all_msgs[target.seen_msgs :], None)
        target.chat_messages.extend(translated)
    else:
        target.chat_messages = anth._translate_messages(all_msgs, body.get("system"))
        if "system" in body:
            target.system_hash = message_hash(body.get("system"))
        target.turns.clear()

    target.seen_msgs = len(all_msgs)
    target.msg_hashes = [message_hash(m) for m in all_msgs]
    if target.tools_schema is None:
        target.tools_schema = anth._tools_to_chat_tools(body.get("tools"))


def _render_prompt_ids_value(
    tokenizer: Any,
    chat_messages: list[dict],
    tools_schema: list[dict] | None,
) -> list[int]:
    enc = tokenizer.apply_chat_template(
        chat_messages,
        tools=tools_schema,
        tokenize=True,
        add_generation_prompt=True,
    )
    ids = enc["input_ids"] if hasattr(enc, "__getitem__") and "input_ids" in enc else enc
    return list(ids)


def _prepare_fresh_request(
    *,
    tokenizer: Any,
    tokenizer_fingerprint_value: dict[str, Any],
    target: Chain,
    body: dict,
    kind: Kind,
    msg_hashes: list[str],
    request_system_hash: str,
    capture_prompt_checkpoint: bool,
    request_index: int,
    sampling_defaults: dict[str, Any],
    max_context_tokens: int,
    is_sub: bool,
) -> _PreparedFreshRequest:
    """Translate and render a request without mutating its Chain or Session."""
    all_msgs = body.get("messages") or []
    if kind == "append":
        translated = anth._translate_messages(all_msgs[target.seen_msgs :], None)
        chat_messages = list(target.chat_messages) + translated
        system_hash = target.system_hash
    else:
        chat_messages = anth._translate_messages(all_msgs, body.get("system"))
        system_hash = request_system_hash if "system" in body else target.system_hash

    tools_schema = target.tools_schema
    if tools_schema is None:
        tools_schema = anth._tools_to_chat_tools(body.get("tools"))
    prompt_ids = _render_prompt_ids_value(tokenizer, chat_messages, tools_schema)

    checkpoint = None
    if capture_prompt_checkpoint:
        checkpoint = _build_prompt_checkpoint(
            request_index=request_index,
            chat_messages=chat_messages,
            tools_schema=tools_schema,
            body=body,
            prompt_ids=prompt_ids,
            sampling_defaults=sampling_defaults,
            max_context_tokens=max_context_tokens,
            tokenizer_fingerprint_value=tokenizer_fingerprint_value,
            is_sub=is_sub,
            request_kind=kind,
        )
    return _PreparedFreshRequest(
        chat_messages=chat_messages,
        tools_schema=tools_schema,
        system_hash=system_hash,
        msg_hashes=list(msg_hashes),
        prompt_ids=prompt_ids,
        checkpoint=checkpoint,
    )


def _apply_prepared_fresh_request(
    session: Session,
    target: Chain,
    prepared: _PreparedFreshRequest,
    kind: Kind,
) -> None:
    """Commit a fully computed request in a short event-loop critical section."""
    target.chat_messages = prepared.chat_messages
    if kind != "append":
        target.system_hash = prepared.system_hash
        target.turns.clear()
    target.seen_msgs = len(prepared.msg_hashes)
    target.msg_hashes = prepared.msg_hashes
    target.tools_schema = prepared.tools_schema
    if prepared.checkpoint is not None:
        session.checkpoints.append(prepared.checkpoint)
        session.request_index += 1


def commit_fingerprint(target: Chain, body: dict, kind: Kind) -> None:
    """Routing-only commit: update hashes / seen_msgs without translation.

    Useful for unit tests that do not need chat_messages.
    """
    all_msgs = body.get("messages") or []
    if kind != "append":
        if "system" in body:
            target.system_hash = message_hash(body.get("system"))
        target.turns.clear()
    target.seen_msgs = len(all_msgs)
    target.msg_hashes = [message_hash(m) for m in all_msgs]


def append_turn(target: Chain, turn: TurnRecord) -> None:
    target.turns.append(turn)


def record_turn(
    session: Session,
    target: Chain,
    turn: TurnRecord,
    *,
    tool_use_ids: list[str] | None = None,
    n_tool_uses: int | None = None,
) -> None:
    """Append to the active chain and the session-wide chronological turn log.

    Prefer ``tool_use_ids`` (minted Anthropic ids). ``n_tool_uses`` is accepted
    only for older unit tests that do not mint ids.
    """
    append_turn(target, turn)
    ids = [str(x) for x in (tool_use_ids or []) if str(x).strip()]
    if not ids and n_tool_uses:
        # Test helper: synthesize stable ids when callers only know the count.
        ids = [f"toolu_test_{i}" for i in range(int(n_tool_uses))]
    session.turn_log.append((turn, ids))


def drain_session_segments(session: Session) -> list[TokenSegment]:
    """Freeze remaining sub as ``subagent`` and main as ``final``, then merge."""
    if session.active_sub is not None and session.active_sub.turns:
        freeze_chain(session, session.active_sub, kind="subagent")
        session.active_sub = None
    if session.main.turns:
        freeze_chain(session, session.main, kind="final")
        session.main.turns.clear()
    segments = merge_turn_segments(session.segments)
    summary = adapter_timing_summary(session)
    if not summary:
        return segments
    return [
        dataclasses.replace(segment, metadata={**segment.metadata, **summary})
        for segment in segments
    ]


_ADAPTER_TIMING_FIELDS = (
    "adapter_loop_delay_ms",
    "adapter_session_lock_wait_ms",
    "adapter_json_ms",
    "adapter_hash_ms",
    "adapter_prepare_tokenize_ms",
    "adapter_parse_ms",
    "adapter_tool_validate_ms",
    "adapter_cpu_queue_ms",
    "adapter_cpu_ms",
    "adapter_sglang_e2e_ms",
    "adapter_total_ms",
)


def adapter_timing_summary(session: Session) -> dict[str, Any]:
    """Compact per-session adapter timings suitable for Sample metadata."""
    if not session.adapter_timings:
        return {}
    out: dict[str, Any] = {
        "adapter_turn_count": len(session.adapter_timings),
        "adapter_failed_request_count": sum(
            int(str(timing.get("status") or "") != "ok")
            for timing in session.adapter_timings
        ),
        "adapter_cpu_workers": int(session.adapter_cpu_workers or 0),
    }
    for key in _ADAPTER_TIMING_FIELDS:
        values = [float(timing.get(key) or 0.0) for timing in session.adapter_timings]
        out[f"{key}_mean"] = sum(values) / len(values)
        out[f"{key}_max"] = max(values)
    return out


def _build_prompt_checkpoint(
    *,
    request_index: int,
    chat_messages: list[dict],
    tools_schema: list[dict] | None,
    body: dict,
    prompt_ids: list[int],
    sampling_defaults: dict[str, Any],
    max_context_tokens: int,
    tokenizer_fingerprint_value: dict[str, Any],
    is_sub: bool,
    request_kind: Kind,
) -> PromptCheckpoint:
    """Build a detached checkpoint without mutating Session state."""
    prompt_hash = prompt_ids_sha256(prompt_ids)
    source_uses, source_results = _wire_tool_ids(body)
    return PromptCheckpoint(
        checkpoint_id=f"{'sub' if is_sub else 'main'}-{request_index}-{prompt_hash[:16]}",
        prompt_ids=list(prompt_ids),
        prompt_sha256=prompt_hash,
        chat_messages=copy.deepcopy(chat_messages),
        tools_schema=copy.deepcopy(tools_schema),
        tools_sha256=canonical_sha256(tools_schema),
        generation_config=_generation_config_values(
            body,
            sampling_defaults=sampling_defaults,
            max_context_tokens=max_context_tokens,
        ),
        tokenizer_fingerprint=copy.deepcopy(tokenizer_fingerprint_value),
        chain_kind="sub" if is_sub else "main",
        request_kind=request_kind,
        request_index=request_index,
        source_tool_use_ids=source_uses,
        source_tool_result_ids=source_results,
    )


def make_prompt_checkpoint(
    session: Session,
    target: Chain,
    body: dict,
    prompt_ids: list[int],
    *,
    is_sub: bool,
    request_kind: Kind,
) -> PromptCheckpoint:
    """Snapshot the exact prompt state immediately before model generation."""
    checkpoint = _build_prompt_checkpoint(
        request_index=session.request_index,
        chat_messages=target.chat_messages,
        tools_schema=target.tools_schema,
        body=body,
        prompt_ids=prompt_ids,
        sampling_defaults=session.sampling_defaults,
        max_context_tokens=session.max_context_tokens,
        tokenizer_fingerprint_value={},  # adapter-owned caller fills this
        is_sub=is_sub,
        request_kind=request_kind,
    )
    session.checkpoints.append(checkpoint)
    session.request_index += 1
    return checkpoint


def link_checkpoint_tool_uses(session: Session, checkpoint: PromptCheckpoint, blocks: list[dict]) -> None:
    """Attach generated wire tool ids to the checkpoint that produced them."""
    uses = [
        block
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id")
    ]
    ids = [str(block.get("id")) for block in uses]
    checkpoint.generated_tool_use_ids = ids
    checkpoint.generated_tool_use_names = {
        str(block.get("id")): str(block.get("name") or "") for block in uses
    }
    for tool_use_id in ids:
        prior = session.checkpoint_by_tool_use_id.get(tool_use_id)
        if prior and prior != checkpoint.checkpoint_id:
            raise RuntimeError(f"tool_use id {tool_use_id!r} belongs to multiple checkpoints")
        session.checkpoint_by_tool_use_id[tool_use_id] = checkpoint.checkpoint_id


def _wire_tool_blocks(body: dict) -> tuple[list[dict], list[dict]]:
    uses: list[dict] = []
    results: list[dict] = []
    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and block.get("id"):
                uses.append(block)
            elif block.get("type") == "tool_result" and block.get("tool_use_id"):
                results.append(block)
    return uses, results


def _tool_schema_names(schema: list[dict] | None) -> tuple[str, ...]:
    names: list[str] = []
    for tool in schema or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if not isinstance(function, dict) or not function.get("name"):
            continue
        names.append(str(function["name"]))
    return tuple(names)


def _runtime_tool_schema_diff(
    checkpoint_schema: list[dict] | None,
    runtime_schema: list[dict] | None,
) -> dict[str, Any] | None:
    """Return a compact audit diff without making runtime history authoritative."""
    if canonical_sha256(runtime_schema) == canonical_sha256(checkpoint_schema):
        return None

    checkpoint_names = _tool_schema_names(checkpoint_schema)
    runtime_names = _tool_schema_names(runtime_schema)
    checkpoint_by_name = {
        str(tool.get("function", {}).get("name")): tool.get("function")
        for tool in checkpoint_schema or []
        if isinstance(tool, dict)
        and isinstance(tool.get("function"), dict)
        and tool["function"].get("name")
    }
    runtime_by_name = {
        str(tool.get("function", {}).get("name")): tool.get("function")
        for tool in runtime_schema or []
        if isinstance(tool, dict)
        and isinstance(tool.get("function"), dict)
        and tool["function"].get("name")
    }
    checkpoint_set = set(checkpoint_by_name)
    runtime_set = set(runtime_by_name)
    changed = sorted(
        name
        for name in checkpoint_set & runtime_set
        if canonical_sha256(checkpoint_by_name[name])
        != canonical_sha256(runtime_by_name[name])
    )
    return {
        "checkpoint_sha256": canonical_sha256(checkpoint_schema),
        "runtime_sha256": canonical_sha256(runtime_schema),
        "missing_names": sorted(checkpoint_set - runtime_set),
        "added_names": sorted(runtime_set - checkpoint_set),
        "changed_names": changed,
        "order_changed": checkpoint_names != runtime_names,
    }


def _runtime_tool_schemas(runtime_schema: list[dict] | None) -> dict[str, dict[str, Any]]:
    """Return detached JSON Schemas keyed by the Claude Code runtime tool name."""
    return {
        str(tool["function"]["name"]): copy.deepcopy(tool["function"].get("parameters") or {})
        for tool in runtime_schema or []
        if isinstance(tool, dict)
        and isinstance(tool.get("function"), dict)
        and tool["function"].get("name")
    }


def _validate_generated_runtime_tool_calls(
    blocks: list[dict],
    runtime_tool_schemas: dict[str, dict[str, Any]],
) -> tuple[list[str], list[dict[str, str]]]:
    """Audit generated tool names and inputs against Claude Code's contract."""
    unavailable: set[str] = set()
    invalid: list[dict[str, str]] = []
    for block in blocks:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        name = str(block.get("name") or "")
        schema = runtime_tool_schemas.get(name)
        if schema is None:
            unavailable.add(name)
            continue
        try:
            validator_cls = jsonschema_validators.validator_for(schema)
            validator_cls.check_schema(schema)
            errors = sorted(
                validator_cls(schema).iter_errors(block.get("input")),
                key=lambda error: (
                    tuple(str(part) for part in error.absolute_path),
                    str(error.validator),
                    error.message,
                ),
            )
        except jsonschema_exceptions.SchemaError as error:
            invalid.append(
                {
                    "tool_use_id": str(block.get("id") or ""),
                    "name": name,
                    "path": "$schema",
                    "validator": "schema",
                    "message": str(error.message)[:240],
                }
            )
            continue
        for error in errors[:3]:
            path = ".".join(str(part) for part in error.absolute_path)
            invalid.append(
                {
                    "tool_use_id": str(block.get("id") or ""),
                    "name": name,
                    "path": path or "$",
                    "validator": str(error.validator or "unknown"),
                    "message": str(error.message)[:240],
                }
            )
    return sorted(unavailable), invalid


def _known_resume_tool_ids(resume: ResumeState) -> set[str]:
    known = set(resume.checkpoint.source_tool_use_ids)
    known.update(resume.checkpoint.source_tool_result_ids)
    known.update(resume.checkpoint.generated_tool_use_ids)
    known.update(resume.completed_tool_use_ids)
    return known


def _validate_resume_wire_ids_without_pending(resume: ResumeState, body: dict) -> None:
    """Fail closed on new tool ids while accepting Claude Code's old replay."""
    known = _known_resume_tool_ids(resume)
    use_blocks, result_blocks = _wire_tool_blocks(body)
    for block in use_blocks:
        tool_id = str(block["id"])
        if tool_id not in known:
            raise ValueError(f"unexpected tool_use id in resumed request: {tool_id}")
    for block in result_blocks:
        tool_id = str(block["tool_use_id"])
        if tool_id not in known:
            raise ValueError(f"unexpected tool_result id in resumed request: {tool_id}")


def _prepare_resume_tool_result_update(
    resume: ResumeState,
    body: dict,
) -> _ResumeToolResultUpdate:
    """Validate replayed tool results and return a detached state update."""
    if not resume.pending_tool_uses:
        raise ValueError("received tool-result update without pending resumed tool calls")

    expected = {str(block["id"]): block for block in resume.pending_tool_uses}
    expected_ids = list(expected)
    known_old = _known_resume_tool_ids(resume)

    use_blocks, result_blocks = _wire_tool_blocks(body)
    seen_uses: dict[str, list[dict]] = {}
    seen_results: dict[str, list[dict]] = {}
    for block in use_blocks:
        tool_id = str(block["id"])
        if tool_id not in expected and tool_id not in known_old:
            raise ValueError(f"unexpected tool_use id in resumed request: {tool_id}")
        if tool_id in expected:
            seen_uses.setdefault(tool_id, []).append(block)
    for block in result_blocks:
        tool_id = str(block["tool_use_id"])
        if tool_id not in expected and tool_id not in known_old:
            raise ValueError(f"unexpected tool_result id in resumed request: {tool_id}")
        if tool_id in expected:
            seen_results.setdefault(tool_id, []).append(block)

    ordered_results: list[dict] = []
    echo_missing_ids: list[str] = []
    echo_mismatches: list[dict[str, str]] = []
    for tool_id in expected_ids:
        echoed = seen_uses.get(tool_id, [])
        results = seen_results.get(tool_id, [])
        if len(echoed) > 1:
            raise ValueError(f"expected at most one echo for tool_use {tool_id}, got {len(echoed)}")
        if len(results) != 1:
            raise ValueError(f"expected one result for tool_use {tool_id}, got {len(results)}")
        expected_block = expected[tool_id]
        if not echoed:
            # Claude Code does not always replay the assistant tool_use block.
            # The adapter already owns that exact model output; the sole
            # authoritative continuation input is the result bound to its
            # freshly generated id.
            echo_missing_ids.append(tool_id)
        else:
            actual_block = echoed[0]
            expected_input_sha256 = canonical_sha256(expected_block.get("input") or {})
            actual_input_sha256 = canonical_sha256(actual_block.get("input") or {})
            if (
                actual_block.get("name") != expected_block.get("name")
                or actual_input_sha256 != expected_input_sha256
            ):
                # Claude Code may normalize the assistant tool payload before
                # replay. Keep the difference observable, but never replace the
                # adapter-owned model output with this non-authoritative echo.
                echo_mismatches.append(
                    {
                        "tool_use_id": tool_id,
                        "expected_name": str(expected_block.get("name") or ""),
                        "actual_name": str(actual_block.get("name") or ""),
                        "expected_input_sha256": expected_input_sha256,
                        "actual_input_sha256": actual_input_sha256,
                    }
                )
        ordered_results.append(copy.deepcopy(results[0]))

    translated = anth._translate_messages(
        [{"role": "user", "content": ordered_results}],
        None,
    )
    return _ResumeToolResultUpdate(
        translated_messages=translated,
        completed_tool_use_ids=expected_ids,
        tool_use_echo_missing_ids=echo_missing_ids,
        tool_use_echo_mismatches=echo_mismatches,
    )


def _record_tool_use_echo_missing(resume: ResumeState, tool_ids: list[str]) -> None:
    if not tool_ids:
        return
    resume.tool_use_echo_missing_count += len(tool_ids)
    resume.tool_use_echo_mismatch_count += len(tool_ids)
    logger.warning(
        "[anthropic_segmented] Claude Code omitted non-authoritative tool_use echo; "
        "using adapter-owned assistant payload ids=%s",
        tool_ids,
    )


def _record_tool_use_echo_mismatches(
    resume: ResumeState,
    mismatches: list[dict[str, str]],
) -> None:
    if not mismatches:
        return
    resume.tool_use_echo_payload_mismatch_count += len(mismatches)
    resume.tool_use_echo_mismatch_count += len(mismatches)
    logger.warning(
        "[anthropic_segmented] non-authoritative Claude Code tool_use echo differs; "
        "keeping adapter-generated assistant payload mismatches=%s",
        mismatches,
    )


def _record_runtime_tool_schema_diff(
    resume: ResumeState,
    diff: dict[str, Any] | None,
) -> None:
    if diff is None:
        return
    resume.runtime_tool_schema_mismatch_count += 1
    logger.warning(
        "[anthropic_segmented] Claude Code runtime tool schema differs from checkpoint; "
        "keeping checkpoint schema authoritative diff=%s",
        diff,
    )


def _record_invalid_generated_runtime_tool_inputs(
    resume: ResumeState,
    invalid: list[dict[str, str]],
) -> None:
    """Record malformed model calls while preserving normal CC error handling."""
    if not invalid:
        return
    invalid_tool_ids = {item.get("tool_use_id", "") for item in invalid}
    resume.generated_runtime_tool_input_invalid_count += len(invalid_tool_ids)
    logger.warning(
        "[anthropic_segmented] resumed model generated tool input outside Claude Code "
        "runtime schema; forwarding it so Claude Code can return its normal error "
        "tool_result invalid=%s",
        invalid,
    )


def _record_unavailable_generated_runtime_tools(
    resume: ResumeState,
    tool_names: list[str],
) -> None:
    """Record unknown model tools while preserving normal CC error handling."""
    if not tool_names:
        return
    resume.generated_runtime_tool_unavailable_count += len(tool_names)
    logger.warning(
        "[anthropic_segmented] resumed model generated tools unavailable in the "
        "Claude Code runtime schema; forwarding them so Claude Code can return "
        "its normal error tool_result tools=%s",
        tool_names,
    )


def consume_resume_tool_results(session: Session, body: dict) -> None:
    """Validate Claude Code's echo, then append only new tool results.

    The full replayed HTTP history is deliberately not translated.  It is not
    authoritative and was the source of token drift.  Old ids are accepted only
    when they were present in the checkpoint; every newly generated tool call
    must be echoed exactly once and receive exactly one result.
    """
    resume = session.resume
    if resume is None:
        raise RuntimeError("resume state is missing")
    update = _prepare_resume_tool_result_update(resume, body)
    session.main.chat_messages.extend(update.translated_messages)
    resume.completed_tool_use_ids.extend(update.completed_tool_use_ids)
    _record_tool_use_echo_missing(resume, update.tool_use_echo_missing_ids)
    _record_tool_use_echo_mismatches(resume, update.tool_use_echo_mismatches)
    resume.pending_tool_uses.clear()


def _prepare_resume_request(
    *,
    tokenizer: Any,
    session: Session,
    body: dict,
) -> _PreparedResumeRequest:
    """Validate and render a resumed request without mutating Session state."""
    resume = session.resume
    if resume is None:
        raise RuntimeError("resume state is missing")
    checkpoint = resume.checkpoint

    request_tools = anth._tools_to_chat_tools(body.get("tools"))
    runtime_schema_diff = _runtime_tool_schema_diff(checkpoint.tools_schema, request_tools)
    runtime_schemas = _runtime_tool_schemas(request_tools)
    runtime_tool_names = tuple(runtime_schemas)
    if canonical_sha256(_generation_config(body, session)) != canonical_sha256(checkpoint.generation_config):
        raise ValueError("generation settings differ from the Stage-1 checkpoint")

    if resume.handshake_pending:
        prompt_ids = list(checkpoint.prompt_ids)
        first_hash = prompt_ids_sha256(prompt_ids)
        return _PreparedResumeRequest(
            prompt_ids=prompt_ids,
            handshake=True,
            first_prompt_sha256=first_hash,
            first_prompt_exact=first_hash == checkpoint.prompt_sha256,
            runtime_tool_names=runtime_tool_names,
            runtime_tool_schemas=runtime_schemas,
            runtime_tool_schema_diff=runtime_schema_diff,
        )

    if not resume.pending_tool_uses:
        _validate_resume_wire_ids_without_pending(resume, body)
        if resume.last_stop_reason == "end_turn":
            return _PreparedResumeRequest(
                prompt_ids=[],
                handshake=False,
                runtime_tool_names=runtime_tool_names,
                runtime_tool_schemas=runtime_schemas,
                runtime_tool_schema_diff=runtime_schema_diff,
                terminal_ack=True,
            )
        if resume.last_stop_reason == "max_tokens":
            chat_messages = list(session.main.chat_messages)
            prompt_ids = _render_prompt_ids_value(
                tokenizer,
                chat_messages,
                session.main.tools_schema,
            )
            return _PreparedResumeRequest(
                prompt_ids=prompt_ids,
                handshake=False,
                chat_messages=chat_messages,
                runtime_tool_names=runtime_tool_names,
                runtime_tool_schemas=runtime_schemas,
                runtime_tool_schema_diff=runtime_schema_diff,
                max_tokens_continuation=True,
            )
        raise ValueError(
            "received resumed request without pending tool calls "
            f"after stop_reason={resume.last_stop_reason or 'unknown'}"
        )

    update = _prepare_resume_tool_result_update(resume, body)
    chat_messages = list(session.main.chat_messages) + update.translated_messages
    prompt_ids = _render_prompt_ids_value(tokenizer, chat_messages, session.main.tools_schema)
    return _PreparedResumeRequest(
        prompt_ids=prompt_ids,
        handshake=False,
        chat_messages=chat_messages,
        completed_tool_use_ids=update.completed_tool_use_ids,
        tool_use_echo_missing_ids=update.tool_use_echo_missing_ids,
        tool_use_echo_mismatches=update.tool_use_echo_mismatches,
        runtime_tool_names=runtime_tool_names,
        runtime_tool_schemas=runtime_schemas,
        runtime_tool_schema_diff=runtime_schema_diff,
    )


def _apply_prepared_resume_request(session: Session, prepared: _PreparedResumeRequest) -> None:
    resume = session.resume
    if resume is None:
        raise RuntimeError("resume state is missing")
    _record_runtime_tool_schema_diff(resume, prepared.runtime_tool_schema_diff)
    if prepared.handshake:
        resume.first_prompt_sha256 = prepared.first_prompt_sha256
        resume.first_prompt_exact = prepared.first_prompt_exact
        if not resume.first_prompt_exact:
            raise ValueError("first resumed prompt hash differs from the Stage-1 checkpoint")
        resume.handshake_pending = False
        resume.handshake_validated = True
        return

    if prepared.terminal_ack:
        resume.post_end_turn_ack_count += 1
        logger.warning(
            "[anthropic_segmented] acknowledging Claude Code request after resumed end_turn "
            "without another model sample count=%d",
            resume.post_end_turn_ack_count,
        )
        return

    if prepared.chat_messages is None:
        raise RuntimeError("prepared resumed request is missing chat_messages")
    session.main.chat_messages = prepared.chat_messages
    resume.completed_tool_use_ids.extend(prepared.completed_tool_use_ids)
    _record_tool_use_echo_missing(resume, prepared.tool_use_echo_missing_ids)
    _record_tool_use_echo_mismatches(resume, prepared.tool_use_echo_mismatches)
    if prepared.max_tokens_continuation:
        resume.max_tokens_continuation_count += 1
        logger.warning(
            "[anthropic_segmented] continuing resumed branch after max_tokens from "
            "adapter-authoritative state count=%d",
            resume.max_tokens_continuation_count,
        )
    resume.pending_tool_uses.clear()


_RESUME_REPLAY_BODY_KEYS = (
    "system",
    "messages",
    "tools",
    "tool_choice",
    "thinking",
    "output_config",
    "context_management",
    *_GENERATION_BODY_KEYS,
)


def _resume_replay_sha256(body: dict[str, Any]) -> str:
    """Hash only fields that identify one logical Claude Code model request."""
    prompt_request: dict[str, Any] = {}
    for key in _RESUME_REPLAY_BODY_KEYS:
        if key not in body:
            continue
        value = _strip_cache_control(body[key])
        if key == "tools" and isinstance(value, list):
            # Tool ordering is a wire detail for this adapter: chat-template
            # rendering uses the checkpoint schema, while execution uses the
            # current runtime schema keyed by name.
            value = sorted(
                value,
                key=lambda tool: (
                    str(tool.get("name") or "") if isinstance(tool, dict) else "",
                    canonical_sha256(tool),
                ),
            )
        prompt_request[key] = value
    return canonical_sha256(prompt_request)


def _decode_request_body(raw: bytes) -> _DecodedRequest:
    body = json.loads(raw)
    if not isinstance(body, dict):
        raise TypeError("Anthropic request body must be a JSON object")
    anth._fold_mid_list_system_into_user(body)
    use_ids, result_ids = _wire_tool_ids(body)
    return _DecodedRequest(
        body=body,
        sha256=_resume_replay_sha256(body),
        wire_sha256=canonical_sha256(body),
        wire_tool_use_ids=tuple(use_ids),
        wire_tool_result_ids=tuple(result_ids),
    )


def _snapshot_resume_mutations(session: Session) -> tuple[ResumeState, list[dict]]:
    """Take a cheap transaction snapshot without copying checkpoint tokens."""
    resume = session.resume
    if resume is None:
        raise RuntimeError("resume state is missing")
    snapshot = copy.copy(resume)
    snapshot.pending_tool_uses = copy.deepcopy(resume.pending_tool_uses)
    snapshot.completed_tool_use_ids = list(resume.completed_tool_use_ids)
    return snapshot, list(session.main.chat_messages)


def _restore_resume_mutations(
    session: Session,
    snapshot: tuple[ResumeState, list[dict]],
) -> None:
    session.resume, session.main.chat_messages = snapshot


def _cache_resume_response(
    resume: ResumeState,
    *,
    request_sha256: str,
    wire_sha256: str,
    body: dict,
    blocks: list[dict],
    stop_reason: str,
    in_tok: int,
    out_tok: int,
) -> _ResumeResponseCache:
    cache = _ResumeResponseCache(
        request_sha256=request_sha256,
        wire_sha256=wire_sha256,
        response_body=anth._render_response(
            body,
            copy.deepcopy(blocks),
            stop_reason,
            in_tok,
            out_tok,
        ),
    )
    resume.cached_response = cache
    return cache


async def _render_cached_resume_response(
    request: web.Request,
    body: dict,
    cache: _ResumeResponseCache,
) -> web.StreamResponse:
    response_body = cache.response_body
    stream = body.get("stream") is True or "text/event-stream" in request.headers.get(
        "Accept", ""
    )
    if stream:
        usage = response_body.get("usage") or {}
        return await anth._render_stream(
            request,
            response_body.get("content") or [],
            str(response_body.get("stop_reason") or "end_turn"),
            int(usage.get("input_tokens") or 0),
            int(usage.get("output_tokens") or 0),
            message_id=str(response_body.get("id") or "") or None,
        )
    return web.json_response(copy.deepcopy(response_body))


def _note_cpu_timing(
    timing: dict[str, Any],
    field: str,
    result: _CpuCallResult,
) -> None:
    timing[field] = float(result.run_ms)
    timing[f"{field[:-3]}_queue_ms"] = float(result.queue_ms)
    timing["adapter_cpu_ms"] = float(timing.get("adapter_cpu_ms") or 0.0) + result.run_ms
    timing["adapter_cpu_queue_ms"] = (
        float(timing.get("adapter_cpu_queue_ms") or 0.0) + result.queue_ms
    )


def _serialize_prompt_checkpoints(
    checkpoints: list[PromptCheckpoint],
) -> list[dict[str, Any]]:
    return [checkpoint.to_dict() for checkpoint in checkpoints]


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class SegmentedAnthropicAdapter:
    """Anthropic Messages HTTP adapter with explicit subagent/wipe/final segments."""

    logger = logger
    log_prefix = "anthropic_segmented"
    max_token_keys = ("max_tokens",)
    stop_keys = ("stop_sequences",)

    def __init__(
        self,
        *,
        tokenizer,
        sglang_url,
        tool_parser=None,
        reasoning_parser=None,
        cpu_workers: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.sglang_url = sglang_url.rstrip("/") if isinstance(sglang_url, str) else sglang_url
        self.tool_parser = tool_parser
        self.reasoning_parser = reasoning_parser
        if cpu_workers is None:
            raw_workers = (os.environ.get("SLIME_ADAPTER_CPU_WORKERS") or "").strip()
            cpu_workers = int(raw_workers) if raw_workers else min(4, max(1, os.cpu_count() or 1))
        if int(cpu_workers) <= 0:
            raise ValueError("cpu_workers must be positive")
        self.cpu_workers = int(cpu_workers)
        self._cpu_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.cpu_workers,
            thread_name_prefix="cc-adapter-cpu",
        )
        self._cpu_executor_closed = False
        self._tokenizer_fingerprint = tokenizer_fingerprint(self.tokenizer)
        self.store: dict[str, Session] = {}
        self.inflight: dict[str, set[asyncio.Task]] = {}
        self.closed: set[str] = set()
        self._sglang_http_session = None
        self._sglang_http_loop = None
        self.app = web.Application(client_max_size=64 * 1024 * 1024)
        self.app.on_cleanup.append(self._cleanup_resources)
        self.app.router.add_get("/health", self._health)
        self.app.router.add_get("/healthz", self._health)
        self.app.router.add_get("/v1/models", self._health)
        self.app.router.add_post("/v1/messages", self._handle_messages)
        self.app.router.add_post("/v1/messages/count_tokens", self._count_tokens)

    async def _run_cpu(self, fn: Callable[[], Any]) -> _CpuCallResult:
        """Run pure CPU work on a bounded pool and wait safely on cancellation.

        ``run_in_executor`` cannot stop a running function.  The handler keeps
        its per-session lock until that function has completed, even when the
        HTTP client disconnects, so a worker can never outlive the state
        snapshot it is reading.
        """
        if self._cpu_executor_closed:
            raise RuntimeError("adapter CPU executor is closed")
        loop = asyncio.get_running_loop()
        submitted = time.perf_counter()

        def invoke() -> _CpuCallResult:
            started = time.perf_counter()
            try:
                value = fn()
            except BaseException as error:  # propagate on the event-loop thread
                return _CpuCallResult(
                    value=None,
                    queue_ms=(started - submitted) * 1000.0,
                    run_ms=(time.perf_counter() - started) * 1000.0,
                    error=error,
                )
            return _CpuCallResult(
                value=value,
                queue_ms=(started - submitted) * 1000.0,
                run_ms=(time.perf_counter() - started) * 1000.0,
            )

        future = loop.run_in_executor(self._cpu_executor, invoke)
        try:
            result = await asyncio.shield(future)
        except asyncio.CancelledError as cancelled:
            # A worker cannot be force-cancelled.  Do not release session.lock
            # while it may still be reading that session's prompt state.
            while not future.done():
                try:
                    await asyncio.shield(future)
                except asyncio.CancelledError:
                    continue
            try:
                completed = future.result()
                if completed.error is not None:
                    self.logger.warning(
                        "[%s] CPU task failed after request cancellation: %s",
                        self.log_prefix,
                        completed.error,
                    )
            except BaseException:
                self.logger.exception("[%s] CPU task failed after cancellation", self.log_prefix)
            raise cancelled
        if result.error is not None:
            raise result.error
        return result

    async def _cleanup_resources(self, _: web.Application) -> None:
        await close_sglang_client(self)
        if self._cpu_executor_closed:
            return
        self._cpu_executor_closed = True
        shutdown = functools.partial(
            self._cpu_executor.shutdown,
            wait=True,
            cancel_futures=True,
        )
        await asyncio.to_thread(shutdown)

    def _new_session(
        self,
        *,
        sampling_defaults: dict | None,
        max_context_tokens: int,
        capture_prompt_checkpoints: bool,
        resume_checkpoint: PromptCheckpoint | dict[str, Any] | None,
    ) -> Session:
        """Construct and validate one Session without publishing it in store."""
        session = Session(
            sampling_defaults=dict(sampling_defaults or {}),
            max_context_tokens=int(max_context_tokens or 0),
            capture_prompt_checkpoints=bool(capture_prompt_checkpoints),
            adapter_cpu_workers=self.cpu_workers,
        )
        if resume_checkpoint is None:
            return session

        checkpoint = PromptCheckpoint.from_dict(resume_checkpoint)
        if checkpoint.chain_kind != "main":
            raise ValueError("token-exact resume currently supports only the main Claude Code chain")
        if checkpoint.request_kind == "wipe":
            raise ValueError("token-exact resume from a compact/wipe request is not supported")
        if prompt_ids_sha256(checkpoint.prompt_ids) != checkpoint.prompt_sha256:
            raise ValueError("resume checkpoint prompt_ids hash mismatch")
        if canonical_sha256(checkpoint.tools_schema) != checkpoint.tools_sha256:
            raise ValueError("resume checkpoint tools schema hash mismatch")
        if canonical_sha256(self._tokenizer_fingerprint) != canonical_sha256(checkpoint.tokenizer_fingerprint):
            raise ValueError("resume checkpoint tokenizer fingerprint mismatch")

        # Re-rendering is an integrity check. Generation still uses saved ids
        # directly, so Stage-2 bootstrap text cannot alter the first prompt.
        rendered = _render_prompt_ids_value(
            self.tokenizer,
            checkpoint.chat_messages,
            checkpoint.tools_schema,
        )
        if rendered != checkpoint.prompt_ids:
            raise ValueError(
                "resume checkpoint no longer renders to the saved prompt ids "
                f"(saved={checkpoint.prompt_sha256[:12]} rendered={prompt_ids_sha256(rendered)[:12]})"
            )
        session.main.chat_messages = copy.deepcopy(checkpoint.chat_messages)
        session.main.tools_schema = copy.deepcopy(checkpoint.tools_schema)
        session.request_index = checkpoint.request_index
        session.resume = ResumeState(checkpoint=checkpoint)
        return session

    def open_session(
        self,
        sid: str,
        *,
        sampling_defaults: dict | None = None,
        max_context_tokens: int = 0,
        capture_prompt_checkpoints: bool = False,
        resume_checkpoint: PromptCheckpoint | dict[str, Any] | None = None,
    ) -> None:
        if sid in self.store:
            raise ValueError(f"session_id {sid!r} already exists; sids must be unique per agent run")
        session = self._new_session(
            sampling_defaults=sampling_defaults,
            max_context_tokens=max_context_tokens,
            capture_prompt_checkpoints=capture_prompt_checkpoints,
            resume_checkpoint=resume_checkpoint,
        )
        self.store[sid] = session

    async def open_session_async(
        self,
        sid: str,
        *,
        sampling_defaults: dict | None = None,
        max_context_tokens: int = 0,
        capture_prompt_checkpoints: bool = False,
        resume_checkpoint: PromptCheckpoint | dict[str, Any] | None = None,
    ) -> None:
        """Open a session without blocking the caller on resume validation."""
        if resume_checkpoint is None:
            self.open_session(
                sid,
                sampling_defaults=sampling_defaults,
                max_context_tokens=max_context_tokens,
                capture_prompt_checkpoints=capture_prompt_checkpoints,
            )
            return
        if sid in self.store:
            raise ValueError(f"session_id {sid!r} already exists; sids must be unique per agent run")
        result = await self._run_cpu(
            functools.partial(
                self._new_session,
                sampling_defaults=sampling_defaults,
                max_context_tokens=max_context_tokens,
                capture_prompt_checkpoints=capture_prompt_checkpoints,
                resume_checkpoint=resume_checkpoint,
            )
        )
        if sid in self.store:
            raise ValueError(f"session_id {sid!r} was opened concurrently")
        self.store[sid] = result.value
        self.logger.info(
            "[%s] sid=%s resume checkpoint validation queue=%.1fms cpu=%.1fms",
            self.log_prefix,
            sid,
            result.queue_ms,
            result.run_ms,
        )

    def export_prompt_checkpoints(self, sid: str, *, clear: bool = False) -> list[dict[str, Any]]:
        """Return a detached, JSON-serializable checkpoint list before finish."""
        session = self.store.get(sid)
        if session is None:
            raise KeyError(f"unknown session_id {sid!r}")
        payload = [checkpoint.to_dict() for checkpoint in session.checkpoints]
        if clear:
            session.checkpoints.clear()
            session.checkpoint_by_tool_use_id.clear()
        return payload

    async def export_prompt_checkpoints_async(
        self,
        sid: str,
        *,
        clear: bool = False,
    ) -> list[dict[str, Any]]:
        """Serialize large checkpoint histories without blocking rollout work."""
        session = self.store.get(sid)
        if session is None:
            raise KeyError(f"unknown session_id {sid!r}")
        if self._cpu_executor_closed:
            return self.export_prompt_checkpoints(sid, clear=clear)
        result = await self._run_cpu(
            functools.partial(
                _serialize_prompt_checkpoints,
                list(session.checkpoints),
            )
        )
        if clear:
            session.checkpoints.clear()
            session.checkpoint_by_tool_use_id.clear()
        return result.value

    def checkpoint_for_tool_use_id(self, sid: str, tool_use_id: str) -> dict[str, Any] | None:
        session = self.store.get(sid)
        if session is None:
            raise KeyError(f"unknown session_id {sid!r}")
        checkpoint_id = session.checkpoint_by_tool_use_id.get(str(tool_use_id))
        if checkpoint_id is None:
            return None
        for checkpoint in session.checkpoints:
            if checkpoint.checkpoint_id == checkpoint_id:
                return checkpoint.to_dict()
        raise RuntimeError(f"checkpoint index points to missing id {checkpoint_id!r}")

    def resume_status(self, sid: str) -> dict[str, Any]:
        session = self.store.get(sid)
        if session is None:
            raise KeyError(f"unknown session_id {sid!r}")
        resume = session.resume
        if resume is None:
            return {"mode": "fresh"}
        return {
            "mode": "token_exact",
            "checkpoint_id": resume.checkpoint.checkpoint_id,
            "prompt_sha256": resume.checkpoint.prompt_sha256,
            "first_prompt_sha256": resume.first_prompt_sha256,
            "first_prompt_exact": resume.first_prompt_exact,
            "handshake_validated": resume.handshake_validated,
            "exact_request_count": resume.exact_request_count,
            "pending_tool_use_ids": [str(block["id"]) for block in resume.pending_tool_uses],
            "completed_tool_use_ids": list(resume.completed_tool_use_ids),
            "last_stop_reason": resume.last_stop_reason,
            "tool_use_echo_mismatch_count": resume.tool_use_echo_mismatch_count,
            "tool_use_echo_missing_count": resume.tool_use_echo_missing_count,
            "tool_use_echo_payload_mismatch_count": resume.tool_use_echo_payload_mismatch_count,
            "runtime_tool_schema_mismatch_count": resume.runtime_tool_schema_mismatch_count,
            "generated_runtime_tool_unavailable_count": (
                resume.generated_runtime_tool_unavailable_count
            ),
            "generated_runtime_tool_input_invalid_count": (
                resume.generated_runtime_tool_input_invalid_count
            ),
            "max_tokens_continuation_count": resume.max_tokens_continuation_count,
            "post_end_turn_ack_count": resume.post_end_turn_ack_count,
            "request_replay_count": resume.request_replay_count,
            "error": resume.error,
        }

    async def shutdown_session(self, sid: str, *, wait_timeout: float = 5.0) -> None:
        self.closed.add(sid)
        tasks = [t for t in self.inflight.pop(sid, ()) if not t.done()]
        if not tasks:
            return

        async def _drain() -> None:
            _, pending = await asyncio.wait(tasks, timeout=wait_timeout)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        loop = tasks[0].get_loop()
        try:
            await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(_drain(), loop))
        except Exception:
            self.logger.exception("[%s] sid=%s shutdown drain failed", self.log_prefix, sid)

    async def finish_session(self, sid: str, *, wait_timeout: float = 5.0) -> list[TokenSegment]:
        await self.shutdown_session(sid, wait_timeout=wait_timeout)
        session = self.store.pop(sid, None)
        if session is None:
            return []
        if self._cpu_executor_closed:
            return drain_session_segments(session)
        result = await self._run_cpu(functools.partial(drain_session_segments, session))
        return result.value

    @staticmethod
    async def _health(_: web.Request) -> web.Response:
        return web.Response(status=200, text="ok")

    @staticmethod
    async def _count_tokens(request: web.Request) -> web.Response:
        await request.read()
        return web.json_response({"input_tokens": 0})

    def _session_id(self, request: web.Request) -> str:
        return sid_from_bearer(request) or (request.headers.get("X-Api-Key") or "").strip() or "default"

    def _render_prompt_ids(self, target: Chain) -> list[int]:
        return _render_prompt_ids_value(
            self.tokenizer,
            target.chat_messages,
            target.tools_schema,
        )

    def _parse_and_blocks(self, target: Chain, output_ids: list[int], finish: str):
        raw = self.tokenizer.decode(output_ids, skip_special_tokens=False) if output_ids else ""
        parsed = parse_model_output(
            raw,
            tools_schema=target.tools_schema,
            tool_parser_name=self.tool_parser,
            reasoning_parser_name=self.reasoning_parser,
        )
        blocks, stop_reason, manager_message = anth._build_reply_parts(parsed, finish)
        dispatch_id = ""
        for b in blocks:
            if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") in SUBAGENT_TOOLS:
                dispatch_id = str(b.get("id") or "")
        return blocks, stop_reason, manager_message, dispatch_id

    def _resume_prompt_ids(self, session: Session, body: dict) -> list[int]:
        prepared = _prepare_resume_request(
            tokenizer=self.tokenizer,
            session=session,
            body=body,
        )
        _apply_prepared_resume_request(session, prepared)
        return prepared.prompt_ids

    @staticmethod
    def _resume_failure(session: Session, error: Exception | str) -> web.Response:
        message = str(error)
        if session.resume is not None:
            session.resume.error = message
        logger.warning("[anthropic_segmented] token-exact resume rejected: %s", message)
        return web.json_response(
            {"type": "error", "error": {"type": "invalid_request_error", "message": message}},
            status=409,
        )

    async def _handle_messages(self, request: web.Request) -> web.StreamResponse:
        total_started = time.perf_counter()
        timing: dict[str, Any] = {
            "adapter_cpu_ms": 0.0,
            "adapter_cpu_queue_ms": 0.0,
        }
        loop_probe = time.perf_counter()
        await asyncio.sleep(0)
        timing["adapter_loop_delay_ms"] = (time.perf_counter() - loop_probe) * 1000.0

        raw_body = await request.read()
        json_result = await self._run_cpu(functools.partial(_decode_request_body, raw_body))
        _note_cpu_timing(timing, "adapter_json_ms", json_result)
        decoded_request = json_result.value
        body = decoded_request.body
        request_sha256 = decoded_request.sha256
        request_wire_sha256 = decoded_request.wire_sha256
        sid = self._session_id(request)
        if sid in self.closed:
            return web.Response(status=503, text="session closed")

        session = self.store.setdefault(
            sid,
            Session(adapter_cpu_workers=self.cpu_workers),
        )
        if not session.adapter_cpu_workers:
            session.adapter_cpu_workers = self.cpu_workers
        task = asyncio.current_task()
        if task is not None:
            self.inflight.setdefault(sid, set()).add(task)
        status = "error"
        resume_response_cache: _ResumeResponseCache | None = None
        try:
            lock_started = time.perf_counter()
            async with session.lock:
                timing["adapter_session_lock_wait_ms"] = (
                    time.perf_counter() - lock_started
                ) * 1000.0
                checkpoint: PromptCheckpoint | None = None
                if session.resume is not None:
                    target, is_sub = session.main, False
                    if session.resume.error:
                        status = "resume_rejected"
                        return self._resume_failure(session, session.resume.error)
                    cached = session.resume.cached_response
                    if cached is not None and cached.request_sha256 == request_sha256:
                        session.resume.request_replay_count += 1
                        logger.warning(
                            "[anthropic_segmented] replaying cached response for duplicate "
                            "Claude Code request count=%d request_sha256=%s "
                            "wire_changed=%s cached_wire_sha256=%s wire_sha256=%s",
                            session.resume.request_replay_count,
                            request_sha256,
                            cached.wire_sha256 != request_wire_sha256,
                            cached.wire_sha256,
                            request_wire_sha256,
                        )
                        status = "ok"
                        return await _render_cached_resume_response(request, body, cached)
                    if cached is not None and session.resume.pending_tool_uses:
                        pending_ids = {
                            str(block.get("id") or "")
                            for block in session.resume.pending_tool_uses
                            if isinstance(block, dict) and block.get("id")
                        }
                        matching_result_ids = [
                            tool_id
                            for tool_id in decoded_request.wire_tool_result_ids
                            if tool_id in pending_ids
                        ]
                        unknown_result_ids = [
                            tool_id
                            for tool_id in decoded_request.wire_tool_result_ids
                            if tool_id not in _known_resume_tool_ids(session.resume)
                        ]
                        if not matching_result_ids and not unknown_result_ids:
                            # An outstanding tool response is the only legal
                            # predecessor of its result. If Claude Code sends a
                            # request with none of those results, it cannot
                            # advance the session: the prior response was lost
                            # or its retry body was rewritten. Re-serve the
                            # exact cached tool ids instead of sampling again or
                            # turning transport behavior into missing_result.
                            session.resume.request_replay_count += 1
                            logger.warning(
                                "[anthropic_segmented] replaying cached response from pending "
                                "state count=%d pending_ids=%s request_sha_changed=%s "
                                "wire_changed=%s cached_request_sha256=%s request_sha256=%s "
                                "cached_wire_sha256=%s wire_sha256=%s",
                                session.resume.request_replay_count,
                                sorted(pending_ids),
                                cached.request_sha256 != request_sha256,
                                cached.wire_sha256 != request_wire_sha256,
                                cached.request_sha256,
                                request_sha256,
                                cached.wire_sha256,
                                request_wire_sha256,
                            )
                            status = "ok"
                            return await _render_cached_resume_response(request, body, cached)
                    try:
                        prepare_result = await self._run_cpu(
                            functools.partial(
                                _prepare_resume_request,
                                tokenizer=self.tokenizer,
                                session=session,
                                body=body,
                            )
                        )
                        _note_cpu_timing(
                            timing,
                            "adapter_prepare_tokenize_ms",
                            prepare_result,
                        )
                        prepared_resume = prepare_result.value
                        resume_mutation_snapshot = _snapshot_resume_mutations(session)
                        try:
                            _apply_prepared_resume_request(session, prepared_resume)
                        except (TypeError, ValueError, RuntimeError):
                            _restore_resume_mutations(session, resume_mutation_snapshot)
                            raise
                        if prepared_resume.terminal_ack:
                            blocks = [{"type": "text", "text": ""}]
                            stop_reason = "end_turn"
                            in_tok = out_tok = 0
                            resume_response_cache = _cache_resume_response(
                                session.resume,
                                request_sha256=request_sha256,
                                wire_sha256=request_wire_sha256,
                                body=body,
                                blocks=blocks,
                                stop_reason=stop_reason,
                                in_tok=in_tok,
                                out_tok=out_tok,
                            )
                            status = "ok"
                            return await _render_cached_resume_response(
                                request,
                                body,
                                resume_response_cache,
                            )
                        prompt_ids = prepared_resume.prompt_ids
                    except (TypeError, ValueError, RuntimeError) as error:
                        status = "resume_rejected"
                        return self._resume_failure(session, error)
                else:
                    hash_result = await self._run_cpu(
                        functools.partial(
                            _request_fingerprints,
                            body,
                            session.main.system_hash,
                        )
                    )
                    _note_cpu_timing(timing, "adapter_hash_ms", hash_result)
                    msg_hashes, request_system_hash = hash_result.value
                    target, is_sub, kind = select_chain(
                        session,
                        body,
                        msg_hashes=msg_hashes,
                        req_system_hash=request_system_hash,
                    )
                    prepare_result = await self._run_cpu(
                        functools.partial(
                            _prepare_fresh_request,
                            tokenizer=self.tokenizer,
                            tokenizer_fingerprint_value=self._tokenizer_fingerprint,
                            target=target,
                            body=body,
                            kind=kind,
                            msg_hashes=msg_hashes,
                            request_system_hash=request_system_hash,
                            capture_prompt_checkpoint=session.capture_prompt_checkpoints,
                            request_index=session.request_index,
                            sampling_defaults=session.sampling_defaults,
                            max_context_tokens=session.max_context_tokens,
                            is_sub=is_sub,
                        )
                    )
                    _note_cpu_timing(
                        timing,
                        "adapter_prepare_tokenize_ms",
                        prepare_result,
                    )
                    prepared_fresh = prepare_result.value
                    _apply_prepared_fresh_request(session, target, prepared_fresh, kind)
                    prompt_ids = prepared_fresh.prompt_ids
                    checkpoint = prepared_fresh.checkpoint

                try:
                    sglang_started = time.perf_counter()
                    turn = await call_sglang_generate(
                        prompt_ids,
                        session,
                        body,
                        adapter=self,
                        session_id=sid,
                    )
                    timing["adapter_sglang_e2e_ms"] = (
                        time.perf_counter() - sglang_started
                    ) * 1000.0
                    parse_result = await self._run_cpu(
                        functools.partial(
                            self._parse_and_blocks,
                            target,
                            turn.output_ids,
                            turn.finish_reason,
                        )
                    )
                except BaseException:
                    if session.resume is not None:
                        _restore_resume_mutations(session, resume_mutation_snapshot)
                    raise
                _note_cpu_timing(timing, "adapter_parse_ms", parse_result)
                blocks, stop_reason, manager_message, dispatch_id = parse_result.value
                tool_use_ids = [
                    str(block.get("id") or "")
                    for block in blocks
                    if isinstance(block, dict)
                    and block.get("type") == "tool_use"
                    and block.get("id")
                ]
                if checkpoint is not None:
                    link_checkpoint_tool_uses(session, checkpoint, blocks)
                if session.resume is not None:
                    validate_result = await self._run_cpu(
                        functools.partial(
                            _validate_generated_runtime_tool_calls,
                            blocks,
                            prepared_resume.runtime_tool_schemas,
                        )
                    )
                    _note_cpu_timing(timing, "adapter_tool_validate_ms", validate_result)
                    unavailable_tool_names, invalid_tool_inputs = validate_result.value
                    if unavailable_tool_names:
                        _record_unavailable_generated_runtime_tools(
                            session.resume,
                            unavailable_tool_names,
                        )
                    if invalid_tool_inputs:
                        _record_invalid_generated_runtime_tool_inputs(
                            session.resume,
                            invalid_tool_inputs,
                        )
                    if dispatch_id:
                        status = "resume_rejected"
                        return self._resume_failure(
                            session,
                            "Task/Agent subagent dispatch is not supported in a token-exact resumed branch",
                        )
                    session.main.chat_messages.append(copy.deepcopy(manager_message))
                    session.resume.pending_tool_uses = [
                        copy.deepcopy(block)
                        for block in blocks
                        if isinstance(block, dict) and block.get("type") == "tool_use"
                    ]
                    session.resume.last_stop_reason = stop_reason
                    session.resume.exact_request_count += 1
                record_turn(session, target, turn, tool_use_ids=tool_use_ids)
                if session.resume is None and dispatch_id and not is_sub:
                    start_sub_chain(session, dispatch_id)
                in_tok, out_tok = len(prompt_ids), len(turn.output_ids)
                if session.resume is not None:
                    resume_response_cache = _cache_resume_response(
                        session.resume,
                        request_sha256=request_sha256,
                        wire_sha256=request_wire_sha256,
                        body=body,
                        blocks=blocks,
                        stop_reason=stop_reason,
                        in_tok=in_tok,
                        out_tok=out_tok,
                    )

            if resume_response_cache is not None:
                response = await _render_cached_resume_response(
                    request,
                    body,
                    resume_response_cache,
                )
            elif body.get("stream") is True or "text/event-stream" in request.headers.get(
                "Accept", ""
            ):
                response = await anth._render_stream(
                    request,
                    blocks,
                    stop_reason,
                    in_tok,
                    out_tok,
                )
            else:
                response = web.json_response(
                    anth._render_response(body, blocks, stop_reason, in_tok, out_tok)
                )
            status = "ok"
            return response
        finally:
            timing["status"] = status
            timing["adapter_total_ms"] = (time.perf_counter() - total_started) * 1000.0
            for field in _ADAPTER_TIMING_FIELDS:
                timing.setdefault(field, 0.0)
            session.adapter_timings.append(timing)
            self.logger.debug(
                "[%s] sid=%s status=%s total=%.1fms queue=%.1fms cpu=%.1fms "
                "sglang=%.1fms lock=%.1fms",
                self.log_prefix,
                sid,
                status,
                timing["adapter_total_ms"],
                timing["adapter_cpu_queue_ms"],
                timing["adapter_cpu_ms"],
                timing["adapter_sglang_e2e_ms"],
                timing["adapter_session_lock_wait_ms"],
            )
            if task is not None:
                self.inflight.get(sid, set()).discard(task)


__all__ = [
    "SUBAGENT_TOOLS",
    "Chain",
    "PromptCheckpoint",
    "ResumeState",
    "SegmentedAnthropicAdapter",
    "Session",
    "append_turn",
    "canonical_sha256",
    "close_subagent_if_done",
    "commit_fingerprint",
    "commit_request",
    "consume_resume_tool_results",
    "drain_session_segments",
    "freeze_chain",
    "link_checkpoint_tool_uses",
    "make_prompt_checkpoint",
    "message_hash",
    "prompt_ids_sha256",
    "record_turn",
    "select_chain",
    "start_sub_chain",
    "tokenizer_fingerprint",
]
