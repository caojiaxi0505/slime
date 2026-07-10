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
import dataclasses
import hashlib
import json
import logging
from typing import Any

from aiohttp import web

from slime.agent.adapters import anthropic as anth
from slime.agent.adapters.common import call_sglang_generate, sid_from_bearer
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
    lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock)
    segments: list[TurnSegment] = dataclasses.field(default_factory=list)


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


def select_chain(session: Session, body: dict) -> tuple[Chain, bool, Kind]:
    """Decide which chain this turn operates on and classify the request.

    Side effects:
    - Closing a finished sub freezes a ``subagent`` segment and clears ``active_sub``.
    - A ``wipe`` against a chain that already has turns freezes a ``wipe`` segment
      (turns stay on the chain until :func:`commit_request` clears them).

    Returns ``(target, is_sub, kind)`` where kind is ``new`` | ``append`` | ``wipe``.
    """
    close_subagent_if_done(session, body)

    msg_hashes = _request_msg_hashes(body)
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


def drain_session_segments(session: Session) -> list[TokenSegment]:
    """Freeze remaining sub as ``subagent`` and main as ``final``, then merge."""
    if session.active_sub is not None and session.active_sub.turns:
        freeze_chain(session, session.active_sub, kind="subagent")
        session.active_sub = None
    if session.main.turns:
        freeze_chain(session, session.main, kind="final")
        session.main.turns.clear()
    return merge_turn_segments(session.segments)


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
    ) -> None:
        self.tokenizer = tokenizer
        self.sglang_url = sglang_url.rstrip("/") if isinstance(sglang_url, str) else sglang_url
        self.tool_parser = tool_parser
        self.reasoning_parser = reasoning_parser
        self.store: dict[str, Session] = {}
        self.inflight: dict[str, set[asyncio.Task]] = {}
        self.closed: set[str] = set()
        self.app = web.Application(client_max_size=64 * 1024 * 1024)
        self.app["adapter"] = self
        self.app.router.add_get("/health", self._health)
        self.app.router.add_get("/healthz", self._health)
        self.app.router.add_get("/v1/models", self._health)
        self.app.router.add_post("/v1/messages", self._handle_messages)
        self.app.router.add_post("/v1/messages/count_tokens", self._count_tokens)

    def open_session(
        self,
        sid: str,
        *,
        sampling_defaults: dict | None = None,
        max_context_tokens: int = 0,
    ) -> None:
        if sid in self.store:
            raise ValueError(f"session_id {sid!r} already exists; sids must be unique per agent run")
        self.store[sid] = Session(
            sampling_defaults=dict(sampling_defaults or {}),
            max_context_tokens=int(max_context_tokens or 0),
        )

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
        return drain_session_segments(session)

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
        enc = self.tokenizer.apply_chat_template(
            target.chat_messages,
            tools=target.tools_schema,
            tokenize=True,
            add_generation_prompt=True,
        )
        ids = enc["input_ids"] if hasattr(enc, "__getitem__") and "input_ids" in enc else enc
        return list(ids)

    def _parse_and_blocks(self, target: Chain, output_ids: list[int], finish: str):
        raw = self.tokenizer.decode(output_ids, skip_special_tokens=False) if output_ids else ""
        parsed = parse_model_output(
            raw,
            tools_schema=target.tools_schema,
            tool_parser_name=self.tool_parser,
            reasoning_parser_name=self.reasoning_parser,
        )
        blocks, stop_reason, _manager_message = anth._build_reply_parts(parsed, finish)
        dispatch_id = ""
        for b in blocks:
            if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") in SUBAGENT_TOOLS:
                dispatch_id = str(b.get("id") or "")
        return blocks, stop_reason, dispatch_id

    async def _handle_messages(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        sid = self._session_id(request)
        if sid in self.closed:
            return web.Response(status=503, text="session closed")

        anth._fold_mid_list_system_into_user(body)
        session = self.store.setdefault(sid, Session())
        task = asyncio.current_task()
        self.inflight.setdefault(sid, set()).add(task)
        try:
            async with session.lock:
                target, is_sub, kind = select_chain(session, body)
                commit_request(target, body, kind)
                prompt_ids = self._render_prompt_ids(target)
                turn = await call_sglang_generate(
                    prompt_ids,
                    session,
                    body,
                    adapter=self,
                    session_id=sid,
                )
                blocks, stop_reason, dispatch_id = self._parse_and_blocks(
                    target, turn.output_ids, turn.finish_reason
                )
                append_turn(target, turn)
                if dispatch_id and not is_sub:
                    start_sub_chain(session, dispatch_id)
                in_tok, out_tok = len(prompt_ids), len(turn.output_ids)

            stream = body.get("stream") is True or "text/event-stream" in request.headers.get("Accept", "")
            if stream:
                return await anth._render_stream(request, blocks, stop_reason, in_tok, out_tok)
            return web.json_response(anth._render_response(body, blocks, stop_reason, in_tok, out_tok))
        finally:
            self.inflight.get(sid, set()).discard(task)


__all__ = [
    "SUBAGENT_TOOLS",
    "Chain",
    "SegmentedAnthropicAdapter",
    "Session",
    "append_turn",
    "close_subagent_if_done",
    "commit_fingerprint",
    "commit_request",
    "drain_session_segments",
    "freeze_chain",
    "message_hash",
    "select_chain",
    "start_sub_chain",
]
