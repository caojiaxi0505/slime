"""Anthropic Messages proxy for collecting Claude Code SFT data from a remote
OpenAI-compatible model.

This adapter is intentionally SFT-only: it does not build RL token segments or
logprobs.  It translates Claude Code's Anthropic ``/v1/messages`` requests to a
remote ``/v1/chat/completions`` endpoint, returns Anthropic-shaped responses to
Claude Code, and writes one JSONL row per real model turn containing the exact
messages/tools sent to the remote model and the assistant response received.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import dataclasses
import hashlib
import json
import logging
import os
import random
import secrets
import threading
import time
from typing import Any

import aiohttp
from aiohttp import web

from slime.agent.adapters import anthropic as anth
from slime.agent.adapters.anthropic_segmented import (
    _append_jsonl,
    _env_flag,
    _json_safe,
    canonical_sha256,
    prompt_ids_sha256,
)
from slime.agent.adapters.common import flatten_content, sid_from_bearer
from slime.agent.aiohttp_threaded import FilteredAccessLogger, run_app_in_thread

logger = logging.getLogger(__name__)


class TeacherContextIntegrityError(ValueError):
    """A resumed Teacher request cannot be derived from authoritative state."""

    bucket = "context_integrity"


@dataclasses.dataclass
class _CachedTeacherReply:
    request_wire_sha256: str
    blocks: list[dict[str, Any]]
    stop_reason: str
    input_tokens: int
    output_tokens: int


@dataclasses.dataclass
class _PendingTeacherCommit:
    """One remote reply that must be durable before session state advances."""

    request_wire_sha256: str
    record: dict[str, Any]
    messages: list[dict[str, Any]]
    pending_tool_uses: list[dict[str, Any]]
    completed_tool_use_ids: list[str]
    known_wire_tool_ids: set[str]
    last_stop_reason: str
    turn_index: int
    cached_reply: _CachedTeacherReply


@dataclasses.dataclass
class _TeacherResumeSession:
    checkpoint_id: str
    checkpoint_prompt_sha256: str
    checkpoint_messages_sha256: str
    messages: list[dict[str, Any]]
    tools_schema: list[dict[str, Any]] | None
    log_path: str
    attempt_id: str
    turn_index: int = 0
    pending_tool_uses: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    completed_tool_use_ids: list[str] = dataclasses.field(default_factory=list)
    known_wire_tool_ids: set[str] = dataclasses.field(default_factory=set)
    last_stop_reason: str = ""
    error: str = ""
    cached_reply: _CachedTeacherReply | None = None
    pending_commit: _PendingTeacherCommit | None = None
    lock: asyncio.Lock | None = None


def _wire_tool_blocks(body: dict[str, Any], block_type: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        out.extend(
            copy.deepcopy(block)
            for block in content
            if isinstance(block, dict) and block.get("type") == block_type
        )
    return out


def _wire_tool_ids(body: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for block in _wire_tool_blocks(body, "tool_use"):
        if block.get("id"):
            ids.add(str(block["id"]))
    for block in _wire_tool_blocks(body, "tool_result"):
        if block.get("tool_use_id"):
            ids.add(str(block["tool_use_id"]))
    return ids


def _tool_result_messages(
    state: _TeacherResumeSession,
    body: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Extract only results for the adapter-owned pending Teacher tool calls."""
    expected = [str(block.get("id") or "") for block in state.pending_tool_uses]
    if not expected or any(not value for value in expected):
        raise TeacherContextIntegrityError("resumed Teacher state has no valid pending tool ids")

    by_id: dict[str, list[dict[str, Any]]] = {}
    for block in _wire_tool_blocks(body, "tool_result"):
        tool_id = str(block.get("tool_use_id") or "")
        if tool_id in expected:
            by_id.setdefault(tool_id, []).append(block)
            continue
        if tool_id and tool_id not in state.known_wire_tool_ids and tool_id not in state.completed_tool_use_ids:
            raise TeacherContextIntegrityError(
                f"unknown tool_result id in resumed Teacher request: {tool_id}"
            )

    messages: list[dict[str, Any]] = []
    for tool_id in expected:
        matches = by_id.get(tool_id, [])
        if len(matches) != 1:
            raise TeacherContextIntegrityError(
                f"expected exactly one tool_result for {tool_id}, got {len(matches)}"
            )
        messages.append(
            {
                "role": "tool",
                "content": flatten_content(matches[0].get("content")),
            }
        )
    return messages, expected


def _attempt_claim_path(log_path: str) -> str:
    return f"{log_path}.claim"


def _claim_attempt_log_path(log_dir: str, sid: str) -> tuple[str, str]:
    """Reserve a unique attempt without precreating its JSONL data file.

    The training filesystem rejects append-open on a zero-byte file that was
    first created with ``O_EXCL``.  A separate, non-empty claim file preserves
    atomic name allocation while allowing the first JSONL append to create the
    data file normally.
    """
    os.makedirs(log_dir, exist_ok=True)
    sid_hash = hashlib.sha256(str(sid).encode("utf-8")).hexdigest()[:16]
    for _ in range(32):
        attempt_id = secrets.token_hex(8)
        path = os.path.join(log_dir, f"{sid_hash}.{attempt_id}.sft_turns.jsonl")
        claim_path = _attempt_claim_path(path)
        try:
            fd = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            continue
        try:
            payload = f"attempt_id={attempt_id}\n".encode("ascii")
            offset = 0
            while offset < len(payload):
                written = os.write(fd, payload[offset:])
                if written <= 0:
                    raise OSError("short write while reserving Teacher SFT attempt")
                offset += written
        except BaseException:
            os.close(fd)
            try:
                os.unlink(claim_path)
            except OSError:
                pass
            raise
        os.close(fd)
        return path, attempt_id
    raise RuntimeError(f"failed to allocate a unique Teacher SFT attempt for sid hash={sid_hash}")


def _discard_unwritten_attempt(log_path: str) -> None:
    for path in (log_path, _attempt_claim_path(log_path)):
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def _preflight_sft_log_dir(log_dir: str) -> None:
    """Exercise the exact claim → append → read sequence on the live mount."""
    sid = f"__teacher_sft_preflight__:{os.getpid()}:{secrets.token_hex(8)}"
    path, attempt_id = _claim_attempt_log_path(log_dir, sid)
    record = {
        "record_type": "teacher_sft_filesystem_preflight",
        "attempt_id": attempt_id,
    }
    try:
        _append_jsonl(path, record)
        with open(path, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if rows != [record]:
            raise RuntimeError(
                f"Teacher SFT filesystem preflight readback mismatch: {path}"
            )
    finally:
        _discard_unwritten_attempt(path)


def _messages_to_openai(body: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    system = body.get("system")
    if system:
        messages.append({"role": "system", "content": flatten_content(system)})

    for msg in body.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        blocks = content if isinstance(content, list) else [{"type": "text", "text": flatten_content(content)}]
        if role == "user":
            text_parts: list[str] = []
            for block in blocks:
                if not isinstance(block, dict):
                    text_parts.append(flatten_content(block))
                    continue
                if block.get("type") == "tool_result":
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": str(block.get("tool_use_id") or ""),
                            "content": flatten_content(block.get("content")),
                        }
                    )
                elif block.get("type") == "text":
                    text_parts.append(str(block.get("text") or ""))
                else:
                    text_parts.append(flatten_content(block))
            if text_parts:
                messages.append({"role": "user", "content": "".join(text_parts)})
        elif role == "assistant":
            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text_parts.append(str(block.get("text") or ""))
                elif block.get("type") == "thinking":
                    reasoning_parts.append(str(block.get("thinking") or ""))
                elif block.get("type") == "tool_use":
                    tool_calls.append(
                        {
                            "id": str(block.get("id") or f"toolu_{secrets.token_hex(8)}"),
                            "type": "function",
                            "function": {
                                "name": str(block.get("name") or ""),
                                "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                            },
                        }
                    )
            out: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts)}
            if reasoning_parts:
                out["reasoning_content"] = "".join(reasoning_parts)
            if tool_calls:
                out["tool_calls"] = tool_calls
            messages.append(out)
        elif role == "system":
            messages.append({"role": "system", "content": flatten_content(content)})
    return messages


def _tool_call_input(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str) and arguments.strip():
        try:
            value = json.loads(arguments)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _openai_message_to_anthropic(message: dict[str, Any], finish_reason: str) -> tuple[list[dict], str, dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    reasoning = str(message.get("reasoning_content") or "")
    content = message.get("content")
    text = content if isinstance(content, str) else ""
    if reasoning:
        blocks.append({"type": "thinking", "thinking": reasoning})
    if text:
        blocks.append({"type": "text", "text": text})

    manager_tool_calls: list[dict[str, Any]] = []
    for tool_call in message.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        fn = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
        name = str(fn.get("name") or "")
        if not name:
            continue
        args = _tool_call_input(fn.get("arguments"))
        tool_id = str(tool_call.get("id") or f"toolu_{secrets.token_hex(8)}")
        blocks.append({"type": "tool_use", "id": tool_id, "name": name, "input": args})
        manager_tool_calls.append(
            {
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
        )

    if not blocks:
        blocks.append({"type": "text", "text": ""})

    if manager_tool_calls:
        stop_reason = "tool_use"
    elif finish_reason == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"

    manager_message: dict[str, Any] = {"role": "assistant", "content": text}
    if reasoning:
        manager_message["reasoning_content"] = reasoning
    if manager_tool_calls:
        manager_message["tool_calls"] = manager_tool_calls
    return blocks, stop_reason, manager_message


def _request_hash(value: Any) -> str:
    payload = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _thinking_type(raw: str | None) -> str:
    value = (raw or "").strip().lower()
    if value in {"0", "false", "no", "off", "disabled", "none"}:
        return "disabled"
    return "enabled"


def _retry_delays_from_env() -> list[float]:
    raw = (os.environ.get("SLIME_REMOTE_OPENAI_RETRY_DELAYS_SEC") or "15,30,60").strip()
    out: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = float(part)
        except ValueError:
            continue
        if value > 0:
            out.append(value)
    return out


class RemoteOpenAISFTAdapter:
    log_prefix = "remote_openai_sft_adapter"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        sft_log_dir: str,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 20,
        reasoning_effort: str = "max",
        thinking_type: str | None = None,
        max_turns_per_sid: int | None = None,
        require_registered_sessions: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.sft_log_dir = sft_log_dir
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.reasoning_effort = (reasoning_effort or "max").strip() or "max"
        self.thinking_type = _thinking_type(
            thinking_type if thinking_type is not None else os.environ.get("SLIME_REMOTE_OPENAI_THINKING_TYPE")
        )
        self.include_session_id = _env_flag("SLIME_AGENT_SFT_LOG_INCLUDE_SESSION_ID")
        self.include_wire_request = _env_flag("SLIME_AGENT_SFT_LOG_INCLUDE_WIRE_REQUEST")
        self.max_inflight = int(os.environ.get("SLIME_REMOTE_OPENAI_MAX_INFLIGHT") or "64")
        self.remote_sem = asyncio.Semaphore(self.max_inflight) if self.max_inflight > 0 else None
        self.connector_limit = int(
            os.environ.get("SLIME_REMOTE_OPENAI_CONNECTOR_LIMIT")
            or str(max(0, self.max_inflight * 2))
        )
        self.retry_delays = _retry_delays_from_env()
        # The teacher adapter starts before Stage-1, but real teacher traffic
        # starts only after a failed student trial is ready for relabeling.
        # An optional tiny completion prevents a remote deployment from being
        # marked idle during that intentional gap. It never enters SFT logs or
        # training samples and is disabled when the interval is zero.
        self.keepalive_sec = max(
            0.0, float(os.environ.get("SLIME_REMOTE_OPENAI_KEEPALIVE_SEC") or "0")
        )
        self.keepalive_max_tokens = max(
            1, int(os.environ.get("SLIME_REMOTE_OPENAI_KEEPALIVE_MAX_TOKENS") or "1")
        )
        self.keepalive_inflight = max(
            1, int(os.environ.get("SLIME_REMOTE_OPENAI_KEEPALIVE_INFLIGHT") or "4")
        )
        self._keepalive_task: asyncio.Task | None = None
        self._active_real_requests = 0
        self._last_real_request_monotonic = time.monotonic()
        # Stage-2 teacher relabels stop the harness after a fixed number of
        # teacher turns; the cap is enforced here because Claude Code has no
        # flag for it.
        self.max_turns_per_sid = max_turns_per_sid
        self.require_registered_sessions = bool(require_registered_sessions)
        self.turn_index: dict[str, int] = {}
        self._generic_log_paths: dict[str, str] = {}
        self._resume_sessions: dict[str, _TeacherResumeSession] = {}
        self._resume_registry_lock = threading.Lock()
        self.http: aiohttp.ClientSession | None = None
        self.app = web.Application(client_max_size=128 * 1024 * 1024)
        self.app.on_startup.append(self._startup)
        self.app.on_cleanup.append(self._cleanup)
        self.app.router.add_get("/health", self._health)
        self.app.router.add_get("/healthz", self._health)
        self.app.router.add_post("/v1/messages", self._handle_messages)
        self.app.router.add_post("/v1/messages/count_tokens", self._count_tokens)

    def register_resume_session(self, sid: str, checkpoint: dict[str, Any]) -> str:
        """Register one authoritative checkpoint and return its unique JSONL path."""
        sid = str(sid or "")
        if not sid:
            raise TeacherContextIntegrityError("Teacher resume session id is required")
        if not isinstance(checkpoint, dict):
            raise TeacherContextIntegrityError("Teacher resume checkpoint must be a dict")

        checkpoint_id = str(checkpoint.get("checkpoint_id") or "")
        messages = checkpoint.get("chat_messages")
        tools_schema = checkpoint.get("tools_schema")
        prompt_ids = [int(value) for value in (checkpoint.get("prompt_ids") or [])]
        prompt_sha256 = str(checkpoint.get("prompt_sha256") or "")
        tools_sha256 = str(checkpoint.get("tools_sha256") or "")
        if not checkpoint_id:
            raise TeacherContextIntegrityError("Teacher resume checkpoint_id is required")
        if not isinstance(messages, list):
            raise TeacherContextIntegrityError(
                "Teacher resume checkpoint.chat_messages must be a list"
            )
        if tools_schema is not None and not isinstance(tools_schema, list):
            raise TeacherContextIntegrityError(
                "Teacher resume checkpoint.tools_schema must be a list or null"
            )
        if not prompt_ids or prompt_ids_sha256(prompt_ids) != prompt_sha256:
            raise TeacherContextIntegrityError(
                "Teacher resume checkpoint prompt_ids hash mismatch"
            )
        if canonical_sha256(tools_schema) != tools_sha256:
            raise TeacherContextIntegrityError(
                "Teacher resume checkpoint tools schema hash mismatch"
            )

        log_path, attempt_id = _claim_attempt_log_path(self.sft_log_dir, sid)
        state = _TeacherResumeSession(
            checkpoint_id=checkpoint_id,
            checkpoint_prompt_sha256=prompt_sha256,
            checkpoint_messages_sha256=canonical_sha256(messages),
            messages=copy.deepcopy(messages),
            tools_schema=copy.deepcopy(tools_schema),
            log_path=log_path,
            attempt_id=attempt_id,
        )
        with self._resume_registry_lock:
            if sid in self._resume_sessions:
                _discard_unwritten_attempt(log_path)
                raise TeacherContextIntegrityError(
                    f"Teacher resume session {sid!r} is already registered"
                )
            self._resume_sessions[sid] = state
        return log_path

    def resume_session_status(self, sid: str) -> dict[str, Any]:
        with self._resume_registry_lock:
            state = self._resume_sessions.get(str(sid))
            if state is None:
                raise KeyError(f"Teacher resume session {sid!r} is not registered")
            return {
                "checkpoint_id": state.checkpoint_id,
                "checkpoint_prompt_sha256": state.checkpoint_prompt_sha256,
                "checkpoint_messages_sha256": state.checkpoint_messages_sha256,
                "log_path": state.log_path,
                "attempt_id": state.attempt_id,
                "turn_index": state.turn_index,
                "last_stop_reason": state.last_stop_reason,
                "pending_tool_use_ids": [
                    str(block.get("id") or "") for block in state.pending_tool_uses
                ],
                "persistence_pending": state.pending_commit is not None,
                "persistence_pending_turn_index": (
                    state.pending_commit.turn_index - 1
                    if state.pending_commit is not None
                    else None
                ),
                "error": state.error,
            }

    def close_resume_session(self, sid: str) -> None:
        with self._resume_registry_lock:
            self._resume_sessions.pop(str(sid), None)

    def _resume_session(self, sid: str) -> _TeacherResumeSession | None:
        with self._resume_registry_lock:
            return self._resume_sessions.get(str(sid))

    def _log_path(self, sid: str, state: _TeacherResumeSession | None) -> str:
        if state is not None:
            return state.log_path
        with self._resume_registry_lock:
            path = self._generic_log_paths.get(sid)
            if path is None:
                path, _ = _claim_attempt_log_path(self.sft_log_dir, sid)
                self._generic_log_paths[sid] = path
            return path

    async def _startup(self, _: web.Application) -> None:
        await asyncio.to_thread(_preflight_sft_log_dir, self.sft_log_dir)
        logger.info(
            "[%s] Teacher SFT filesystem preflight passed log_dir=%s",
            self.log_prefix,
            self.sft_log_dir,
        )
        if self.keepalive_sec <= 0:
            return
        self._keepalive_task = asyncio.create_task(
            self._keepalive_loop(), name="remote-openai-sft-keepalive"
        )
        logger.info(
            "[%s] keepalive enabled interval=%.1fs max_tokens=%d inflight=%d",
            self.log_prefix,
            self.keepalive_sec,
            self.keepalive_max_tokens,
            self.keepalive_inflight,
        )

    async def _cleanup(self, _: web.Application) -> None:
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass
            self._keepalive_task = None
        if self.http is not None:
            await self.http.close()
            self.http = None

    async def _client(self) -> aiohttp.ClientSession:
        if self.http is None or self.http.closed:
            timeout = aiohttp.ClientTimeout(total=int(os.environ.get("SLIME_REMOTE_OPENAI_TIMEOUT_SEC") or "1200"))
            connector = aiohttp.TCPConnector(
                limit=self.connector_limit if self.connector_limit > 0 else 0,
                limit_per_host=0,
                ttl_dns_cache=300,
                keepalive_timeout=60,
                enable_cleanup_closed=True,
            )
            self.http = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self.http

    async def _turn_cap_stop_response(
        self,
        request: web.Request,
        body: dict[str, Any],
        sid: str,
    ) -> web.StreamResponse:
        """Stop a capped teacher relabel without using a retryable HTTP error.

        Claude Code retries 429 for several minutes.  A normal Anthropic
        ``end_turn`` response makes the capped teacher finish immediately while
        keeping the logged SFT rows limited to the real remote teacher turns.
        """
        logger.info(
            "[%s] sid=%s reached max_turns_per_sid=%d; returning synthetic end_turn",
            self.log_prefix,
            sid,
            int(self.max_turns_per_sid or 0),
        )
        blocks = [{"type": "text", "text": "Done."}]
        if body.get("stream") is True or "text/event-stream" in request.headers.get("Accept", ""):
            return await anth._render_stream(request, blocks, "end_turn", 0, 0)
        return web.json_response(anth._render_response(body, blocks, "end_turn", 0, 0))

    async def _post_chat_completions(
        self,
        payload: dict[str, Any],
        *,
        retry_delays: list[float] | None = None,
    ) -> tuple[int, str, dict[str, Any] | None]:
        client = await self._client()
        retry_statuses = {408, 409, 425, 429, 500, 502, 503, 504}
        delays = self.retry_delays if retry_delays is None else retry_delays
        attempts = len(delays) + 1
        last_status = 0
        last_text = ""

        for attempt in range(attempts):
            if attempt:
                delay = delays[attempt - 1] + random.uniform(0.0, 2.0)
                logger.warning(
                    "remote OpenAI retry attempt=%d/%d sleep=%.1fs previous_status=%s body=%.300s",
                    attempt + 1,
                    attempts,
                    delay,
                    last_status,
                    last_text,
                )
                await asyncio.sleep(delay)

            try:
                if self.remote_sem is None:
                    resp_ctx = client.post(
                        f"{self.base_url}/v1/chat/completions",
                        json=payload,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                    )
                    async with resp_ctx as resp:
                        text = await resp.text()
                        status = resp.status
                else:
                    async with self.remote_sem:
                        resp_ctx = client.post(
                            f"{self.base_url}/v1/chat/completions",
                            json=payload,
                            headers={
                                "Authorization": f"Bearer {self.api_key}",
                                "Content-Type": "application/json",
                            },
                        )
                        async with resp_ctx as resp:
                            text = await resp.text()
                            status = resp.status
            except Exception as exc:
                last_status = 0
                last_text = f"{type(exc).__name__}: {exc}"
                if attempt < attempts - 1:
                    continue
                return last_status, last_text, None

            last_status = status
            last_text = text
            if status < 400:
                try:
                    return status, text, json.loads(text)
                except json.JSONDecodeError:
                    return status, text, None
            if status not in retry_statuses or attempt >= attempts - 1:
                return status, text, None

        return last_status, last_text, None

    def _keepalive_payload(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": "healthcheck"}],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": self.keepalive_max_tokens,
            "thinking": {"type": "disabled"},
        }

    async def _keepalive_loop(self) -> None:
        """Keep an idle remote deployment warm without creating training data."""
        while True:
            await asyncio.sleep(self.keepalive_sec)
            idle_for = time.monotonic() - self._last_real_request_monotonic
            if self._active_real_requests or idle_for < self.keepalive_sec:
                continue
            try:
                results = await asyncio.gather(
                    *(
                        self._post_chat_completions(
                            self._keepalive_payload(), retry_delays=[]
                        )
                        for _ in range(self.keepalive_inflight)
                    ),
                    return_exceptions=True,
                )
                ok = 0
                for result in results:
                    if isinstance(result, BaseException):
                        logger.warning(
                            "[%s] keepalive exception: %s",
                            self.log_prefix,
                            result,
                        )
                        continue
                    status, text, _ = result
                    if status < 400:
                        ok += 1
                        continue
                    logger.warning(
                        "[%s] keepalive failed status=%d body=%.200s",
                        self.log_prefix,
                        status,
                        text,
                    )
                logger.info(
                    "[%s] keepalive batch complete ok=%d/%d idle_for=%.1fs",
                    self.log_prefix,
                    ok,
                    self.keepalive_inflight,
                    idle_for,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("[%s] keepalive exception: %s", self.log_prefix, exc)

    @staticmethod
    async def _health(_: web.Request) -> web.Response:
        return web.Response(status=200, text="ok")

    @staticmethod
    async def _count_tokens(request: web.Request) -> web.Response:
        await request.read()
        return web.json_response({"input_tokens": 0})

    @staticmethod
    def _session_id(request: web.Request) -> str:
        return sid_from_bearer(request) or (request.headers.get("X-Api-Key") or "").strip() or "default"

    async def _render_cached_reply(
        self,
        request: web.Request,
        body: dict[str, Any],
        cached: _CachedTeacherReply,
    ) -> web.StreamResponse:
        if body.get("stream") is True or "text/event-stream" in request.headers.get("Accept", ""):
            return await anth._render_stream(
                request,
                copy.deepcopy(cached.blocks),
                cached.stop_reason,
                cached.input_tokens,
                cached.output_tokens,
            )
        return web.json_response(
            anth._render_response(
                body,
                copy.deepcopy(cached.blocks),
                cached.stop_reason,
                cached.input_tokens,
                cached.output_tokens,
            )
        )

    @staticmethod
    def _commit_pending_teacher_turn(state: _TeacherResumeSession) -> None:
        """Persist one prepared row, then atomically advance in-memory state."""
        pending = state.pending_commit
        if pending is None:
            raise RuntimeError("Teacher SFT commit requested without a pending record")
        _append_jsonl(state.log_path, pending.record)
        state.messages = pending.messages
        state.pending_tool_uses = pending.pending_tool_uses
        state.completed_tool_use_ids = pending.completed_tool_use_ids
        state.known_wire_tool_ids = pending.known_wire_tool_ids
        state.last_stop_reason = pending.last_stop_reason
        state.turn_index = pending.turn_index
        state.cached_reply = pending.cached_reply
        state.pending_commit = None
        state.error = ""

    @staticmethod
    def _persistence_error_response(
        state: _TeacherResumeSession,
        exc: OSError,
    ) -> web.Response:
        message = f"teacher_sft_persistence:{type(exc).__name__}:{exc}"
        state.error = message
        logger.exception(
            "[remote_openai_sft_adapter] Teacher turn is not committed; "
            "exact request may retry checkpoint=%s attempt=%s turn=%d error=%s",
            state.checkpoint_id,
            state.attempt_id,
            state.turn_index,
            exc,
        )
        return web.json_response(
            {
                "type": "error",
                "error": {"type": "api_error", "message": message},
            },
            status=500,
        )

    @staticmethod
    def _context_error(
        state: _TeacherResumeSession,
        message: str,
    ) -> web.Response:
        state.error = str(message)
        logger.error(
            "[remote_openai_sft_adapter] rejecting resumed Teacher context "
            "checkpoint=%s attempt=%s error=%s",
            state.checkpoint_id,
            state.attempt_id,
            message,
        )
        return web.json_response(
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": f"teacher_context_integrity:{message}",
                },
            },
            status=400,
        )

    @staticmethod
    def _registered_context(
        state: _TeacherResumeSession,
        body: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, list[str]]:
        """Return the exact Teacher payload history without trusting CC replay."""
        if state.turn_index == 0:
            return copy.deepcopy(state.messages), copy.deepcopy(state.tools_schema), []
        if state.pending_tool_uses:
            tool_messages, completed_ids = _tool_result_messages(state, body)
            return (
                copy.deepcopy(state.messages) + tool_messages,
                copy.deepcopy(state.tools_schema),
                completed_ids,
            )
        raise TeacherContextIntegrityError(
            "received another request without pending Teacher tool calls "
            f"after stop_reason={state.last_stop_reason or 'unknown'}"
        )

    async def _handle_messages(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        anth._fold_mid_list_system_into_user(body)
        sid = self._session_id(request)
        state = self._resume_session(sid)
        if self.require_registered_sessions and state is None:
            logger.error(
                "[%s] rejecting unregistered Teacher session sid_hash=%s",
                self.log_prefix,
                hashlib.sha256(sid.encode("utf-8")).hexdigest()[:16],
            )
            return web.json_response(
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "teacher_context_integrity:session_not_registered",
                    },
                },
                status=400,
            )

        if state is not None:
            if state.lock is None:
                state.lock = asyncio.Lock()
            async with state.lock:
                return await self._handle_messages_locked(request, body, sid, state)
        return await self._handle_messages_locked(request, body, sid, None)

    async def _handle_messages_locked(
        self,
        request: web.Request,
        body: dict[str, Any],
        sid: str,
        state: _TeacherResumeSession | None,
    ) -> web.StreamResponse:
        request_wire_sha256 = _request_hash(body)
        if state is not None and state.pending_commit is not None:
            pending = state.pending_commit
            if pending.request_wire_sha256 != request_wire_sha256:
                return self._context_error(
                    state,
                    "received a different request while the previous Teacher "
                    "turn is waiting for durable persistence",
                )
            try:
                self._commit_pending_teacher_turn(state)
            except OSError as exc:
                return self._persistence_error_response(state, exc)
            logger.warning(
                "[%s] persisted and replayed pending Teacher response "
                "checkpoint=%s attempt=%s",
                self.log_prefix,
                state.checkpoint_id,
                state.attempt_id,
            )
            return await self._render_cached_reply(
                request,
                body,
                pending.cached_reply,
            )
        if state is not None and state.cached_reply is not None:
            cached = state.cached_reply
            if cached.request_wire_sha256 == request_wire_sha256:
                logger.warning(
                    "[%s] replaying cached Teacher response checkpoint=%s attempt=%s",
                    self.log_prefix,
                    state.checkpoint_id,
                    state.attempt_id,
                )
                return await self._render_cached_reply(request, body, cached)

        turn_index = state.turn_index if state is not None else self.turn_index.get(sid, 0)
        # Count successful teacher turns only. A remote 429/5xx becomes 502 and
        # must not look like "already took MAX_STEPS" to Claude Code.
        if self.max_turns_per_sid is not None and turn_index >= self.max_turns_per_sid:
            return await self._turn_cap_stop_response(request, body, sid)

        completed_ids: list[str] = []
        try:
            if state is not None:
                messages, tools, completed_ids = self._registered_context(state, body)
            else:
                messages = _messages_to_openai(body)
                tools = anth._tools_to_chat_tools(body.get("tools"))
        except TeacherContextIntegrityError as exc:
            assert state is not None
            return self._context_error(state, str(exc))

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": int(body.get("max_tokens") or 16384),
            "thinking": {"type": self.thinking_type},
        }
        if self.thinking_type != "disabled":
            payload["reasoning_effort"] = self.reasoning_effort
        if tools:
            payload["tools"] = tools
        if body.get("stop_sequences"):
            payload["stop"] = body.get("stop_sequences")

        started = time.time()
        self._active_real_requests += 1
        self._last_real_request_monotonic = time.monotonic()
        try:
            status, text, data = await self._post_chat_completions(payload)
        finally:
            self._active_real_requests -= 1
            self._last_real_request_monotonic = time.monotonic()
        if status >= 400 or data is None:
            logger.warning("remote OpenAI error status=%s body=%.500s", status, text)
            return web.json_response(
                {"type": "error", "error": {"type": "api_error", "message": text[:1000]}},
                status=502,
            )

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        finish_reason = str(choice.get("finish_reason") or "")
        blocks, stop_reason, manager_message = _openai_message_to_anthropic(message, finish_reason)
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        in_tok = int(usage.get("prompt_tokens") or 0)
        out_tok = int(usage.get("completion_tokens") or 0)

        sft_record: dict[str, Any] = {
            "version": 2 if state is not None else 1,
            "created_at_unix": time.time(),
            "session_id_sha256": hashlib.sha256(sid.encode("utf-8")).hexdigest(),
            "turn_index": turn_index,
            "request_sha256": _request_hash(payload),
            "request_wire_sha256": _request_hash(body),
            "provider": "remote_openai",
            "model": self.model,
            "latency_sec": time.time() - started,
            "prompt": {
                "messages": copy.deepcopy(messages),
                "tools_schema": copy.deepcopy(tools),
                "tools_sha256": canonical_sha256(tools),
                "generation_config": {
                    "model": self.model,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "reasoning_effort": payload.get("reasoning_effort"),
                    "thinking": payload["thinking"],
                    "max_tokens": payload["max_tokens"],
                    "context_length": int(os.environ.get("SLIME_REMOTE_OPENAI_CONTEXT_LENGTH") or "131072"),
                },
                "prompt_token_count": in_tok,
            },
            "response": {
                "message": manager_message,
                "blocks": blocks,
                "raw_openai_message": message,
                "finish_reason": finish_reason,
                "stop_reason": stop_reason,
                "output_token_count": out_tok,
            },
            "usage": usage,
        }
        if state is not None:
            sft_record["conditioning"] = {
                "kind": "checkpoint_authoritative",
                "checkpoint_id": state.checkpoint_id,
                "checkpoint_prompt_sha256": state.checkpoint_prompt_sha256,
                "checkpoint_messages_sha256": state.checkpoint_messages_sha256,
                "attempt_id": state.attempt_id,
            }
        if self.include_session_id:
            sft_record["session_id"] = sid
        if self.include_wire_request:
            sft_record["wire_request"] = body
            sft_record["remote_request"] = payload
            sft_record["remote_response"] = data

        if state is not None:
            cached_reply = _CachedTeacherReply(
                request_wire_sha256=request_wire_sha256,
                blocks=copy.deepcopy(blocks),
                stop_reason=stop_reason,
                input_tokens=in_tok,
                output_tokens=out_tok,
            )
            next_messages = copy.deepcopy(messages)
            next_messages.append(copy.deepcopy(manager_message))
            state.pending_commit = _PendingTeacherCommit(
                request_wire_sha256=request_wire_sha256,
                record=sft_record,
                messages=next_messages,
                pending_tool_uses=[
                    copy.deepcopy(block)
                    for block in blocks
                    if isinstance(block, dict) and block.get("type") == "tool_use"
                ],
                completed_tool_use_ids=(
                    list(state.completed_tool_use_ids) + list(completed_ids)
                ),
                known_wire_tool_ids=(
                    set(state.known_wire_tool_ids)
                    | (_wire_tool_ids(body) if turn_index == 0 else set())
                ),
                last_stop_reason=stop_reason,
                turn_index=turn_index + 1,
                cached_reply=cached_reply,
            )
            try:
                self._commit_pending_teacher_turn(state)
            except OSError as exc:
                return self._persistence_error_response(state, exc)
        else:
            _append_jsonl(self._log_path(sid, state), sft_record)
            self.turn_index[sid] = turn_index + 1

        if body.get("stream") is True or "text/event-stream" in request.headers.get("Accept", ""):
            return await anth._render_stream(request, blocks, stop_reason, in_tok, out_tok)
        return web.json_response(anth._render_response(body, blocks, stop_reason, in_tok, out_tok))


def build_app_from_env() -> RemoteOpenAISFTAdapter:
    base_url = (os.environ.get("SLIME_REMOTE_OPENAI_BASE_URL") or "").strip()
    api_key = (os.environ.get("SLIME_REMOTE_OPENAI_API_KEY") or "").strip()
    model = (os.environ.get("SLIME_REMOTE_OPENAI_MODEL") or "").strip()
    sft_log_dir = (os.environ.get("SLIME_AGENT_SFT_LOG_DIR") or "").strip()
    if not base_url:
        raise RuntimeError("SLIME_REMOTE_OPENAI_BASE_URL is required")
    if not api_key:
        raise RuntimeError("SLIME_REMOTE_OPENAI_API_KEY is required")
    if not model:
        raise RuntimeError("SLIME_REMOTE_OPENAI_MODEL is required")
    if not sft_log_dir:
        raise RuntimeError("SLIME_AGENT_SFT_LOG_DIR is required")
    return RemoteOpenAISFTAdapter(
        base_url=base_url,
        api_key=api_key,
        model=model,
        sft_log_dir=sft_log_dir,
        temperature=float(os.environ.get("SLIME_REMOTE_OPENAI_TEMPERATURE") or "1"),
        top_p=float(os.environ.get("SLIME_REMOTE_OPENAI_TOP_P") or "0.95"),
        top_k=int(os.environ.get("SLIME_REMOTE_OPENAI_TOP_K") or "20"),
        reasoning_effort=os.environ.get("SLIME_REMOTE_OPENAI_REASONING_EFFORT") or "max",
        thinking_type=os.environ.get("SLIME_REMOTE_OPENAI_THINKING_TYPE"),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("SLIME_ADAPTER_BIND_HOST") or "0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("SLIME_ADAPTER_PORT") or "18001"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    adapter = build_app_from_env()
    logger.info("starting remote OpenAI SFT adapter on %s:%s -> %s", args.host, args.port, adapter.base_url)
    handle = run_app_in_thread(
        adapter.app,
        host=args.host,
        port=args.port,
        thread_name="remote-openai-sft-adapter",
        runner_kwargs={"handler_cancellation": True, "access_log_class": FilteredAccessLogger},
    )
    logger.info("adapter ready port=%s", handle.port)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
