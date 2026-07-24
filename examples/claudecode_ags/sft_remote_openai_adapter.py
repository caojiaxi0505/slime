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
import hashlib
import json
import logging
import os
import random
import secrets
import time
from typing import Any

import aiohttp
from aiohttp import web

from slime.agent.adapters import anthropic as anth
from slime.agent.adapters.anthropic_segmented import (
    _append_jsonl,
    _env_flag,
    _json_safe,
    _session_log_path,
    canonical_sha256,
)
from slime.agent.adapters.common import flatten_content, sid_from_bearer
from slime.agent.aiohttp_threaded import FilteredAccessLogger, run_app_in_thread

logger = logging.getLogger(__name__)


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
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.sft_log_dir = sft_log_dir
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.reasoning_effort = reasoning_effort
        self.include_session_id = _env_flag("SLIME_AGENT_SFT_LOG_INCLUDE_SESSION_ID")
        self.include_wire_request = _env_flag("SLIME_AGENT_SFT_LOG_INCLUDE_WIRE_REQUEST")
        max_inflight = int(os.environ.get("SLIME_REMOTE_OPENAI_MAX_INFLIGHT") or "0")
        self.remote_sem = asyncio.Semaphore(max_inflight) if max_inflight > 0 else None
        self.retry_delays = _retry_delays_from_env()
        self.turn_index: dict[str, int] = {}
        self.http: aiohttp.ClientSession | None = None
        self.app = web.Application(client_max_size=128 * 1024 * 1024)
        self.app.on_cleanup.append(self._cleanup)
        self.app.router.add_get("/health", self._health)
        self.app.router.add_get("/healthz", self._health)
        self.app.router.add_post("/v1/messages", self._handle_messages)
        self.app.router.add_post("/v1/messages/count_tokens", self._count_tokens)

    async def _cleanup(self, _: web.Application) -> None:
        if self.http is not None:
            await self.http.close()
            self.http = None

    async def _client(self) -> aiohttp.ClientSession:
        if self.http is None or self.http.closed:
            timeout = aiohttp.ClientTimeout(total=int(os.environ.get("SLIME_REMOTE_OPENAI_TIMEOUT_SEC") or "1200"))
            self.http = aiohttp.ClientSession(timeout=timeout)
        return self.http

    async def _post_chat_completions(self, payload: dict[str, Any]) -> tuple[int, str, dict[str, Any] | None]:
        client = await self._client()
        retry_statuses = {408, 409, 425, 429, 500, 502, 503, 504}
        attempts = len(self.retry_delays) + 1
        last_status = 0
        last_text = ""

        for attempt in range(attempts):
            if attempt:
                delay = self.retry_delays[attempt - 1] + random.uniform(0.0, 2.0)
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

    async def _handle_messages(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        anth._fold_mid_list_system_into_user(body)
        sid = self._session_id(request)
        messages = _messages_to_openai(body)
        tools = anth._tools_to_chat_tools(body.get("tools"))
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "reasoning_effort": self.reasoning_effort,
            "max_tokens": int(body.get("max_tokens") or 16384),
        }
        if tools:
            payload["tools"] = tools
        if body.get("stop_sequences"):
            payload["stop"] = body.get("stop_sequences")

        started = time.time()
        status, text, data = await self._post_chat_completions(payload)
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

        turn_index = self.turn_index.get(sid, 0)
        self.turn_index[sid] = turn_index + 1
        sft_record: dict[str, Any] = {
            "version": 1,
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
                    "reasoning_effort": self.reasoning_effort,
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
        if self.include_session_id:
            sft_record["session_id"] = sid
        if self.include_wire_request:
            sft_record["wire_request"] = body
            sft_record["remote_request"] = payload
            sft_record["remote_response"] = data
        _append_jsonl(_session_log_path(self.sft_log_dir, sid), sft_record)

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
