"""The teacher adapter stops a capped Stage-2 relabel without remote retries."""

from __future__ import annotations

import asyncio
import json
import os

import pytest
from aiohttp.test_utils import TestClient, TestServer
import examples.claudecode_ags.sft_remote_openai_adapter as adapter_module
from examples.claudecode_ags.sft_remote_openai_adapter import RemoteOpenAISFTAdapter
from slime.agent.adapters.anthropic_segmented import canonical_sha256, prompt_ids_sha256


def _adapter(tmp_path, cap):
    return RemoteOpenAISFTAdapter(
        base_url="http://unused",
        api_key="k",
        model="teacher",
        sft_log_dir=str(tmp_path),
        max_turns_per_sid=cap,
    )


def _checkpoint():
    prompt_ids = [11, 12, 13]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "Read",
                "description": "read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"],
                },
            },
        }
    ]
    return {
        "checkpoint_id": "main-7-test",
        "prompt_ids": prompt_ids,
        "prompt_sha256": prompt_ids_sha256(prompt_ids),
        "chat_messages": [
            {"role": "system", "content": "authoritative system"},
            {"role": "user", "content": "fix the bug"},
        ],
        "tools_schema": tools,
        "tools_sha256": canonical_sha256(tools),
    }


def test_turn_cap_kills_run_after_n_teacher_turns(tmp_path):
    adapter = _adapter(tmp_path, 2)
    calls = 0

    async def _fake_post(payload):
        nonlocal calls
        calls += 1
        del payload
        return 200, "", {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}

    adapter._post_chat_completions = _fake_post

    async def run_case():
        client = TestClient(TestServer(adapter.app))
        await client.start_server()
        try:
            body = {"model": "m", "max_tokens": 4, "messages": [{"role": "user", "content": "x"}]}
            headers = {"Authorization": "Bearer sid-cap"}
            return [(await client.post("/v1/messages", headers=headers, json=body)).status for _ in range(3)]
        finally:
            await client.close()

    statuses = asyncio.run(run_case())
    assert statuses == [200, 200, 200]
    assert calls == 2


def test_remote_429_does_not_count_as_a_completed_step(tmp_path):
    adapter = _adapter(tmp_path, 2)
    calls = 0

    async def _fake_post(payload):
        nonlocal calls
        calls += 1
        del payload
        if calls == 1:
            return 429, "rate limited", None
        return 200, "", {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}

    adapter._post_chat_completions = _fake_post

    async def run_case():
        client = TestClient(TestServer(adapter.app))
        await client.start_server()
        try:
            body = {"model": "m", "max_tokens": 4, "messages": [{"role": "user", "content": "x"}]}
            headers = {"Authorization": "Bearer sid-cap"}
            return [(await client.post("/v1/messages", headers=headers, json=body)).status for _ in range(4)]
        finally:
            await client.close()

    statuses = asyncio.run(run_case())
    # First remote 429 is surfaced as 502, not the turn cap; two successes
    # still fit in MAX_STEPS=2, and the fourth request is a local non-retry
    # end_turn that does not call the remote teacher.
    assert statuses == [502, 200, 200, 200]
    assert calls == 3


def test_uncapped_adapter_keeps_serving(tmp_path):
    adapter = _adapter(tmp_path, None)

    async def _fake_post(payload):
        del payload
        return 200, "", {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}

    adapter._post_chat_completions = _fake_post

    async def run_case():
        client = TestClient(TestServer(adapter.app))
        await client.start_server()
        try:
            body = {"model": "m", "max_tokens": 4, "messages": [{"role": "user", "content": "x"}]}
            headers = {"Authorization": "Bearer sid-free"}
            return [(await client.post("/v1/messages", headers=headers, json=body)).status for _ in range(3)]
        finally:
            await client.close()

    assert asyncio.run(run_case()) == [200, 200, 200]


def test_payload_enables_max_thinking(tmp_path):
    adapter = _adapter(tmp_path, None)
    seen = {}

    async def _fake_post(payload):
        seen.update(payload)
        return 200, "", {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}

    adapter._post_chat_completions = _fake_post

    async def run_case():
        client = TestClient(TestServer(adapter.app))
        await client.start_server()
        try:
            body = {"model": "m", "max_tokens": 4, "messages": [{"role": "user", "content": "x"}]}
            resp = await client.post("/v1/messages", headers={"Authorization": "Bearer sid-think"}, json=body)
            return resp.status
        finally:
            await client.close()

    assert asyncio.run(run_case()) == 200
    assert seen["thinking"] == {"type": "enabled"}
    assert seen["reasoning_effort"] == "max"


def test_payload_can_disable_thinking(tmp_path, monkeypatch):
    monkeypatch.setenv("SLIME_REMOTE_OPENAI_THINKING_TYPE", "disabled")
    adapter = _adapter(tmp_path, None)
    seen = {}

    async def _fake_post(payload):
        seen.update(payload)
        return 200, "", {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}

    adapter._post_chat_completions = _fake_post

    async def run_case():
        client = TestClient(TestServer(adapter.app))
        await client.start_server()
        try:
            body = {"model": "m", "max_tokens": 4, "messages": [{"role": "user", "content": "x"}]}
            return (await client.post("/v1/messages", headers={"Authorization": "Bearer sid-off"}, json=body)).status
        finally:
            await client.close()

    assert asyncio.run(run_case()) == 200
    assert seen["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in seen


def test_keepalive_starts_four_requests_in_parallel(tmp_path, monkeypatch):
    monkeypatch.setenv("SLIME_REMOTE_OPENAI_KEEPALIVE_SEC", "0.001")
    monkeypatch.setenv("SLIME_REMOTE_OPENAI_KEEPALIVE_INFLIGHT", "4")
    adapter = _adapter(tmp_path, None)
    active = 0
    peak = 0
    started = asyncio.Event()
    release = asyncio.Event()

    async def _fake_post(payload, *, retry_delays=None):
        nonlocal active, peak
        del payload, retry_delays
        active += 1
        peak = max(peak, active)
        if active == 4:
            started.set()
        await release.wait()
        active -= 1
        return 200, "", {}

    adapter._post_chat_completions = _fake_post

    async def run_case():
        task = asyncio.create_task(adapter._keepalive_loop())
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
            assert active == 4
            assert peak == 4
        finally:
            release.set()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    asyncio.run(run_case())


def test_registered_resume_uses_checkpoint_then_real_tool_result(tmp_path):
    adapter = RemoteOpenAISFTAdapter(
        base_url="http://unused",
        api_key="k",
        model="teacher",
        sft_log_dir=str(tmp_path),
        max_turns_per_sid=2,
        require_registered_sessions=True,
    )
    checkpoint = _checkpoint()
    log_path = adapter.register_resume_session("sid-exact", checkpoint)
    payloads = []

    async def _fake_post(payload):
        payloads.append(payload)
        if len(payloads) == 1:
            return 200, "", {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "toolu_teacher_1",
                                    "type": "function",
                                    "function": {
                                        "name": "Read",
                                        "arguments": '{"file_path":"/testbed/a.py"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        return 200, "", {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "done"},
                    "finish_reason": "stop",
                }
            ]
        }

    adapter._post_chat_completions = _fake_post

    async def run_case():
        client = TestClient(TestServer(adapter.app))
        await client.start_server()
        try:
            first = {
                "model": "m",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "resume handshake"}],
                "system": "wrong reconstructed system",
                "tools": [{"name": "WrongTool", "input_schema": {"type": "object"}}],
            }
            first_resp = await client.post(
                "/v1/messages",
                headers={"Authorization": "Bearer sid-exact"},
                json=first,
            )
            second = {
                "model": "m",
                "max_tokens": 64,
                "messages": [
                    {"role": "user", "content": "resume handshake"},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_teacher_1",
                                "content": "print('ok')",
                            }
                        ],
                    },
                ],
            }
            second_resp = await client.post(
                "/v1/messages",
                headers={"Authorization": "Bearer sid-exact"},
                json=second,
            )
            return first_resp.status, second_resp.status
        finally:
            await client.close()

    assert asyncio.run(run_case()) == (200, 200)
    assert payloads[0]["messages"] == checkpoint["chat_messages"]
    assert payloads[0]["tools"] == checkpoint["tools_schema"]
    assert payloads[1]["messages"][:2] == checkpoint["chat_messages"]
    assert payloads[1]["messages"][2]["role"] == "assistant"
    assert "id" not in payloads[1]["messages"][2]["tool_calls"][0]
    assert payloads[1]["messages"][3] == {
        "role": "tool",
        "content": "print('ok')",
    }
    rows = [json.loads(line) for line in open(log_path, encoding="utf-8") if line.strip()]
    assert [row["turn_index"] for row in rows] == [0, 1]
    assert all(row["version"] == 2 for row in rows)
    assert rows[0]["prompt"]["messages"] == checkpoint["chat_messages"]
    assert rows[1]["prompt"]["messages"] == payloads[1]["messages"]
    assert len({row["conditioning"]["attempt_id"] for row in rows}) == 1


@pytest.mark.parametrize("result_count", [0, 2])
def test_registered_resume_rejects_missing_or_duplicate_tool_result(tmp_path, result_count):
    adapter = RemoteOpenAISFTAdapter(
        base_url="http://unused",
        api_key="k",
        model="teacher",
        sft_log_dir=str(tmp_path),
        require_registered_sessions=True,
    )
    adapter.register_resume_session("sid-missing", _checkpoint())

    async def _fake_post(payload):
        del payload
        return 200, "", {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "toolu_expected",
                                "type": "function",
                                "function": {"name": "Read", "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

    adapter._post_chat_completions = _fake_post

    async def run_case():
        client = TestClient(TestServer(adapter.app))
        await client.start_server()
        try:
            headers = {"Authorization": "Bearer sid-missing"}
            first = await client.post(
                "/v1/messages",
                headers=headers,
                json={"messages": [{"role": "user", "content": "handshake"}]},
            )
            second = await client.post(
                "/v1/messages",
                headers=headers,
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "toolu_expected",
                                    "content": f"result-{index}",
                                }
                                for index in range(result_count)
                            ],
                        }
                    ]
                },
            )
            return first.status, second.status, await second.text()
        finally:
            await client.close()

    first_status, second_status, text = asyncio.run(run_case())
    assert first_status == 200
    assert second_status == 400
    assert "teacher_context_integrity" in text


def test_same_sid_after_restart_gets_a_new_attempt_file(tmp_path):
    checkpoint = _checkpoint()
    first = RemoteOpenAISFTAdapter(
        base_url="http://unused",
        api_key="k",
        model="teacher",
        sft_log_dir=str(tmp_path),
    )
    first_path = first.register_resume_session("stable-sid", checkpoint)
    first.close_resume_session("stable-sid")

    restarted = RemoteOpenAISFTAdapter(
        base_url="http://unused",
        api_key="k",
        model="teacher",
        sft_log_dir=str(tmp_path),
    )
    second_path = restarted.register_resume_session("stable-sid", checkpoint)
    assert first_path != second_path
    assert not os.path.exists(first_path)
    assert not os.path.exists(second_path)
    assert open(f"{first_path}.claim", encoding="utf-8").read().startswith("attempt_id=")
    assert open(f"{second_path}.claim", encoding="utf-8").read().startswith("attempt_id=")


def test_registered_resume_exact_retry_is_idempotent(tmp_path):
    adapter = RemoteOpenAISFTAdapter(
        base_url="http://unused",
        api_key="k",
        model="teacher",
        sft_log_dir=str(tmp_path),
        require_registered_sessions=True,
    )
    log_path = adapter.register_resume_session("sid-retry", _checkpoint())
    calls = 0

    async def _fake_post(payload):
        nonlocal calls
        del payload
        calls += 1
        return 200, "", {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "done"},
                    "finish_reason": "stop",
                }
            ]
        }

    adapter._post_chat_completions = _fake_post

    async def run_case():
        client = TestClient(TestServer(adapter.app))
        await client.start_server()
        try:
            body = {"messages": [{"role": "user", "content": "same handshake"}]}
            headers = {"Authorization": "Bearer sid-retry"}
            first = await client.post("/v1/messages", headers=headers, json=body)
            second = await client.post("/v1/messages", headers=headers, json=body)
            return first.status, second.status
        finally:
            await client.close()

    assert asyncio.run(run_case()) == (200, 200)
    assert calls == 1
    assert len([line for line in open(log_path, encoding="utf-8") if line.strip()]) == 1


def test_registered_resume_append_failure_is_transactional_and_retryable(
    tmp_path,
    monkeypatch,
):
    adapter = RemoteOpenAISFTAdapter(
        base_url="http://unused",
        api_key="k",
        model="teacher",
        sft_log_dir=str(tmp_path),
        require_registered_sessions=True,
    )
    log_path = adapter.register_resume_session("sid-persist-retry", _checkpoint())
    calls = 0
    failed_once = False
    real_append = adapter_module._append_jsonl

    def _fail_first_teacher_append(path, record):
        nonlocal failed_once
        if record.get("version") == 2 and not failed_once:
            failed_once = True
            raise OSError(22, "injected append failure")
        real_append(path, record)

    monkeypatch.setattr(adapter_module, "_append_jsonl", _fail_first_teacher_append)

    async def _fake_post(payload):
        nonlocal calls
        del payload
        calls += 1
        if calls == 1:
            return 200, "", {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "toolu_after_persist",
                                    "type": "function",
                                    "function": {
                                        "name": "Read",
                                        "arguments": '{"file_path":"/testbed/a.py"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        return 200, "", {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "done"},
                    "finish_reason": "stop",
                }
            ]
        }

    adapter._post_chat_completions = _fake_post

    async def run_case():
        client = TestClient(TestServer(adapter.app))
        await client.start_server()
        try:
            headers = {"Authorization": "Bearer sid-persist-retry"}
            handshake = {"messages": [{"role": "user", "content": "same handshake"}]}
            first = await client.post("/v1/messages", headers=headers, json=handshake)
            after_failure = adapter.resume_session_status("sid-persist-retry")
            retry = await client.post("/v1/messages", headers=headers, json=handshake)
            after_retry = adapter.resume_session_status("sid-persist-retry")
            tool_result = await client.post(
                "/v1/messages",
                headers=headers,
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "toolu_after_persist",
                                    "content": "file contents",
                                }
                            ],
                        }
                    ]
                },
            )
            return (
                first.status,
                retry.status,
                tool_result.status,
                after_failure,
                after_retry,
            )
        finally:
            await client.close()

    first_status, retry_status, tool_status, after_failure, after_retry = asyncio.run(
        run_case()
    )
    assert (first_status, retry_status, tool_status) == (500, 200, 200)
    assert after_failure["turn_index"] == 0
    assert after_failure["pending_tool_use_ids"] == []
    assert after_failure["persistence_pending"] is True
    assert after_retry["turn_index"] == 1
    assert after_retry["pending_tool_use_ids"] == ["toolu_after_persist"]
    assert after_retry["persistence_pending"] is False
    assert calls == 2
    rows = [json.loads(line) for line in open(log_path, encoding="utf-8") if line.strip()]
    assert [row["turn_index"] for row in rows] == [0, 1]


def test_strict_teacher_adapter_rejects_unregistered_session(tmp_path):
    adapter = RemoteOpenAISFTAdapter(
        base_url="http://unused",
        api_key="k",
        model="teacher",
        sft_log_dir=str(tmp_path),
        require_registered_sessions=True,
    )

    async def run_case():
        client = TestClient(TestServer(adapter.app))
        await client.start_server()
        try:
            response = await client.post(
                "/v1/messages",
                headers={"Authorization": "Bearer unknown"},
                json={"messages": [{"role": "user", "content": "handshake"}]},
            )
            return response.status, await response.text()
        finally:
            await client.close()

    status, text = asyncio.run(run_case())
    assert status == 400
    assert "session_not_registered" in text
