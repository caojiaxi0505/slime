"""Checkpoint and token-exact resume tests for the segmented adapter."""

from __future__ import annotations

import asyncio
import copy
import json
from unittest.mock import patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from slime.agent.adapters.anthropic_segmented import (
    PromptCheckpoint,
    ResumeState,
    SegmentedAnthropicAdapter,
    Session,
    consume_resume_tool_results,
    prompt_ids_sha256,
)
from slime.agent.trajectory import TurnRecord
from tests.test_agent._fakes import FakeSGLangServer, FakeTokenizer


TOOLS = [
    {
        "name": "Read",
        "description": "read a file",
        "input_schema": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "Bash",
        "description": "run a command",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
            "additionalProperties": False,
        },
    },
]


def _tool_xml(name: str, key: str, value: str) -> str:
    return (
        f"<tool_call><function={name}>"
        f"<parameter={key}>{value}</parameter>"
        "</function></tool_call>"
    )


async def _stage1_checkpoint() -> tuple[dict, list[int], str]:
    reply = _tool_xml("Read", "file_path", "/repo/a.py")
    async with FakeSGLangServer([[(-0.1, 701)]]) as sglang:
        tokenizer = FakeTokenizer(outputs={(701,): reply})
        adapter = SegmentedAnthropicAdapter(tokenizer=tokenizer, sglang_url=sglang.url)
        adapter.open_session("stage1", capture_prompt_checkpoints=True)
        client = TestClient(TestServer(adapter.app))
        await client.start_server()
        try:
            response = await client.post(
                "/v1/messages",
                headers={"Authorization": "Bearer stage1"},
                json={
                    "model": "m",
                    "max_tokens": 16,
                    "tools": TOOLS,
                    "system": "stable system",
                    "messages": [{"role": "user", "content": "fix the bug"}],
                },
            )
            data = await response.json()
            checkpoints = await adapter.export_prompt_checkpoints_async("stage1")
        finally:
            await client.close()
        await adapter.finish_session("stage1")

    assert response.status == 200
    tool_use = next(block for block in data["content"] if block["type"] == "tool_use")
    assert len(checkpoints) == 1
    assert checkpoints[0]["generated_tool_use_ids"] == [tool_use["id"]]
    assert checkpoints[0]["generated_tool_use_names"] == {tool_use["id"]: "Read"}
    assert checkpoints[0]["prompt_ids"] == sglang.requests[0]["input_ids"]
    return checkpoints[0], sglang.requests[0]["input_ids"], tool_use["id"]


def test_prompt_checkpoint_roundtrip_and_hash_is_stable():
    checkpoint = PromptCheckpoint(
        checkpoint_id="main-0-deadbeef",
        prompt_ids=[1, 2, 0xFFFFFFFF],
        prompt_sha256=prompt_ids_sha256([1, 2, 0xFFFFFFFF]),
        chat_messages=[{"role": "user", "content": "x"}],
        tools_schema=None,
        tools_sha256="abc",
        generation_config={},
        tokenizer_fingerprint={},
        chain_kind="main",
        request_kind="new",
        request_index=0,
    )
    assert PromptCheckpoint.from_dict(checkpoint.to_dict()).to_dict() == checkpoint.to_dict()
    assert prompt_ids_sha256([1, 2, 3]) == prompt_ids_sha256([1, 2, 3])
    assert prompt_ids_sha256([1, 2, 3]) != prompt_ids_sha256([3, 2, 1])


def test_sft_turn_logger_writes_real_request_response_context(tmp_path, monkeypatch):
    async def run_case():
        monkeypatch.setenv("SLIME_AGENT_SFT_LOG_DIR", str(tmp_path))
        reply = "I will inspect the file."
        async with FakeSGLangServer([[(-0.1, 701)]]) as sglang:
            tokenizer = FakeTokenizer(outputs={(701,): reply})
            adapter = SegmentedAnthropicAdapter(tokenizer=tokenizer, sglang_url=sglang.url)
            adapter.open_session(
                "sft-session",
                sampling_defaults={"temperature": 1.0},
                max_context_tokens=4096,
            )
            client = TestClient(TestServer(adapter.app))
            await client.start_server()
            try:
                response = await client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer sft-session"},
                    json={
                        "model": "m",
                        "max_tokens": 16,
                        "tools": TOOLS,
                        "system": "stable system",
                        "messages": [{"role": "user", "content": "fix the bug"}],
                    },
                )
                assert response.status == 200
                await response.json()
            finally:
                await client.close()
            await adapter.finish_session("sft-session")

        files = list(tmp_path.glob("*.sft_turns.jsonl"))
        assert len(files) == 1
        rows = [json.loads(line) for line in files[0].read_text().splitlines()]
        assert len(rows) == 1
        row = rows[0]
        assert row["session_id_sha256"]
        assert "session_id" not in row
        assert row["request_kind"] == "new"
        assert row["prompt"]["messages"][0]["role"] == "system"
        assert row["prompt"]["messages"][0]["content"] == "stable system"
        assert row["prompt"]["tools_schema"]
        assert row["prompt"]["generation_config"]["request"]["max_tokens"] == 16
        assert row["prompt"]["prompt_token_count"] == len(sglang.requests[0]["input_ids"])
        assert row["response"]["raw_output_text"] == reply
        assert row["response"]["message"]["role"] == "assistant"
        assert row["response"]["message"]["content"] == reply
        assert "prompt_ids" not in row["prompt"]
        assert "output_ids" not in row["response"]

    asyncio.run(run_case())


def test_async_resume_session_validation_uses_cpu_executor():
    async def run_case():
        checkpoint, _, _ = await _stage1_checkpoint()
        adapter = SegmentedAnthropicAdapter(
            tokenizer=FakeTokenizer(),
            sglang_url="http://unused",
            cpu_workers=2,
        )
        try:
            await adapter.open_session_async(
                "async-resume",
                resume_checkpoint=checkpoint,
            )
            status = adapter.resume_status("async-resume")
            assert status["mode"] == "token_exact"
            assert status["checkpoint_id"] == checkpoint["checkpoint_id"]
        finally:
            await adapter._cleanup_resources(adapter.app)

    asyncio.run(run_case())


@pytest.mark.parametrize("case", ["duplicate", "unexpected"])
def test_resume_tool_result_validation_rejects_duplicate_and_unknown_ids(case):
    checkpoint = PromptCheckpoint(
        checkpoint_id="main-0-checkpoint",
        prompt_ids=[1],
        prompt_sha256=prompt_ids_sha256([1]),
        chat_messages=[],
        tools_schema=None,
        tools_sha256="",
        generation_config={},
        tokenizer_fingerprint={},
        chain_kind="main",
        request_kind="new",
        request_index=0,
    )
    pending = {"type": "tool_use", "id": "toolu_new", "name": "Read", "input": {}}
    session = Session(resume=ResumeState(checkpoint=checkpoint, pending_tool_uses=[pending]))
    results = [
        {"type": "tool_result", "tool_use_id": "toolu_new", "content": "ok"},
    ]
    if case == "duplicate":
        results.append(copy.deepcopy(results[0]))
        expected = "expected one result"
    else:
        results.append({"type": "tool_result", "tool_use_id": "toolu_unknown", "content": "x"})
        expected = "unexpected tool_result id"
    body = {
        "messages": [
            {"role": "assistant", "content": [pending]},
            {"role": "user", "content": results},
        ]
    }
    with pytest.raises(ValueError, match=expected):
        consume_resume_tool_results(session, body)


def test_resume_audits_non_authoritative_tool_echo_payload_without_rejecting(caplog):
    checkpoint = PromptCheckpoint(
        checkpoint_id="main-0-checkpoint",
        prompt_ids=[1],
        prompt_sha256=prompt_ids_sha256([1]),
        chat_messages=[],
        tools_schema=None,
        tools_sha256="",
        generation_config={},
        tokenizer_fingerprint={},
        chain_kind="main",
        request_kind="new",
        request_index=0,
    )
    pending = {
        "type": "tool_use",
        "id": "toolu_new",
        "name": "Read",
        "input": {"file_path": "/repo/a.py"},
    }
    session = Session(resume=ResumeState(checkpoint=checkpoint, pending_tool_uses=[pending]))
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_new",
                        "name": "Read",
                        "input": {"file_path": "/repo/./a.py"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_new",
                        "content": "file contents",
                    }
                ],
            },
        ]
    }

    consume_resume_tool_results(session, body)

    assert session.resume is not None
    assert session.resume.error == ""
    assert session.resume.completed_tool_use_ids == ["toolu_new"]
    assert session.resume.tool_use_echo_mismatch_count == 1
    assert session.main.chat_messages == [{"role": "tool", "content": "file contents"}]
    assert "non-authoritative Claude Code tool_use echo differs" in caplog.text


def test_resume_accepts_missing_non_authoritative_tool_echo(caplog):
    checkpoint = PromptCheckpoint(
        checkpoint_id="main-0-checkpoint",
        prompt_ids=[1],
        prompt_sha256=prompt_ids_sha256([1]),
        chat_messages=[],
        tools_schema=None,
        tools_sha256="",
        generation_config={},
        tokenizer_fingerprint={},
        chain_kind="main",
        request_kind="new",
        request_index=0,
    )
    pending = {
        "type": "tool_use",
        "id": "toolu_new",
        "name": "Read",
        "input": {"file_path": "/repo/a.py"},
    }
    session = Session(resume=ResumeState(checkpoint=checkpoint, pending_tool_uses=[pending]))
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_new",
                        "content": "file contents",
                    }
                ],
            }
        ]
    }

    consume_resume_tool_results(session, body)

    assert session.resume is not None
    assert session.resume.error == ""
    assert session.resume.completed_tool_use_ids == ["toolu_new"]
    assert session.resume.tool_use_echo_missing_count == 1
    assert session.resume.tool_use_echo_payload_mismatch_count == 0
    assert session.resume.tool_use_echo_mismatch_count == 1
    assert session.main.chat_messages == [{"role": "tool", "content": "file contents"}]
    assert "omitted non-authoritative tool_use echo" in caplog.text


def test_resume_first_request_uses_saved_ids_and_later_uses_authoritative_state():
    async def run_case():
        checkpoint, saved_ids, _old_tool_id = await _stage1_checkpoint()
        branch_tool = _tool_xml("Read", "file_path", "/repo/b.py")
        async with FakeSGLangServer([[(-0.2, 801)], [(-0.3, 802)]]) as sglang:
            tokenizer = FakeTokenizer(outputs={(801,): branch_tool, (802,): "finished"})
            adapter = SegmentedAnthropicAdapter(tokenizer=tokenizer, sglang_url=sglang.url)
            adapter.open_session("stage2", resume_checkpoint=checkpoint)
            client = TestClient(TestServer(adapter.app))
            await client.start_server()
            try:
                # This bootstrap history is intentionally wrong.  Only its tool
                # schema and generation settings are treated as a handshake.
                first = await client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer stage2"},
                    json={
                        "model": "m",
                        "max_tokens": 16,
                        "tools": TOOLS,
                        "system": "different sandbox system text",
                        "messages": [{"role": "user", "content": "distorted bootstrap"}],
                    },
                )
                first_data = await first.json()
                branch_use = next(block for block in first_data["content"] if block["type"] == "tool_use")
                first_http_session = adapter._sglang_http_session

                second = await client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer stage2"},
                    json={
                        "model": "m",
                        "max_tokens": 16,
                        "tools": TOOLS,
                        "system": "still different",
                        "messages": [
                            {"role": "user", "content": "distorted bootstrap"},
                            {"role": "assistant", "content": [copy.deepcopy(branch_use)]},
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": branch_use["id"],
                                        "content": "file contents",
                                    }
                                ],
                            },
                        ],
                    },
                )
                second_data = await second.json()
                status = adapter.resume_status("stage2")
                reused_http_session = adapter._sglang_http_session is first_http_session
            finally:
                await client.close()
            segments = await adapter.finish_session("stage2")

        assert first.status == 200
        assert second.status == 200
        assert reused_http_session
        assert first_http_session.closed
        assert second_data["content"] == [{"type": "text", "text": "finished"}]
        assert sglang.requests[0]["input_ids"] == saved_ids

        reference = FakeTokenizer()
        expected_messages = copy.deepcopy(checkpoint["chat_messages"])
        expected_messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "Read", "arguments": {"file_path": "/repo/b.py"}},
                    }
                ],
            }
        )
        expected_messages.append({"role": "tool", "content": "file contents"})
        expected_second = reference.apply_chat_template(
            expected_messages,
            tools=checkpoint["tools_schema"],
            tokenize=True,
            add_generation_prompt=True,
        )
        assert sglang.requests[1]["input_ids"] == expected_second
        assert status == {
            "mode": "token_exact",
            "checkpoint_id": checkpoint["checkpoint_id"],
            "prompt_sha256": checkpoint["prompt_sha256"],
            "first_prompt_sha256": checkpoint["prompt_sha256"],
            "first_prompt_exact": True,
            "handshake_validated": True,
            "exact_request_count": 2,
            "pending_tool_use_ids": [],
            "completed_tool_use_ids": [branch_use["id"]],
            "last_stop_reason": "end_turn",
            "tool_use_echo_mismatch_count": 0,
            "tool_use_echo_missing_count": 0,
            "tool_use_echo_payload_mismatch_count": 0,
            "runtime_tool_schema_mismatch_count": 0,
            "generated_runtime_tool_unavailable_count": 0,
            "generated_runtime_tool_input_invalid_count": 0,
            "max_tokens_continuation_count": 0,
            "post_end_turn_ack_count": 0,
            "request_replay_count": 0,
            "error": "",
        }
        assert len(segments) == 1 and segments[0].prompt_ids == saved_ids
        assert sum(segments[0].loss_mask) >= 1

    asyncio.run(run_case())


def test_resume_continues_from_authoritative_state_after_max_tokens():
    async def run_case():
        checkpoint, saved_ids, _ = await _stage1_checkpoint()
        async with FakeSGLangServer(
            [[(-0.1, 811)], [(-0.1, 812)]],
            finish_reason="length",
        ) as sglang:
            tokenizer = FakeTokenizer(outputs={(811,): "partial", (812,): "continued"})
            adapter = SegmentedAnthropicAdapter(tokenizer=tokenizer, sglang_url=sglang.url)
            adapter.open_session("max-tokens", resume_checkpoint=checkpoint)
            client = TestClient(TestServer(adapter.app))
            await client.start_server()
            try:
                first = await client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer max-tokens"},
                    json={"max_tokens": 16, "tools": TOOLS, "messages": []},
                )
                second = await client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer max-tokens"},
                    json={
                        "max_tokens": 16,
                        "tools": TOOLS,
                        "messages": [
                            {"role": "assistant", "content": [{"type": "text", "text": "partial"}]}
                        ],
                    },
                )
                status = adapter.resume_status("max-tokens")
            finally:
                await client.close()
            await adapter.finish_session("max-tokens")

        assert first.status == second.status == 200
        assert sglang.requests[0]["input_ids"] == saved_ids
        assert len(sglang.requests) == 2
        assert status["exact_request_count"] == 2
        assert status["max_tokens_continuation_count"] == 1
        assert status["last_stop_reason"] == "max_tokens"
        assert status["error"] == ""

    asyncio.run(run_case())


def test_resume_replays_duplicate_claude_requests_without_resampling():
    async def run_case():
        checkpoint, saved_ids, _ = await _stage1_checkpoint()
        branch_tool = _tool_xml("Bash", "command", "pytest -q")
        async with FakeSGLangServer(
            [[(-0.1, 815)], [(-0.1, 816)]],
        ) as sglang:
            tokenizer = FakeTokenizer(outputs={(815,): branch_tool, (816,): "done"})
            adapter = SegmentedAnthropicAdapter(tokenizer=tokenizer, sglang_url=sglang.url)
            adapter.open_session("duplicate-request", resume_checkpoint=checkpoint)
            client = TestClient(TestServer(adapter.app))
            await client.start_server()
            first_body = {"max_tokens": 16, "tools": TOOLS, "messages": []}
            try:
                first = await client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer duplicate-request"},
                    json=first_body,
                )
                first_data = await first.json()
                first_retry = await client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer duplicate-request"},
                    json={
                        **first_body,
                        "model": "wire-only-model-change",
                        "metadata": {"retry": 1},
                        "tools": list(reversed(TOOLS)),
                    },
                )
                first_retry_data = await first_retry.json()
                tool_use = next(
                    block for block in first_data["content"] if block["type"] == "tool_use"
                )
                result_body = {
                    "max_tokens": 16,
                    "tools": TOOLS,
                    "messages": [
                        {"role": "assistant", "content": first_data["content"]},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_use["id"],
                                    "content": "ok",
                                }
                            ],
                        },
                    ],
                }
                second = await client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer duplicate-request"},
                    json=result_body,
                )
                second_data = await second.json()
                second_retry = await client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer duplicate-request"},
                    json={
                        **result_body,
                        "model": "another-wire-only-model-change",
                        "metadata": {"retry": 2},
                        "tools": list(reversed(TOOLS)),
                    },
                )
                second_retry_data = await second_retry.json()
                status = adapter.resume_status("duplicate-request")
            finally:
                await client.close()
            await adapter.finish_session("duplicate-request")

        assert first.status == first_retry.status == second.status == second_retry.status == 200
        assert first_retry_data == first_data
        assert second_retry_data == second_data
        assert sglang.requests[0]["input_ids"] == saved_ids
        assert len(sglang.requests) == 2
        assert status["request_replay_count"] == 2
        assert status["exact_request_count"] == 2
        assert status["last_stop_reason"] == "end_turn"
        assert status["error"] == ""

    asyncio.run(run_case())


def test_resume_forwards_invalid_tool_input_and_consumes_claude_code_error_result():
    async def run_case():
        checkpoint, _, _ = await _stage1_checkpoint()
        malformed_read = _tool_xml("Read", "options", "not-a-read-parameter")
        async with FakeSGLangServer([[(-0.1, 822)], [(-0.1, 823)]]) as sglang:
            tokenizer = FakeTokenizer(outputs={(822,): malformed_read, (823,): "recovered"})
            adapter = SegmentedAnthropicAdapter(tokenizer=tokenizer, sglang_url=sglang.url)
            adapter.open_session("invalid-tool-input", resume_checkpoint=checkpoint)
            client = TestClient(TestServer(adapter.app))
            await client.start_server()
            try:
                response = await client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer invalid-tool-input"},
                    json={"max_tokens": 16, "tools": TOOLS, "messages": []},
                )
                data = await response.json()
                tool_use = next(block for block in data["content"] if block["type"] == "tool_use")
                continued = await client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer invalid-tool-input"},
                    json={
                        "max_tokens": 16,
                        "tools": TOOLS,
                        "messages": [
                            {"role": "assistant", "content": data["content"]},
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": tool_use["id"],
                                        "content": (
                                            "<tool_use_error>InputValidationError: Read "
                                            "requires file_path</tool_use_error>"
                                        ),
                                        "is_error": True,
                                    }
                                ],
                            },
                        ],
                    },
                )
                status = adapter.resume_status("invalid-tool-input")
            finally:
                await client.close()
            await adapter.finish_session("invalid-tool-input")

        assert response.status == continued.status == 200
        assert status["pending_tool_use_ids"] == []
        assert status["completed_tool_use_ids"] == [tool_use["id"]]
        assert status["generated_runtime_tool_input_invalid_count"] == 1
        assert status["exact_request_count"] == 2
        assert status["error"] == ""
        assert len(sglang.requests) == 2

    asyncio.run(run_case())


def test_resume_replays_duplicate_streaming_request_with_same_message_id():
    async def run_case():
        checkpoint, _, _ = await _stage1_checkpoint()
        async with FakeSGLangServer([[(-0.1, 819)]]) as sglang:
            tokenizer = FakeTokenizer(outputs={(819,): "done"})
            adapter = SegmentedAnthropicAdapter(tokenizer=tokenizer, sglang_url=sglang.url)
            adapter.open_session("duplicate-stream", resume_checkpoint=checkpoint)
            client = TestClient(TestServer(adapter.app))
            await client.start_server()
            body = {"max_tokens": 16, "tools": TOOLS, "messages": [], "stream": True}
            try:
                first = await client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer duplicate-stream"},
                    json=body,
                )
                first_text = await first.text()
                retried = await client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer duplicate-stream"},
                    json=body,
                )
                retried_text = await retried.text()
                status = adapter.resume_status("duplicate-stream")
            finally:
                await client.close()
            await adapter.finish_session("duplicate-stream")

        assert first.status == retried.status == 200
        assert first_text == retried_text
        assert len(sglang.requests) == 1
        assert status["request_replay_count"] == 1
        assert status["exact_request_count"] == 1
        assert status["error"] == ""

    asyncio.run(run_case())


def test_resume_replays_from_pending_state_when_retry_history_changes():
    async def run_case():
        checkpoint, _, _ = await _stage1_checkpoint()
        branch_tool = _tool_xml("Bash", "command", "pytest -q")
        async with FakeSGLangServer(
            [[(-0.1, 823)], [(-0.1, 824)]],
        ) as sglang:
            tokenizer = FakeTokenizer(outputs={(823,): branch_tool, (824,): "done"})
            adapter = SegmentedAnthropicAdapter(tokenizer=tokenizer, sglang_url=sglang.url)
            adapter.open_session("pending-state-replay", resume_checkpoint=checkpoint)
            client = TestClient(TestServer(adapter.app))
            await client.start_server()
            try:
                first = await client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer pending-state-replay"},
                    json={"max_tokens": 16, "tools": TOOLS, "messages": []},
                )
                first_data = await first.json()
                # This is not hash-identical to the request that produced the
                # pending Bash call. With no result for that outstanding id it
                # cannot advance state and must receive the same response.
                rewritten_retry = await client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer pending-state-replay"},
                    json={
                        "max_tokens": 16,
                        "tools": TOOLS,
                        "system": "rewritten transport retry",
                        "messages": [{"role": "user", "content": "retry marker"}],
                    },
                )
                retry_data = await rewritten_retry.json()
                tool_use = next(
                    block for block in first_data["content"] if block["type"] == "tool_use"
                )
                result = await client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer pending-state-replay"},
                    json={
                        "max_tokens": 16,
                        "tools": TOOLS,
                        "messages": [
                            {"role": "assistant", "content": first_data["content"]},
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": tool_use["id"],
                                        "content": "ok",
                                    }
                                ],
                            },
                        ],
                    },
                )
                status = adapter.resume_status("pending-state-replay")
            finally:
                await client.close()
            await adapter.finish_session("pending-state-replay")

        assert first.status == rewritten_retry.status == result.status == 200
        assert retry_data == first_data
        assert len(sglang.requests) == 2
        assert status["request_replay_count"] == 1
        assert status["completed_tool_use_ids"] == [tool_use["id"]]
        assert status["error"] == ""

    asyncio.run(run_case())


def test_resume_rolls_back_consumed_tool_result_when_generation_fails():
    async def run_case():
        checkpoint, _, _ = await _stage1_checkpoint()
        branch_tool = _tool_xml("Read", "file_path", "/repo/b.py")
        tokenizer = FakeTokenizer(outputs={(817,): branch_tool, (818,): "done"})
        adapter = SegmentedAnthropicAdapter(tokenizer=tokenizer, sglang_url="http://unused")
        adapter.open_session("transaction-rollback", resume_checkpoint=checkpoint)
        calls = 0

        async def flaky_generate(prompt_ids, session, body, *, adapter, session_id=None):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("transient upstream failure")
            output_id = 817 if calls == 1 else 818
            return TurnRecord(
                prompt_ids=list(prompt_ids),
                output_ids=[output_id],
                finish_reason="stop",
                output_log_probs=[-0.1],
            )

        client = TestClient(TestServer(adapter.app))
        await client.start_server()
        try:
            with patch(
                "slime.agent.adapters.anthropic_segmented.call_sglang_generate",
                side_effect=flaky_generate,
            ):
                first = await client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer transaction-rollback"},
                    json={"max_tokens": 16, "tools": TOOLS, "messages": []},
                )
                first_data = await first.json()
                tool_use = next(
                    block for block in first_data["content"] if block["type"] == "tool_use"
                )
                result_body = {
                    "max_tokens": 16,
                    "tools": TOOLS,
                    "messages": [
                        {"role": "assistant", "content": first_data["content"]},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_use["id"],
                                    "content": "file contents",
                                }
                            ],
                        },
                    ],
                }
                failed = await client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer transaction-rollback"},
                    json=result_body,
                )
                after_failure = adapter.resume_status("transaction-rollback")
                retried = await client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer transaction-rollback"},
                    json=result_body,
                )
                status = adapter.resume_status("transaction-rollback")
        finally:
            await client.close()
        await adapter.finish_session("transaction-rollback")

        assert first.status == retried.status == 200
        assert failed.status == 500
        assert after_failure["pending_tool_use_ids"] == [tool_use["id"]]
        assert after_failure["completed_tool_use_ids"] == []
        assert after_failure["exact_request_count"] == 1
        assert after_failure["error"] == ""
        assert calls == 3
        assert status["pending_tool_use_ids"] == []
        assert status["completed_tool_use_ids"] == [tool_use["id"]]
        assert status["exact_request_count"] == 2
        assert status["last_stop_reason"] == "end_turn"
        assert status["error"] == ""

    asyncio.run(run_case())


def test_resume_acknowledges_post_end_turn_request_without_resampling():
    async def run_case():
        checkpoint, saved_ids, _ = await _stage1_checkpoint()
        async with FakeSGLangServer([[(-0.1, 821)]]) as sglang:
            tokenizer = FakeTokenizer(outputs={(821,): "done"})
            adapter = SegmentedAnthropicAdapter(tokenizer=tokenizer, sglang_url=sglang.url)
            adapter.open_session("post-end", resume_checkpoint=checkpoint)
            client = TestClient(TestServer(adapter.app))
            await client.start_server()
            try:
                first = await client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer post-end"},
                    json={"max_tokens": 16, "tools": TOOLS, "messages": []},
                )
                second = await client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer post-end"},
                    json={
                        "max_tokens": 16,
                        "tools": TOOLS,
                        "messages": [
                            {"role": "assistant", "content": [{"type": "text", "text": "done"}]}
                        ],
                    },
                )
                second_data = await second.json()
                status = adapter.resume_status("post-end")
            finally:
                await client.close()
            await adapter.finish_session("post-end")

        assert first.status == second.status == 200
        assert sglang.requests[0]["input_ids"] == saved_ids
        assert len(sglang.requests) == 1
        assert second_data["stop_reason"] == "end_turn"
        assert second_data["content"] == [{"type": "text", "text": ""}]
        assert status["exact_request_count"] == 1
        assert status["post_end_turn_ack_count"] == 1
        assert status["error"] == ""

    asyncio.run(run_case())


def test_resume_keeps_checkpoint_tool_schema_authoritative_and_audits_drift():
    async def run_case():
        checkpoint, saved_ids, _ = await _stage1_checkpoint()
        runtime_tools = copy.deepcopy(TOOLS)
        runtime_tools.reverse()
        runtime_tools[0]["description"] = "runtime-only description"
        async with FakeSGLangServer([[(-0.1, 831)]]) as sglang:
            tokenizer = FakeTokenizer(outputs={(831,): "done"})
            adapter = SegmentedAnthropicAdapter(tokenizer=tokenizer, sglang_url=sglang.url)
            adapter.open_session("schema-drift", resume_checkpoint=checkpoint)
            client = TestClient(TestServer(adapter.app))
            await client.start_server()
            try:
                response = await client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer schema-drift"},
                    json={"max_tokens": 16, "tools": runtime_tools, "messages": []},
                )
                status = adapter.resume_status("schema-drift")
            finally:
                await client.close()
            await adapter.finish_session("schema-drift")

        assert response.status == 200
        assert sglang.requests[0]["input_ids"] == saved_ids
        assert status["runtime_tool_schema_mismatch_count"] == 1
        assert status["error"] == ""

    asyncio.run(run_case())


def test_resume_forwards_missing_runtime_tool_and_consumes_claude_code_error_result():
    async def run_case():
        checkpoint, _, _ = await _stage1_checkpoint()
        branch_tool = _tool_xml("Read", "file_path", "/repo/b.py")
        async with FakeSGLangServer([[(-0.1, 841)], [(-0.1, 842)]]) as sglang:
            tokenizer = FakeTokenizer(outputs={(841,): branch_tool, (842,): "recovered"})
            adapter = SegmentedAnthropicAdapter(tokenizer=tokenizer, sglang_url=sglang.url)
            adapter.open_session("missing-runtime-tool", resume_checkpoint=checkpoint)
            client = TestClient(TestServer(adapter.app))
            await client.start_server()
            try:
                response = await client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer missing-runtime-tool"},
                    json={"max_tokens": 16, "tools": [TOOLS[1]], "messages": []},
                )
                data = await response.json()
                tool_use = next(block for block in data["content"] if block["type"] == "tool_use")
                continued = await client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer missing-runtime-tool"},
                    json={
                        "max_tokens": 16,
                        "tools": [TOOLS[1]],
                        "messages": [
                            {"role": "assistant", "content": data["content"]},
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": tool_use["id"],
                                        "content": (
                                            "<tool_use_error>Error: No such tool available: "
                                            "Read</tool_use_error>"
                                        ),
                                        "is_error": True,
                                    }
                                ],
                            },
                        ],
                    },
                )
                status = adapter.resume_status("missing-runtime-tool")
            finally:
                await client.close()
            await adapter.finish_session("missing-runtime-tool")

        assert response.status == continued.status == 200
        assert status["generated_runtime_tool_unavailable_count"] == 1
        assert status["completed_tool_use_ids"] == [tool_use["id"]]
        assert status["exact_request_count"] == 2
        assert status["error"] == ""
        assert len(sglang.requests) == 2

    asyncio.run(run_case())


def test_resume_accepts_parallel_results_and_preserves_failed_result_content():
    async def run_case():
        checkpoint, saved_ids, _ = await _stage1_checkpoint()
        parallel = _tool_xml("Read", "file_path", "/repo/a.py") + _tool_xml("Bash", "command", "exit 7")
        async with FakeSGLangServer([[(-0.1, 901)], [(-0.1, 902)]]) as sglang:
            tokenizer = FakeTokenizer(outputs={(901,): parallel, (902,): "recovered"})
            adapter = SegmentedAnthropicAdapter(tokenizer=tokenizer, sglang_url=sglang.url)
            adapter.open_session("parallel", resume_checkpoint=checkpoint)
            client = TestClient(TestServer(adapter.app))
            await client.start_server()
            try:
                first = await client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer parallel"},
                    json={
                        "model": "m",
                        "max_tokens": 16,
                        "tools": TOOLS,
                        "messages": [{"role": "user", "content": "ignored"}],
                    },
                )
                first_data = await first.json()
                uses = [block for block in first_data["content"] if block["type"] == "tool_use"]
                assert len(uses) == 2
                second = await client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer parallel"},
                    json={
                        "model": "m",
                        "max_tokens": 16,
                        "tools": TOOLS,
                        "messages": [
                            {"role": "assistant", "content": copy.deepcopy(uses)},
                            {
                                "role": "user",
                                # Reverse wire order: the adapter must append in
                                # model-generation order, not arrival order.
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": uses[1]["id"],
                                        "content": "Exit code 7",
                                        "is_error": True,
                                    },
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": uses[0]["id"],
                                        "content": "source text",
                                    },
                                ],
                            },
                        ],
                    },
                )
                status = adapter.resume_status("parallel")
            finally:
                await client.close()
            await adapter.finish_session("parallel")

        assert first.status == second.status == 200
        assert sglang.requests[0]["input_ids"] == saved_ids
        rendered_messages = tokenizer.rendered[-1][0]
        assert rendered_messages[-2:] == [
            {"role": "tool", "content": "source text"},
            {"role": "tool", "content": "Exit code 7"},
        ]
        assert status["error"] == "" and status["exact_request_count"] == 2

    asyncio.run(run_case())


def test_resume_missing_tool_result_fails_closed_without_sampling_again():
    async def run_case():
        checkpoint, _, _ = await _stage1_checkpoint()
        parallel = _tool_xml("Read", "file_path", "/repo/a.py") + _tool_xml("Bash", "command", "exit 7")
        async with FakeSGLangServer([[(-0.1, 1001)]]) as sglang:
            tokenizer = FakeTokenizer(outputs={(1001,): parallel})
            adapter = SegmentedAnthropicAdapter(tokenizer=tokenizer, sglang_url=sglang.url)
            adapter.open_session("missing", resume_checkpoint=checkpoint)
            client = TestClient(TestServer(adapter.app))
            await client.start_server()
            try:
                first = await client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer missing"},
                    json={"max_tokens": 16, "tools": TOOLS, "messages": []},
                )
                first_data = await first.json()
                uses = [block for block in first_data["content"] if block["type"] == "tool_use"]
                second = await client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer missing"},
                    json={
                        "max_tokens": 16,
                        "tools": TOOLS,
                        "messages": [
                            {"role": "assistant", "content": copy.deepcopy(uses)},
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": uses[0]["id"],
                                        "content": "only one result",
                                    }
                                ],
                            },
                        ],
                    },
                )
                error = await second.json()
                status = adapter.resume_status("missing")
            finally:
                await client.close()
            await adapter.finish_session("missing")

        assert first.status == 200
        assert second.status == 409
        assert "expected one result" in error["error"]["message"]
        assert "expected one result" in status["error"]
        assert len(sglang.requests) == 1

    asyncio.run(run_case())


def test_resume_allows_completed_stage2_tool_ids_in_later_full_history():
    async def run_case():
        checkpoint, _, _ = await _stage1_checkpoint()
        first_xml = _tool_xml("Read", "file_path", "/repo/a.py")
        second_xml = _tool_xml("Bash", "command", "true")
        async with FakeSGLangServer(
            [[(-0.1, 1101)], [(-0.1, 1102)], [(-0.1, 1103)]]
        ) as sglang:
            tokenizer = FakeTokenizer(
                outputs={(1101,): first_xml, (1102,): second_xml, (1103,): "done"}
            )
            adapter = SegmentedAnthropicAdapter(tokenizer=tokenizer, sglang_url=sglang.url)
            adapter.open_session("three-turn", resume_checkpoint=checkpoint)
            client = TestClient(TestServer(adapter.app))
            await client.start_server()
            try:
                first = await client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer three-turn"},
                    json={"max_tokens": 16, "tools": TOOLS, "messages": []},
                )
                use1 = next(block for block in (await first.json())["content"] if block["type"] == "tool_use")
                history = [
                    {"role": "assistant", "content": [copy.deepcopy(use1)]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": use1["id"], "content": "source"}
                        ],
                    },
                ]
                second = await client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer three-turn"},
                    json={"max_tokens": 16, "tools": TOOLS, "messages": copy.deepcopy(history)},
                )
                use2 = next(block for block in (await second.json())["content"] if block["type"] == "tool_use")
                history.extend(
                    [
                        {"role": "assistant", "content": [copy.deepcopy(use2)]},
                        {
                            "role": "user",
                            "content": [
                                {"type": "tool_result", "tool_use_id": use2["id"], "content": "ok"}
                            ],
                        },
                    ]
                )
                third = await client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer three-turn"},
                    json={"max_tokens": 16, "tools": TOOLS, "messages": history},
                )
                third_data = await third.json()
                status = adapter.resume_status("three-turn")
            finally:
                await client.close()
            await adapter.finish_session("three-turn")

        assert first.status == second.status == third.status == 200
        assert third_data["content"] == [{"type": "text", "text": "done"}]
        assert status["completed_tool_use_ids"] == [use1["id"], use2["id"]]
        assert status["exact_request_count"] == 3 and status["error"] == ""
        assert len(sglang.requests) == 3

    asyncio.run(run_case())
