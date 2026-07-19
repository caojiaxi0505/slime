"""Concurrency regressions for the Claude Code Anthropic adapter."""

from __future__ import annotations

import asyncio
import threading

import pytest
from aiohttp.test_utils import TestClient, TestServer

from slime.agent.adapters import anthropic_segmented as segmented
from slime.agent.trajectory import TurnRecord


class _BlockingTokenizer:
    name_or_path = "blocking-tokenizer"
    chat_template = "test-template"
    special_tokens_map: dict = {}
    vocab_size = 128

    def __init__(self, started: threading.Event, release: threading.Event):
        self.started = started
        self.release = release

    def __len__(self):
        return self.vocab_size

    def apply_chat_template(self, messages, **_kwargs):
        if "slow" in str(messages):
            self.started.set()
            if not self.release.wait(timeout=5):
                raise TimeoutError("test did not release slow tokenizer")
            return [11]
        return [12]

    def decode(self, _ids, **_kwargs):
        return "done"


async def _instant_generate(prompt_ids, _session, _body, **_kwargs):
    return TurnRecord(
        prompt_ids=list(prompt_ids),
        output_ids=[99],
        output_log_probs=[-0.1],
        finish_reason="stop",
    )


def test_slow_tokenizer_does_not_block_health_or_another_session(monkeypatch):
    async def run_case():
        started = threading.Event()
        release = threading.Event()
        tokenizer = _BlockingTokenizer(started, release)
        adapter = segmented.SegmentedAnthropicAdapter(
            tokenizer=tokenizer,
            sglang_url="http://unused",
            cpu_workers=2,
        )
        adapter.open_session("slow")
        adapter.open_session("fast")
        monkeypatch.setattr(segmented, "call_sglang_generate", _instant_generate)

        client = TestClient(TestServer(adapter.app))
        await client.start_server()
        slow_task = asyncio.create_task(
            client.post(
                "/v1/messages",
                headers={"Authorization": "Bearer slow"},
                json={"messages": [{"role": "user", "content": "slow"}]},
            )
        )
        try:
            started_ok = await asyncio.wait_for(
                asyncio.to_thread(started.wait, 2),
                timeout=3,
            )
            assert started_ok

            health = await asyncio.wait_for(client.get("/health"), timeout=0.5)
            assert health.status == 200

            fast = await asyncio.wait_for(
                client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer fast"},
                    json={"messages": [{"role": "user", "content": "fast"}]},
                ),
                timeout=1,
            )
            assert fast.status == 200
            assert not slow_task.done()
        finally:
            release.set()
            slow = await asyncio.wait_for(slow_task, timeout=2)
            assert slow.status == 200
            await client.close()

    asyncio.run(run_case())


def test_cpu_worker_finishes_before_cancelled_handler_can_release_state():
    async def run_case():
        started = threading.Event()
        release = threading.Event()
        adapter = segmented.SegmentedAnthropicAdapter(
            tokenizer=_BlockingTokenizer(started, release),
            sglang_url="http://unused",
            cpu_workers=1,
        )

        def blocked_work():
            started.set()
            release.wait(timeout=5)
            return "finished"

        task = asyncio.create_task(adapter._run_cpu(blocked_work))
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        await asyncio.sleep(0.05)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await adapter._cleanup_resources(adapter.app)

    asyncio.run(run_case())


def test_hot_path_hashes_each_wire_message_only_once(monkeypatch):
    async def run_case():
        started = threading.Event()
        release = threading.Event()
        release.set()
        adapter = segmented.SegmentedAnthropicAdapter(
            tokenizer=_BlockingTokenizer(started, release),
            sglang_url="http://unused",
            cpu_workers=1,
        )
        adapter.open_session("hash-once")
        monkeypatch.setattr(segmented, "call_sglang_generate", _instant_generate)
        original = segmented.message_hash
        calls = 0
        calls_lock = threading.Lock()

        def counted(value):
            nonlocal calls
            with calls_lock:
                calls += 1
            return original(value)

        monkeypatch.setattr(segmented, "message_hash", counted)
        client = TestClient(TestServer(adapter.app))
        await client.start_server()
        try:
            response = await client.post(
                "/v1/messages",
                headers={"Authorization": "Bearer hash-once"},
                json={
                    "system": "system",
                    "messages": [
                        {"role": "user", "content": "one"},
                        {"role": "assistant", "content": "two"},
                    ],
                },
            )
            assert response.status == 200
            assert calls == 3  # two messages plus the system block
        finally:
            await client.close()

    asyncio.run(run_case())


def test_timing_summary_is_attached_to_drained_segments(monkeypatch):
    async def run_case():
        started = threading.Event()
        release = threading.Event()
        release.set()
        adapter = segmented.SegmentedAnthropicAdapter(
            tokenizer=_BlockingTokenizer(started, release),
            sglang_url="http://unused",
            cpu_workers=2,
        )
        adapter.open_session("timed")
        monkeypatch.setattr(segmented, "call_sglang_generate", _instant_generate)
        client = TestClient(TestServer(adapter.app))
        await client.start_server()
        try:
            response = await client.post(
                "/v1/messages",
                headers={"Authorization": "Bearer timed"},
                json={"messages": [{"role": "user", "content": "fast"}]},
            )
            assert response.status == 200
            segments = await adapter.finish_session("timed")
            assert len(segments) == 1
            metadata = segments[0].metadata
            assert metadata["adapter_turn_count"] == 1
            assert metadata["adapter_cpu_workers"] == 2
            assert metadata["adapter_prepare_tokenize_ms_mean"] >= 0
            assert metadata["adapter_sglang_e2e_ms_mean"] >= 0
        finally:
            await client.close()

    asyncio.run(run_case())
