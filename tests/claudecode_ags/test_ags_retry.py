import asyncio
from types import SimpleNamespace

import aiohttp
import pytest

from slime.agent import sandbox
from slime.agent import sandbox_ags


def test_transient_request_retries_five_times_with_capped_schedule(monkeypatch):
    calls = 0
    sleeps = []

    async def operation():
        nonlocal calls
        calls += 1
        if calls <= 5:
            raise aiohttp.ServerDisconnectedError()
        return "ok"

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(sandbox_ags.random, "uniform", lambda _low, _high: 0.0)

    result = asyncio.run(
        sandbox_ags._retry_transient_request(
            operation,
            max_retries=5,
            delays=(15.0, 30.0, 60.0),
            description="test",
        )
    )

    assert result == "ok"
    assert calls == 6
    assert sleeps == [15.0, 30.0, 60.0, 60.0, 60.0]


def test_non_transient_request_is_not_retried(monkeypatch):
    calls = 0

    async def operation():
        nonlocal calls
        calls += 1
        raise RuntimeError("remote command failed")

    async def fail_sleep(_delay):
        raise AssertionError("non-transient errors must not sleep")

    monkeypatch.setattr(asyncio, "sleep", fail_sleep)

    with pytest.raises(RuntimeError, match="remote command failed"):
        asyncio.run(
            sandbox_ags._retry_transient_request(
                operation,
                max_retries=5,
                delays=(15.0, 30.0, 60.0),
                description="test",
            )
        )

    assert calls == 1


def test_done_marker_poll_defaults_to_fifteen_seconds(monkeypatch):
    sleeps = []

    class FakeSandbox:
        async def exec(self, *_args, **_kwargs):
            return 0, "0", ""

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.delenv("SLIME_AGENT_DONE_POLL_INTERVAL_SEC", raising=False)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    result = asyncio.run(
        sandbox._await_done_marker(
            FakeSandbox(),
            "/tmp/done",
            user="agent",
            time_budget_sec=60,
        )
    )

    assert result == 0
    assert sleeps == [15.0]


def test_execute_retries_reuse_request_id(monkeypatch):
    request_ids = []

    class FakeResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def json(self):
            return {"exit_code": 0}

    class FakeSession:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def post(self, _url, *, json, headers):
            del json
            request_ids.append(headers["X-Request-ID"])
            if len(request_ids) < 3:
                raise aiohttp.ServerDisconnectedError()
            return FakeResponse()

    class FakeRuntime:
        _api_url = "https://runtime.example"
        _headers = {"X-Access-Token": "secret"}

        async def _ensure_valid_token(self):
            return None

        async def _handle_response_errors(self, _response):
            return None

    class FakeCommand:
        def model_dump(self):
            return {"command": "true"}

    async def fake_sleep(_delay):
        return None

    monkeypatch.setattr(sandbox_ags.aiohttp, "ClientSession", FakeSession)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(sandbox_ags.random, "uniform", lambda _low, _high: 0.0)

    sb = sandbox_ags.AGSSandbox("image")
    sb._deployment = SimpleNamespace(runtime=FakeRuntime())
    sb._rex_command_response_cls = lambda **values: values
    result = asyncio.run(sb._execute_idempotent_with_retry(FakeCommand()))

    assert result == {"exit_code": 0}
    assert len(request_ids) == 3
    assert len(set(request_ids)) == 1
