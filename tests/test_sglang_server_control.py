import asyncio

from slime.backends.sglang_utils import server_control


def test_abort_server_requires_stable_idle(monkeypatch):
    # A late request appears after the first zero-load observation.  The drain
    # must reset its idle streak and only return after three later idle polls.
    loads = iter([0, 1, 0, 0, 0])
    abort_calls = []

    async def fake_post(url, payload):
        abort_calls.append((url, payload))

    async def fake_get(_url):
        return {"num_running_reqs": next(loads), "num_waiting_reqs": 0}

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(server_control, "post", fake_post)
    monkeypatch.setattr(server_control, "get", fake_get)
    monkeypatch.setattr(server_control.asyncio, "sleep", no_sleep)

    asyncio.run(
        server_control.abort_server_until_idle(
            "http://worker",
            retry_interval=0,
            idle_confirmations=3,
            idle_confirm_interval=0,
        )
    )

    assert len(abort_calls) == 5


def test_num_requests_from_nested_router_load():
    load = {
        "loads": [
            {"num_running_reqs": 2, "num_waiting_reqs": 1},
            {"num_total_reqs": 4},
        ]
    }
    assert server_control.num_requests_from_load(load) == 7
