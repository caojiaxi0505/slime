import asyncio
import logging
from typing import Any

from slime.utils.http_utils import get, post

logger = logging.getLogger(__name__)

ABORT_RETRY_INTERVAL_SECONDS = 3
IDLE_CONFIRMATIONS = 3
IDLE_CONFIRM_INTERVAL_SECONDS = 1


def num_requests_from_load(load: Any) -> int:
    if isinstance(load, list):
        return sum(num_requests_from_load(item) for item in load)

    if not isinstance(load, dict):
        return 0

    if "loads" in load:
        return num_requests_from_load(load["loads"])

    for key in ("num_reqs", "num_total_reqs", "total_reqs"):
        value = load.get(key)
        if isinstance(value, int):
            return value

    running = load.get("num_running_reqs", load.get("total_running_reqs"))
    waiting = load.get("num_waiting_reqs", load.get("total_waiting_reqs"))
    return (running if isinstance(running, int) else 0) + (waiting if isinstance(waiting, int) else 0)


async def _abort_server_once(url: str) -> None:
    try:
        await post(f"{url}/abort_request", {"abort_all": True})
    except Exception as e:
        logger.warning(f"Failed to abort SGLang server at {url}: {e}")


async def _get_server_num_requests(url: str) -> int:
    return num_requests_from_load(await get(f"{url}/v1/loads?include=core"))


async def abort_server_until_idle(
    url: str,
    retry_interval: int = ABORT_RETRY_INTERVAL_SECONDS,
    *,
    idle_confirmations: int = IDLE_CONFIRMATIONS,
    idle_confirm_interval: int = IDLE_CONFIRM_INTERVAL_SECONDS,
) -> None:
    """Abort requests until the server stays idle across several polls.

    A single zero-load observation is insufficient: an already-running adapter
    handler can submit its next SGLang turn immediately after that observation.
    Re-aborting and requiring a short stable-idle window drains those late
    arrivals before cache flush/offload.  This does not quiesce the adapter;
    callers must still ensure agent producers have been terminated.
    """
    attempt = 1
    consecutive_idle = 0
    while True:
        logger.info(f"Abort request for SGLang server {url}")
        await _abort_server_once(url)

        try:
            num_requests = await _get_server_num_requests(url)
        except Exception as e:
            logger.warning(f"Failed to get SGLang server load from {url}: {e}")
            return

        if num_requests <= 0:
            consecutive_idle += 1
            if consecutive_idle >= idle_confirmations:
                return
            await asyncio.sleep(idle_confirm_interval)
            attempt += 1
            continue

        consecutive_idle = 0

        logger.info(
            f"SGLang server {url} still has {num_requests} requests after abort attempt {attempt}; "
            f"retrying in {retry_interval} seconds."
        )
        await asyncio.sleep(retry_interval)
        attempt += 1


async def abort_servers_until_idle(urls: list[str]) -> None:
    await asyncio.gather(*(abort_server_until_idle(url) for url in urls))
