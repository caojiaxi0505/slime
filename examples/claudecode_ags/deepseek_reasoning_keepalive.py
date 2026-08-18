"""Keep a remote OpenAI-compatible reasoning deployment warm.

This process is intentionally independent from Slime training.  It maintains a
fixed number of long-running reasoning requests and discards their text.  Only
status, latency, and token usage are logged.
"""

from __future__ import annotations

import concurrent.futures
import itertools
import json
import logging
import os
import random
import signal
import threading
import time
from typing import Any
from urllib import error, request


logger = logging.getLogger("deepseek-reasoning-keepalive")


HARD_PROMPTS = (
    """Give a rigorous solution to the following problem. Check every hidden
assumption, present at least two candidate approaches, reject any invalid
approach explicitly, and finish with a verification of the final result.

Let a_1,...,a_n be nonzero integers such that for every nonempty subset S of
{1,...,n}, the sum of a_i over i in S is nonzero. Determine the strongest
general lower bound you can prove for the number of distinct subset sums, and
characterize all equality cases. Do not rely on an unstated generic-position
assumption.""",
    """Analyze this distributed-systems design as if reviewing a production
protocol. Four replicas use quorum reads and writes, leases, retries, and
asynchronous replication. A client may retry the same write after losing the
response; clocks have bounded drift but unbounded offset; a replica may pause
for an arbitrary duration without losing memory. Derive the precise conditions
needed for linearizability and exactly-once effects. Construct counterexamples
for every condition you find necessary, then give a corrected protocol and a
clear safety argument.""",
    """Design an algorithm for maintaining the exact number of strongly
connected components under an online sequence of directed-edge insertions and
deletions. Start by deriving realistic lower bounds and explaining which target
complexities are impossible. Then propose the strongest correct algorithm you
can justify, prove its invariants, analyze worst-case and amortized complexity,
and work through an adversarial example that breaks a tempting naive method.""",
    """Consider policy-gradient training on variable-length trajectories with
a single terminal reward, token-mean loss normalization, and group-relative
advantages. Derive from first principles how trajectory length changes the
expected gradient and its variance. Separate bias from variance, analyze the
effect of reward/length correlation, give a concrete finite MDP counterexample
to an incorrect intuitive claim, and state an estimator that corrects the
identified issue together with its assumptions.""",
)


def _positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name) or default)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _load_env_file() -> None:
    """Load simple KEY=VALUE entries without copying secrets into Pod YAML."""
    path = (os.environ.get("SLIME_REASONING_KEEPALIVE_ENV_FILE") or "").strip()
    if not path or not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key.startswith("export "):
                key = key.removeprefix("export ").strip()
            if not key or key in os.environ:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            os.environ[key] = value


def _payload(model: str, prompt: str, max_tokens: int, request_id: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are solving a difficult reasoning task. Think deeply, "
                    "test competing hypotheses, and verify the answer before concluding."
                ),
            },
            {"role": "user", "content": f"request_id={request_id}\n\n{prompt}"},
        ],
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": max_tokens,
        "thinking": {"type": "enabled"},
        "reasoning_effort": "max",
    }


def _usage(data: Any) -> tuple[int, int]:
    if not isinstance(data, dict) or not isinstance(data.get("usage"), dict):
        return 0, 0
    usage = data["usage"]
    input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    return input_tokens, output_tokens


def _worker(
    worker_id: int,
    *,
    base_url: str,
    api_key: str,
    model: str,
    max_tokens: int,
    timeout_sec: int,
    stop: threading.Event,
) -> None:
    prompt_cycle = itertools.cycle(HARD_PROMPTS[worker_id:] + HARD_PROMPTS[:worker_id])
    sequence = 0
    while not stop.is_set():
        sequence += 1
        request_id = f"keepalive-w{worker_id}-n{sequence}-{time.time_ns()}"
        payload = _payload(model, next(prompt_cycle), max_tokens, request_id)
        started = time.monotonic()
        http_request = request.Request(
            f"{base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=timeout_sec) as response:
                raw = response.read().decode("utf-8", errors="replace")
            elapsed = time.monotonic() - started
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning(
                    "worker=%d request=%d invalid_json elapsed=%.1fs",
                    worker_id,
                    sequence,
                    elapsed,
                )
                stop.wait(2.0)
                continue
            input_tokens, output_tokens = _usage(data)
            logger.info(
                "worker=%d request=%d ok elapsed=%.1fs input_tokens=%d output_tokens=%d",
                worker_id,
                sequence,
                elapsed,
                input_tokens,
                output_tokens,
            )
        except error.HTTPError as exc:
            logger.warning(
                "worker=%d request=%d status=%d elapsed=%.1fs",
                worker_id,
                sequence,
                exc.code,
                time.monotonic() - started,
            )
            stop.wait(min(30.0, 2.0 + random.random() * 3.0))
        except Exception as exc:
            logger.warning(
                "worker=%d request=%d exception=%s elapsed=%.1fs",
                worker_id,
                sequence,
                type(exc).__name__,
                time.monotonic() - started,
            )
            stop.wait(min(30.0, 2.0 + random.random() * 3.0))


def _run() -> None:
    _load_env_file()
    base_url = (os.environ.get("SLIME_REMOTE_OPENAI_BASE_URL") or "").strip().rstrip("/")
    api_key = (os.environ.get("SLIME_REMOTE_OPENAI_API_KEY") or "").strip()
    model = (os.environ.get("SLIME_REMOTE_OPENAI_MODEL") or "").strip()
    if not base_url or not api_key or not model:
        raise RuntimeError(
            "SLIME_REMOTE_OPENAI_BASE_URL, SLIME_REMOTE_OPENAI_API_KEY, and "
            "SLIME_REMOTE_OPENAI_MODEL are required"
        )

    inflight = _positive_int("SLIME_REASONING_KEEPALIVE_INFLIGHT", 4)
    max_tokens = _positive_int("SLIME_REASONING_KEEPALIVE_MAX_TOKENS", 16384)
    timeout_sec = _positive_int("SLIME_REASONING_KEEPALIVE_TIMEOUT_SEC", 1800)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda _signum, _frame: stop.set())

    logger.info(
        "starting model=%s inflight=%d max_tokens=%d timeout=%ds",
        model,
        inflight,
        max_tokens,
        timeout_sec,
    )
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=inflight,
        thread_name_prefix="reasoning-keepalive",
    ) as executor:
        workers = [
            executor.submit(
                _worker,
                worker_id,
                base_url=base_url,
                api_key=api_key,
                model=model,
                max_tokens=max_tokens,
                timeout_sec=timeout_sec,
                stop=stop,
            )
            for worker_id in range(inflight)
        ]
        stop.wait()
        for worker in workers:
            worker.cancel()
    logger.info("stopped")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    _run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
