#!/usr/bin/env python3
"""Minimal SegmentedAnthropicAdapter HTTP server for L2 / smoke deploy.

Env:
  HF_CHECKPOINT       — tokenizer / model path (required)
  SGLANG_URL          — default http://127.0.0.1:${SGLANG_PORT}
  SGLANG_PORT         — default 30000 (used if SGLANG_URL unset)
  SLIME_ADAPTER_BIND_HOST — default 0.0.0.0
  SLIME_ADAPTER_PORT / ADAPTER_PORT — default 18001
  SGLANG_TOOL_CALL_PARSER / SGLANG_REASONING_PARSER — optional
"""

from __future__ import annotations

import logging
import os
import sys

from slime.agent.adapters.anthropic_segmented import SegmentedAnthropicAdapter
from slime.agent.aiohttp_threaded import FilteredAccessLogger, run_app_in_thread
from slime.utils.processing_utils import load_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("serve_adapter")


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def main() -> int:
    hf = _env("HF_CHECKPOINT")
    if not hf:
        logger.error("HF_CHECKPOINT is required")
        return 2

    sglang_port = int(_env("SGLANG_PORT", "30000") or "30000")
    sglang_url = _env("SGLANG_URL") or f"http://127.0.0.1:{sglang_port}"
    bind_host = _env("SLIME_ADAPTER_BIND_HOST", "0.0.0.0") or "0.0.0.0"
    bind_port = int(_env("ADAPTER_PORT") or _env("SLIME_ADAPTER_PORT", "18001") or "18001")

    logger.info("loading tokenizer from %s", hf)
    tokenizer = load_tokenizer(hf, trust_remote_code=True)
    adapter = SegmentedAnthropicAdapter(
        tokenizer=tokenizer,
        sglang_url=sglang_url,
        tool_parser=_env("SGLANG_TOOL_CALL_PARSER") or None,
        reasoning_parser=_env("SGLANG_REASONING_PARSER") or None,
    )
    logger.info("starting adapter on %s:%s -> %s", bind_host, bind_port, sglang_url)
    handle = run_app_in_thread(
        adapter.app,
        host=bind_host,
        port=bind_port,
        thread_name="cc-ags-l2-adapter",
        runner_kwargs={
            "handler_cancellation": True,
            "access_log_class": FilteredAccessLogger,
        },
    )
    logger.info("adapter ready port=%s /health", handle.port)
    # Block forever (Deployment restartPolicy)
    try:
        import time

        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
