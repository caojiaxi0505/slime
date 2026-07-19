"""Canonical Claude Code ``--input-format stream-json`` serialization."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any


def initial_user_event(prompt: str) -> dict[str, Any]:
    """Return the canonical stream-json event for a new task prompt."""
    return {
        "type": "user",
        "message": {
            "role": "user",
            "content": [{"type": "text", "text": str(prompt or "")}],
        },
    }


def encode_stream_events(events: Iterable[dict[str, Any]]) -> str:
    """Encode complete Claude Code input events as compact newline-delimited JSON."""
    return "".join(json.dumps(event, separators=(",", ":")) + "\n" for event in events)


def initial_prompt_jsonl(prompt: str) -> str:
    """Encode a new task prompt through the same stream protocol used by resume."""
    return encode_stream_events([initial_user_event(prompt)])
