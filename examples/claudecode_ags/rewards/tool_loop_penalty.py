"""Detect consecutive identical tool-call signatures in one episode."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any


TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>", re.DOTALL
)
PARAM_RE = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
ASSISTANT_RE = re.compile(r"<\|im_start\|>assistant\s*(.*?)<\|im_end\|>", re.DOTALL)


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def assistant_turns(response: str) -> Iterable[str]:
    """Yield assistant turns from one decoded adapter segment."""
    if not response:
        return
    first_marker = response.find("<|im_start|>")
    first_end = response.find("<|im_end|>")
    if first_end >= 0 and (first_marker < 0 or first_end < first_marker):
        yield response[:first_end]
    elif first_marker < 0:
        yield response
    for match in ASSISTANT_RE.finditer(response):
        yield match.group(1)


def tool_signature(turn: str) -> tuple[str, ...] | None:
    """Return an order-insensitive exact signature for one assistant turn."""
    calls: list[str] = []
    for function, body in TOOL_CALL_RE.findall(turn):
        parameters = sorted(
            (name.strip(), _normalize(value)) for name, value in PARAM_RE.findall(body)
        )
        calls.append(
            json.dumps(
                [function.strip(), parameters],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    return tuple(sorted(calls)) if calls else None


def longest_consecutive_tool_signature_run(responses: Iterable[str]) -> int:
    """Longest run of identical non-empty turn signatures across an episode."""
    longest = 0
    current = 0
    previous: tuple[str, ...] | None = None
    for response in responses:
        for turn in assistant_turns(response):
            signature = tool_signature(turn)
            if signature is not None and signature == previous:
                current += 1
            elif signature is not None:
                current = 1
            else:
                current = 0
            previous = signature
            longest = max(longest, current)
    return longest


def episode_reward(*, resolved: bool, timed_out: bool, tool_loop_penalized: bool) -> float:
    """Outcome reward with the repeated-tool-loop rule at highest priority."""
    if tool_loop_penalized:
        return -1.0
    if timed_out:
        return 0.5 if resolved else 0.0
    return 1.0 if resolved else 0.0


def adjust_episode_reward(
    *,
    base_reward: float,
    reward_details: dict[str, Any],
    resolved: bool,
    exit_code: Any,
    responses: Iterable[str],
    timeout_outcome_enabled: bool,
    tool_loop_enabled: bool,
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    """Apply the shared timeout/outcome levels, then the tool-loop override."""
    longest = longest_consecutive_tool_signature_run(responses) if tool_loop_enabled else 0
    penalized = longest >= 3
    timed_out = exit_code == -1

    reward = float(base_reward)
    if timeout_outcome_enabled:
        reward = episode_reward(
            resolved=resolved,
            timed_out=timed_out,
            tool_loop_penalized=penalized,
        )
    elif penalized:
        reward = -1.0

    audit = {
        "consecutive_tool_signature_max": longest,
        "tool_loop_penalized": penalized,
        "tool_loop_detection_enabled": tool_loop_enabled,
        "timeout_outcome_reward_enabled": timeout_outcome_enabled,
        "agent_timed_out": timed_out,
    }
    if timeout_outcome_enabled or tool_loop_enabled:
        reward_details = {
            **reward_details,
            "base_reward_before_outcome_adjustment": float(base_reward),
            "tool_loop_threshold": 3,
            **audit,
        }
    return reward, reward_details, audit
