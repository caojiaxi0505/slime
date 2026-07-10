"""Segment fan-out helpers for compact/subagent rollouts."""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

from slime.utils.types import Sample


@dataclasses.dataclass(frozen=True)
class TokenSegment:
    prompt_ids: list[int]
    response_ids: list[int]
    loss_mask: list[int]
    rollout_log_probs: list[float] = dataclasses.field(default_factory=list)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


def write_segment_to_sample(sample: Sample, segment: TokenSegment, reward: float, tokenizer) -> None:
    sample.tokens = list(segment.prompt_ids) + list(segment.response_ids)
    sample.response_length = len(segment.response_ids)
    sample.loss_mask = list(segment.loss_mask)
    sample.rollout_log_probs = list(segment.rollout_log_probs)
    sample.response = tokenizer.decode(segment.response_ids, skip_special_tokens=False)
    sample.reward = float(reward)
    sample.status = Sample.Status.COMPLETED


def fan_out_sample_segments(
    sample: Sample,
    segments: list[TokenSegment],
    reward: float,
    tokenizer,
    *,
    metadata: dict[str, Any] | None = None,
    rollout_id: int | None = None,
) -> list[Sample]:
    """One Sample per segment; split reward evenly; share rollout_id."""
    k = len(segments)
    if k == 0:
        return []
    per = float(reward) / k
    shared = sample.index if rollout_id is None else rollout_id
    base_md = {**(sample.metadata or {}), **(metadata or {})}
    out: list[Sample] = []
    for i, segment in enumerate(segments):
        sub = sample if i == 0 else copy.copy(sample)
        write_segment_to_sample(sub, segment, per, tokenizer)
        sub.rollout_id = shared
        sub.metadata = {
            **base_md,
            **(segment.metadata or {}),
            "segment_idx": i,
            "num_segments": k,
        }
        out.append(sub)
    return out
