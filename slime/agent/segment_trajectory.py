"""Segment fan-out helpers for compact/subagent rollouts."""

from __future__ import annotations

import copy
import dataclasses
import logging
from typing import Any

from slime.agent.trajectory import TurnRecord
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class TokenSegment:
    prompt_ids: list[int]
    response_ids: list[int]
    loss_mask: list[int]
    rollout_log_probs: list[float] = dataclasses.field(default_factory=list)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class TurnSegment:
    """A frozen group of turns before token-level merge."""

    turns: list[TurnRecord]
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


def make_turn_segment(
    turns: list[TurnRecord],
    *,
    kind: str = "",
    metadata: dict[str, Any] | None = None,
) -> TurnSegment:
    """Freeze turns and attach conventional segment metadata."""
    frozen_turns = list(turns)
    segment_metadata = dict(metadata or {})
    if kind:
        segment_metadata.setdefault("segment_kind", kind)
    segment_metadata.setdefault("finish_reason", frozen_turns[-1].finish_reason if frozen_turns else "")
    return TurnSegment(turns=frozen_turns, metadata=segment_metadata)


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _output_log_probs(turn: TurnRecord) -> list[float]:
    if len(turn.output_log_probs) == len(turn.output_ids):
        return list(turn.output_log_probs)
    logger.warning(
        "[segment_trajectory] turn logprob length mismatch; zeroing (%d ids, %d logprobs)",
        len(turn.output_ids),
        len(turn.output_log_probs),
    )
    return [0.0] * len(turn.output_ids)


def merge_turns(turns: list[TurnRecord], *, metadata: dict[str, Any] | None = None) -> TokenSegment | None:
    """Replay turn records into one linear training segment.

    The first turn's prompt becomes the segment prompt. Later turn prompts are
    stitched against ``prompt + response_so_far``. Any new prompt suffix is
    non-model context and receives loss mask 0.
    """
    if not turns:
        return None

    prompt_ids = list(turns[0].prompt_ids)
    response_ids: list[int] = []
    loss_mask: list[int] = []
    rollout_log_probs: list[float] = []
    output_spans: list[tuple[int, int]] = []

    for i, turn in enumerate(turns):
        if i > 0:
            if turn.prompt_ids[: len(prompt_ids)] != prompt_ids:
                logger.warning("[segment_trajectory] merge prompt base changed; restarting from drifted prompt")
                prompt_ids = list(turn.prompt_ids)
                response_ids = []
                loss_mask = []
                rollout_log_probs = []
                output_spans = []
            else:
                prompt_suffix = turn.prompt_ids[len(prompt_ids) :]
                matched_len = _common_prefix_len(response_ids, prompt_suffix)
                if matched_len < len(response_ids):
                    for start, end in output_spans:
                        if start < matched_len < end:
                            loss_mask[start:matched_len] = [0] * (matched_len - start)
                            rollout_log_probs[start:matched_len] = [0.0] * (matched_len - start)
                    response_ids = response_ids[:matched_len]
                    loss_mask = loss_mask[:matched_len]
                    rollout_log_probs = rollout_log_probs[:matched_len]
                    output_spans = [
                        (start, min(end, matched_len)) for start, end in output_spans if start < matched_len
                    ]

                context_tail = prompt_suffix[matched_len:]
                response_ids.extend(context_tail)
                loss_mask.extend([0] * len(context_tail))
                rollout_log_probs.extend([0.0] * len(context_tail))

        output_start = len(response_ids)
        response_ids.extend(turn.output_ids)
        loss_mask.extend([1] * len(turn.output_ids))
        rollout_log_probs.extend(_output_log_probs(turn))
        output_spans.append((output_start, len(response_ids)))

    rollout_log_probs = [logprob if mask else 0.0 for logprob, mask in zip(rollout_log_probs, loss_mask, strict=True)]

    segment_metadata = dict(metadata or {})
    # Keep the exact model-output boundaries after prompt drift/truncation has
    # been resolved.  Consumers can then change the training scope without
    # guessing turn boundaries from runs of 1s in ``loss_mask``.
    segment_metadata["assistant_output_spans"] = [list(span) for span in output_spans]
    segment_metadata["assistant_turn_count"] = len(output_spans)

    return TokenSegment(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        loss_mask=loss_mask,
        rollout_log_probs=rollout_log_probs,
        metadata=segment_metadata,
    )


def mask_to_first_assistant_turn(segments: list[TokenSegment]) -> tuple[list[TokenSegment], dict[str, int]]:
    """Keep loss only on the first trainable assistant turn.

    Response tokens, prompts, and metadata remain intact, so the complete
    continuation is still available for reward and auditing.  Only the loss
    mask and matching rollout log-probabilities are zeroed after the first
    assistant output span.
    """
    total_trainable = 0
    assistant_turns = 0
    first_location: tuple[int, int, int] | None = None

    for segment_idx, segment in enumerate(segments):
        if len(segment.loss_mask) != len(segment.response_ids):
            raise ValueError(
                "segment loss_mask length mismatch: "
                f"segment={segment_idx} response={len(segment.response_ids)} mask={len(segment.loss_mask)}"
            )
        if segment.rollout_log_probs and len(segment.rollout_log_probs) != len(segment.response_ids):
            raise ValueError(
                "segment rollout_log_probs length mismatch: "
                f"segment={segment_idx} response={len(segment.response_ids)} "
                f"logprobs={len(segment.rollout_log_probs)}"
            )

        raw_spans = segment.metadata.get("assistant_output_spans")
        if not isinstance(raw_spans, list):
            raise ValueError(f"segment {segment_idx} missing assistant_output_spans")

        previous_end = 0
        for turn_idx, raw_span in enumerate(raw_spans):
            if not isinstance(raw_span, (list, tuple)) or len(raw_span) != 2:
                raise ValueError(
                    f"segment {segment_idx} has invalid assistant span at turn {turn_idx}: {raw_span!r}"
                )
            start, end = int(raw_span[0]), int(raw_span[1])
            if start < previous_end or start < 0 or end < start or end > len(segment.response_ids):
                raise ValueError(
                    f"segment {segment_idx} has out-of-range assistant span at turn {turn_idx}: "
                    f"[{start}, {end}) for response length {len(segment.response_ids)}"
                )
            previous_end = end
            assistant_turns += 1
            if first_location is None and any(segment.loss_mask[start:end]):
                first_location = (segment_idx, start, end)
        total_trainable += sum(int(value) for value in segment.loss_mask)

    if first_location is None:
        raise ValueError("no trainable assistant turn found in continuation segments")

    scoped: list[TokenSegment] = []
    kept_trainable = 0
    for segment_idx, segment in enumerate(segments):
        new_mask = [0] * len(segment.loss_mask)
        if segment_idx == first_location[0]:
            start, end = first_location[1:]
            new_mask[start:end] = [int(value) for value in segment.loss_mask[start:end]]
        kept_trainable += sum(new_mask)

        if segment.rollout_log_probs:
            new_logprobs = [
                float(logprob) if mask else 0.0
                for logprob, mask in zip(segment.rollout_log_probs, new_mask, strict=True)
            ]
        else:
            new_logprobs = []
        scoped.append(
            dataclasses.replace(
                segment,
                loss_mask=new_mask,
                rollout_log_probs=new_logprobs,
            )
        )

    return scoped, {
        "assistant_turns": assistant_turns,
        "total_trainable_tokens": total_trainable,
        "kept_trainable_tokens": kept_trainable,
        "masked_trainable_tokens": total_trainable - kept_trainable,
        "first_turn_segment_idx": first_location[0],
    }


def merge_turn_segments(segments: list[TurnSegment]) -> list[TokenSegment]:
    """Merge frozen turn segments and keep every non-empty response."""
    out: list[TokenSegment] = []
    for turn_segment in segments:
        token_segment = merge_turns(turn_segment.turns, metadata=turn_segment.metadata)
        if token_segment is None:
            continue
        if token_segment.response_ids:
            out.append(token_segment)
    return out


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
