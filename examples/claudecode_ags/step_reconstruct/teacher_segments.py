"""Strictly encode authoritative Teacher continuations as student SFT samples."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from slime.agent.adapters.anthropic_segmented import canonical_sha256, prompt_ids_sha256
from slime.utils.types import Sample

# Lazily imported; tests may monkeypatch this name.
MultiTurnLossMaskGenerator = None


class TeacherContextIntegrityError(ValueError):
    """Teacher supervision is not conditioned on the registered checkpoint."""

    bucket = "context_integrity"


def _integrity_error(code: str, detail: str) -> TeacherContextIntegrityError:
    return TeacherContextIntegrityError(f"teacher_context_integrity:{code}:{detail}")


def _assistant_message(row: dict[str, Any]) -> dict[str, Any]:
    response = row.get("response") or {}
    message = response.get("message")
    if isinstance(message, dict) and message.get("role") == "assistant":
        return copy.deepcopy(message)
    raise _integrity_error("missing_assistant", "response.message must be an assistant message")


def _prompt_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    prompt = row.get("prompt") or {}
    messages = prompt.get("messages")
    if not isinstance(messages, list):
        raise _integrity_error("missing_prompt_messages", "prompt.messages must be a list")
    return messages


def _checkpoint_field(checkpoint: Any, name: str, default: Any = None) -> Any:
    if isinstance(checkpoint, dict):
        return checkpoint.get(name, default)
    return getattr(checkpoint, name, default)


def _ordered_rows(sft_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not sft_rows:
        raise _integrity_error("empty_attempt", "Teacher SFT rows are empty")
    rows: list[dict[str, Any]] = []
    for position, row in enumerate(sft_rows):
        if not isinstance(row, dict):
            raise _integrity_error("invalid_row", f"row {position} is not an object")
        turn_index = row.get("turn_index")
        if isinstance(turn_index, bool) or not isinstance(turn_index, int):
            raise _integrity_error(
                "invalid_turn_index", f"row {position} has turn_index={turn_index!r}"
            )
        rows.append(row)
    rows.sort(key=lambda row: row["turn_index"])
    indices = [row["turn_index"] for row in rows]
    expected = list(range(len(rows)))
    if indices != expected:
        raise _integrity_error(
            "non_contiguous_turns", f"expected turn_index={expected}, got {indices}"
        )
    return rows


def _validate_attempt_and_checkpoint(
    checkpoint: Any,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = _ordered_rows(rows)
    checkpoint_id = str(_checkpoint_field(checkpoint, "checkpoint_id", "") or "")
    checkpoint_prompt_sha256 = str(
        _checkpoint_field(checkpoint, "prompt_sha256", "") or ""
    )
    checkpoint_prompt_ids = [
        int(value) for value in (_checkpoint_field(checkpoint, "prompt_ids", []) or [])
    ]
    checkpoint_messages = list(_checkpoint_field(checkpoint, "chat_messages", []) or [])
    checkpoint_messages_sha256 = canonical_sha256(checkpoint_messages)
    checkpoint_tools = _checkpoint_field(checkpoint, "tools_schema", None)
    checkpoint_tools_sha256 = canonical_sha256(checkpoint_tools)
    declared_tools_sha256 = str(_checkpoint_field(checkpoint, "tools_sha256", "") or "")
    if not checkpoint_id or not checkpoint_prompt_sha256:
        raise _integrity_error(
            "invalid_checkpoint", "checkpoint_id and prompt_sha256 are required"
        )
    actual_prompt_sha256 = prompt_ids_sha256(checkpoint_prompt_ids)
    if checkpoint_prompt_sha256 != actual_prompt_sha256:
        raise _integrity_error(
            "checkpoint_prompt_hash",
            f"declared={checkpoint_prompt_sha256} actual={actual_prompt_sha256}",
        )
    if declared_tools_sha256 != checkpoint_tools_sha256:
        raise _integrity_error(
            "checkpoint_tools_hash",
            f"declared={declared_tools_sha256} actual={checkpoint_tools_sha256}",
        )

    attempt_ids: set[str] = set()
    session_hashes: set[str] = set()
    for row in rows:
        turn_index = row["turn_index"]
        if row.get("version") != 2:
            raise _integrity_error(
                "unsupported_log_version",
                f"turn {turn_index} has version={row.get('version')!r}; expected 2",
            )
        conditioning = row.get("conditioning")
        if not isinstance(conditioning, dict):
            raise _integrity_error("missing_conditioning", f"turn {turn_index}")
        if conditioning.get("kind") != "checkpoint_authoritative":
            raise _integrity_error(
                "conditioning_kind",
                f"turn {turn_index} has kind={conditioning.get('kind')!r}",
            )
        expected_fields = {
            "checkpoint_id": checkpoint_id,
            "checkpoint_prompt_sha256": checkpoint_prompt_sha256,
            "checkpoint_messages_sha256": checkpoint_messages_sha256,
        }
        for name, expected_value in expected_fields.items():
            actual_value = str(conditioning.get(name) or "")
            if actual_value != expected_value:
                raise _integrity_error(
                    "checkpoint_binding",
                    f"turn {turn_index} {name}={actual_value!r}, expected {expected_value!r}",
                )
        attempt_id = str(conditioning.get("attempt_id") or "")
        if not attempt_id:
            raise _integrity_error("missing_attempt_id", f"turn {turn_index}")
        attempt_ids.add(attempt_id)

        session_hash = str(row.get("session_id_sha256") or "")
        if not session_hash:
            raise _integrity_error("missing_session_hash", f"turn {turn_index}")
        session_hashes.add(session_hash)

        prompt = row.get("prompt")
        if not isinstance(prompt, dict):
            raise _integrity_error("missing_prompt", f"turn {turn_index}")
        row_tools = prompt.get("tools_schema")
        actual_row_tools_sha256 = canonical_sha256(row_tools)
        declared_row_tools_sha256 = str(prompt.get("tools_sha256") or "")
        if declared_row_tools_sha256 != actual_row_tools_sha256:
            raise _integrity_error(
                "row_tools_hash",
                f"turn {turn_index} declared={declared_row_tools_sha256} "
                f"actual={actual_row_tools_sha256}",
            )
        if actual_row_tools_sha256 != checkpoint_tools_sha256:
            raise _integrity_error(
                "tools_schema_mismatch",
                f"turn {turn_index} row={actual_row_tools_sha256} "
                f"checkpoint={checkpoint_tools_sha256}",
            )

    if len(attempt_ids) != 1:
        raise _integrity_error("mixed_attempts", f"attempt_ids={sorted(attempt_ids)}")
    if len(session_hashes) != 1:
        raise _integrity_error("mixed_sessions", f"count={len(session_hashes)}")
    return rows


def rebuild_messages_from_teacher_turns(
    checkpoint_messages: list[dict[str, Any]],
    sft_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rebuild one exact checkpoint → assistant → tool-result message chain.

    There is deliberately no positional fallback.  If a logged Teacher prompt
    is not exactly the history built so far, the relabel is invalid.
    """
    rows = _ordered_rows(sft_rows)
    messages = copy.deepcopy(list(checkpoint_messages))
    for index, row in enumerate(rows):
        prompt_messages = _prompt_messages(row)
        if prompt_messages != messages:
            raise _integrity_error(
                "prompt_history_mismatch",
                f"turn {row['turn_index']} expected={canonical_sha256(messages)} "
                f"actual={canonical_sha256(prompt_messages)}",
            )
        messages.append(_assistant_message(row))
        if index + 1 >= len(rows):
            continue

        next_prompt = _prompt_messages(rows[index + 1])
        if len(next_prompt) < len(messages) or next_prompt[: len(messages)] != messages:
            raise _integrity_error(
                "next_prompt_prefix_mismatch",
                f"turn {rows[index + 1]['turn_index']} rebuilt={canonical_sha256(messages)} "
                f"actual={canonical_sha256(next_prompt)}",
            )
        interstitial = copy.deepcopy(next_prompt[len(messages) :])
        invalid_roles = [
            message.get("role") if isinstance(message, dict) else type(message).__name__
            for message in interstitial
            if not isinstance(message, dict) or message.get("role") != "tool"
        ]
        if invalid_roles:
            raise _integrity_error(
                "unexpected_interstitial",
                f"turn {rows[index + 1]['turn_index']} roles={invalid_roles}",
            )
        messages.extend(interstitial)
    return messages


def _qwen3_5_checkpoint_preserving_suffix(
    *,
    tokenizer,
    checkpoint_messages: list[dict[str, Any]],
    full_messages: list[dict[str, Any]],
    tools: Any,
    prompt_ids: list[int],
) -> tuple[list[int], list[int]]:
    """Encode only the post-checkpoint text so BPE cannot rewrite the prefix.

    A complete chat-template render can merge the checkpoint's final token with
    the first character of the Teacher response (for example ``"\n" + "\n"``).
    The text is still an exact prefix, but the canonical full-string BPE ids are
    no longer a token-array prefix.  Preserve the authoritative checkpoint ids
    and tokenize only the exact rendered suffix.
    """
    checkpoint_text = tokenizer.apply_chat_template(
        checkpoint_messages,
        tokenize=False,
        tools=tools,
        add_generation_prompt=True,
        return_dict=False,
    )
    if not isinstance(checkpoint_text, str):
        raise _integrity_error("checkpoint_render_type", type(checkpoint_text).__name__)
    checkpoint_render_ids = tokenizer(
        checkpoint_text,
        add_special_tokens=False,
    )["input_ids"]
    if list(checkpoint_render_ids) != prompt_ids:
        raise _integrity_error(
            "checkpoint_render_mismatch",
            f"rendered={len(checkpoint_render_ids)} checkpoint={len(prompt_ids)}",
        )

    rendered_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        tools=tools,
        return_dict=False,
    )
    if not isinstance(rendered_text, str):
        raise _integrity_error("continuation_render_type", type(rendered_text).__name__)
    if not rendered_text.startswith(checkpoint_text):
        raise _integrity_error(
            "rendered_text_prefix_mismatch",
            f"checkpoint_chars={len(checkpoint_text)} rendered_chars={len(rendered_text)}",
        )

    # Match MultiTurnLossMaskGenerator.gen_multi_turn_loss_mask_qwen3_5,
    # but project the character mask only onto the separately-tokenized suffix.
    assistant_header = "<|im_start|>assistant\n"
    think_prefix = "<think>\n"
    end_marker = "<|im_end|>"
    char_mask = [0] * len(rendered_text)
    cursor = 0
    for message in full_messages:
        if message.get("role") != "assistant":
            continue
        header_pos = rendered_text.find(assistant_header, cursor)
        if header_pos < 0:
            raise _integrity_error(
                "assistant_render_missing",
                f"cursor={cursor}",
            )
        content_start = header_pos + len(assistant_header)
        end_pos = rendered_text.find(end_marker, content_start)
        if end_pos < 0:
            raise _integrity_error(
                "assistant_end_missing",
                f"content_start={content_start}",
            )
        span_end = end_pos + len(end_marker)
        if span_end < len(rendered_text) and rendered_text[span_end] == "\n":
            span_end += 1
        cursor = span_end
        if message.get("step_loss_mask", 1) != 1:
            continue
        mask_start = content_start
        if rendered_text.startswith(think_prefix, content_start):
            mask_start += len(think_prefix)
        for position in range(mask_start, span_end):
            char_mask[position] = 1

    suffix_start = len(checkpoint_text)
    suffix_text = rendered_text[suffix_start:]
    char_mask_prefix_sum = [0]
    for value in char_mask:
        char_mask_prefix_sum.append(char_mask_prefix_sum[-1] + value)
    suffix_encoding = tokenizer(
        suffix_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    suffix_ids = [int(value) for value in suffix_encoding["input_ids"]]
    offsets = suffix_encoding.get("offset_mapping")
    if offsets is None:
        raise _integrity_error(
            "suffix_offsets_missing",
            "Qwen3.5 requires a fast tokenizer",
        )
    suffix_mask: list[int] = []
    for start, end in offsets:
        if end <= start:
            suffix_mask.append(0)
            continue
        global_start = suffix_start + int(start)
        global_end = suffix_start + int(end)
        masked_chars = char_mask_prefix_sum[global_end] - char_mask_prefix_sum[global_start]
        suffix_mask.append(1 if masked_chars > 0 else 0)

    token_ids = list(prompt_ids) + suffix_ids
    loss_mask = [0] * len(prompt_ids) + suffix_mask
    decoded = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded != rendered_text:
        raise _integrity_error(
            "checkpoint_suffix_roundtrip_mismatch",
            f"decoded_chars={len(decoded)} rendered_chars={len(rendered_text)}",
        )
    return token_ids, loss_mask


def encode_teacher_continuation_sample(
    *,
    sample: Sample,
    tokenizer,
    checkpoint: Any,
    sft_rows: list[dict[str, Any]],
    loss_mask_type: str = "qwen3",
    metadata: dict[str, Any] | None = None,
) -> Sample:
    """Build one SFT sample only after every context-integrity check passes."""
    global MultiTurnLossMaskGenerator
    if MultiTurnLossMaskGenerator is None:
        from slime.utils.mask_utils import MultiTurnLossMaskGenerator as _MaskGen

        MultiTurnLossMaskGenerator = _MaskGen

    checkpoint_messages = list(_checkpoint_field(checkpoint, "chat_messages", []) or [])
    tools = _checkpoint_field(checkpoint, "tools_schema", None)
    prompt_ids = list(_checkpoint_field(checkpoint, "prompt_ids", []) or [])
    if not prompt_ids:
        raise _integrity_error("missing_prompt_ids", "checkpoint.prompt_ids is required")

    rows = _validate_attempt_and_checkpoint(checkpoint, sft_rows)
    full_messages = rebuild_messages_from_teacher_turns(checkpoint_messages, rows)
    mask_gen = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=loss_mask_type)
    token_ids, loss_mask = mask_gen.get_loss_mask(full_messages, tools=tools)
    if len(token_ids) != len(loss_mask):
        raise _integrity_error(
            "token_mask_length",
            f"tokens={len(token_ids)} mask={len(loss_mask)}",
        )

    prefix_len = len(prompt_ids)
    if len(token_ids) < prefix_len:
        raise _integrity_error(
            "rendered_shorter_than_checkpoint",
            f"rendered={len(token_ids)} checkpoint={prefix_len}",
        )
    preserved_suffix_boundary = False
    if token_ids[:prefix_len] != prompt_ids:
        if loss_mask_type != "qwen3_5":
            raise _integrity_error(
                "token_prefix_mismatch",
                f"checkpoint_tokens={prefix_len}",
            )
        token_ids, loss_mask = _qwen3_5_checkpoint_preserving_suffix(
            tokenizer=tokenizer,
            checkpoint_messages=checkpoint_messages,
            full_messages=full_messages,
            tools=tools,
            prompt_ids=prompt_ids,
        )
        preserved_suffix_boundary = True

    # Never train on the checkpoint prefix, even if the mask generator marked it.
    loss_mask = [0] * prefix_len + [int(value) for value in loss_mask[prefix_len:]]
    if 1 not in loss_mask:
        raise _integrity_error("no_trainable_tokens", "Teacher continuation is empty")

    response_length = mask_gen.get_response_lengths([loss_mask])[0]
    out = copy.copy(sample)
    out.tokens = list(token_ids)
    out.response_length = int(response_length)
    out.loss_mask = list(loss_mask[-response_length:])
    out.rollout_log_probs = [0.0] * out.response_length
    out.response = tokenizer.decode(token_ids[-response_length:], skip_special_tokens=False)
    out.reward = 0.0
    out.status = Sample.Status.COMPLETED
    out.metadata = {
        **(sample.metadata or {}),
        **(metadata or {}),
        "sample_kind": "teacher_sft",
        "teacher_num_turns": len(rows),
        "teacher_prefix_tokens": prefix_len,
        "teacher_trainable_tokens": int(sum(out.loss_mask)),
        "teacher_checkpoint_id": str(_checkpoint_field(checkpoint, "checkpoint_id", "") or ""),
        "teacher_attempt_id": str(rows[0]["conditioning"]["attempt_id"]),
        "teacher_context_integrity": True,
        "teacher_checkpoint_suffix_tokenization": preserved_suffix_boundary,
    }
    return out


def load_sft_turn_rows(path: str) -> list[dict[str, Any]]:
    """Load one unique Teacher attempt and reject malformed/mixed histories."""
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise _integrity_error("invalid_json", f"line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise _integrity_error("invalid_row", f"line {line_number} is not an object")
        rows.append(row)
    rows = _ordered_rows(rows)
    suffix = ".sft_turns.jsonl"
    name = Path(path).name
    stem = name[: -len(suffix)] if name.endswith(suffix) else ""
    parts = stem.split(".")
    if len(parts) == 2 and all(len(value) == 16 for value in parts):
        expected_session_prefix, expected_attempt_id = parts
        for row in rows:
            turn_index = row["turn_index"]
            session_hash = str(row.get("session_id_sha256") or "")
            conditioning = row.get("conditioning") or {}
            attempt_id = str(conditioning.get("attempt_id") or "")
            if not session_hash.startswith(expected_session_prefix):
                raise _integrity_error(
                    "attempt_path_session_mismatch",
                    f"turn {turn_index} path={expected_session_prefix} row={session_hash[:16]}",
                )
            if attempt_id != expected_attempt_id:
                raise _integrity_error(
                    "attempt_path_id_mismatch",
                    f"turn {turn_index} path={expected_attempt_id} row={attempt_id}",
                )
    return rows
