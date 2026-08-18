"""Unit tests for teacher continuation → student SFT encoding."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from slime.agent.adapters.anthropic_segmented import canonical_sha256, prompt_ids_sha256
from examples.claudecode_ags.step_reconstruct.teacher_segments import (
    TeacherContextIntegrityError,
    encode_teacher_continuation_sample,
    load_sft_turn_rows,
    rebuild_messages_from_teacher_turns,
)
from slime.utils.types import Sample


class _FakeTokenizer:
    """Minimal chat-template tokenizer for mask-generator unit tests."""

    def apply_chat_template(self, messages, tokenize=True, tools=None, return_dict=False, **kwargs):
        del tools, return_dict, kwargs
        pieces: list[int] = []
        for message in messages:
            role = message.get("role")
            # Keep ids tiny and deterministic.
            if role == "system":
                pieces.extend([1, 2])
            elif role == "user":
                pieces.extend([3, 4])
            elif role == "assistant":
                text = message.get("content")
                if isinstance(text, list):
                    n = sum(len(str(block.get("text", ""))) for block in text if isinstance(block, dict))
                else:
                    n = len(str(text or ""))
                pieces.extend([5] + [10 + (i % 7) for i in range(max(n, 1))])
            else:
                pieces.extend([9])
        if tokenize:
            return pieces
        return " ".join(str(x) for x in pieces)

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [ord(c) % 50 for c in str(text)]}

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        return ",".join(str(x) for x in ids)


class _BoundaryMergingTokenizer:
    """Tokenizer whose canonical BPE merges a newline across the resume boundary."""

    _merged_newline = 1

    @classmethod
    def _encode(cls, text):
        ids = []
        offsets = []
        index = 0
        while index < len(text):
            if text.startswith("\n\n", index):
                ids.append(cls._merged_newline)
                offsets.append((index, index + 2))
                index += 2
            else:
                ids.append(ord(text[index]) + 10)
                offsets.append((index, index + 1))
                index += 1
        return ids, offsets

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        tools=None,
        return_dict=False,
        add_generation_prompt=False,
        **kwargs,
    ):
        del tools, return_dict, kwargs
        text = "USER<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n<think>\n"
        else:
            assert [message["role"] for message in messages] == ["user", "assistant"]
            text += "<|im_start|>assistant\n<think>\n\nreason<|im_end|>\n"
        return self._encode(text)[0] if tokenize else text

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        del add_special_tokens
        ids, offsets = self._encode(text)
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    def decode(
        self,
        ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        del skip_special_tokens, clean_up_tokenization_spaces
        pieces = []
        for token_id in ids:
            pieces.append("\n\n" if token_id == self._merged_newline else chr(token_id - 10))
        return "".join(pieces)


def _bind_v2_rows(checkpoint, rows, *, attempt_id="attempt-a", session_hash="session-a"):
    for row in rows:
        prompt = row.setdefault("prompt", {})
        prompt["tools_schema"] = checkpoint.tools_schema
        prompt["tools_sha256"] = canonical_sha256(checkpoint.tools_schema)
        row["version"] = 2
        row["session_id_sha256"] = session_hash
        row["conditioning"] = {
            "kind": "checkpoint_authoritative",
            "checkpoint_id": checkpoint.checkpoint_id,
            "checkpoint_prompt_sha256": checkpoint.prompt_sha256,
            "checkpoint_messages_sha256": canonical_sha256(checkpoint.chat_messages),
            "attempt_id": attempt_id,
        }
    return rows


def test_rebuild_messages_appends_assistant_and_tool_context():
    checkpoint_messages = [
        {"role": "user", "content": "fix bug"},
    ]
    rows = [
        {
            "turn_index": 0,
            "response": {"message": {"role": "assistant", "content": "edit a"}},
            "prompt": {"messages": checkpoint_messages},
        },
        {
            "turn_index": 1,
            "response": {"message": {"role": "assistant", "content": "edit b"}},
            "prompt": {
                "messages": [
                    {"role": "user", "content": "fix bug"},
                    {"role": "assistant", "content": "edit a"},
                    {"role": "tool", "content": "tool ok"},
                ]
            },
        },
    ]
    messages = rebuild_messages_from_teacher_turns(checkpoint_messages, rows)
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant"]
    assert messages[2]["content"] == "tool ok"
    assert messages[3]["content"] == "edit b"


def test_encode_teacher_continuation_masks_only_suffix(monkeypatch):
    # Bypass MultiTurnLossMaskGenerator internals with a deterministic stub.
    from examples.claudecode_ags.step_reconstruct import teacher_segments as mod

    seen = {}

    class _StubMaskGen:
        def __init__(self, *args, **kwargs):
            del args
            seen.update(kwargs)

        def get_loss_mask(self, messages, tools=None):
            del tools
            # 4 prefix + 3 trainable assistant tokens
            assert len(messages) >= 2
            return [1, 2, 3, 4, 20, 21, 22], [0, 0, 0, 0, 1, 1, 1]

        def get_response_lengths(self, loss_masks):
            return [len(mask[mask.index(1) :]) if 1 in mask else 0 for mask in loss_masks]

    monkeypatch.setattr(mod, "MultiTurnLossMaskGenerator", _StubMaskGen)

    sample = Sample(prompt="p", index=0, group_index=0, metadata={})
    checkpoint = SimpleNamespace(
        checkpoint_id="main-0-test",
        chat_messages=[{"role": "user", "content": "fix"}],
        tools_schema=None,
        prompt_ids=[1, 2, 3, 4],
        prompt_sha256=prompt_ids_sha256([1, 2, 3, 4]),
        tools_sha256=canonical_sha256(None),
    )
    rows = [
        {
            "turn_index": 0,
            "response": {"message": {"role": "assistant", "content": "patch"}},
            "prompt": {"messages": checkpoint.chat_messages},
        }
    ]
    out = encode_teacher_continuation_sample(
        sample=sample,
        tokenizer=_FakeTokenizer(),
        checkpoint=checkpoint,
        sft_rows=_bind_v2_rows(checkpoint, rows),
        loss_mask_type="qwen3_5",
        metadata={"step_group_key": "g:0:1"},
    )
    assert out.metadata["sample_kind"] == "teacher_sft"
    assert out.tokens[:4] == [1, 2, 3, 4]
    assert out.response_length == 3
    assert out.loss_mask == [1, 1, 1]
    assert out.rollout_log_probs == [0.0, 0.0, 0.0]
    assert out.metadata["step_group_key"] == "g:0:1"
    assert seen["tokenizer_type"] == "qwen3_5"


def test_encode_qwen35_preserves_checkpoint_across_bpe_boundary_merge():
    tokenizer = _BoundaryMergingTokenizer()
    checkpoint_messages = [{"role": "user", "content": "fix"}]
    checkpoint_text = tokenizer.apply_chat_template(
        checkpoint_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(checkpoint_text, add_special_tokens=False)["input_ids"]
    checkpoint = SimpleNamespace(
        checkpoint_id="main-boundary-merge",
        chat_messages=checkpoint_messages,
        tools_schema=None,
        prompt_ids=prompt_ids,
        prompt_sha256=prompt_ids_sha256(prompt_ids),
        tools_sha256=canonical_sha256(None),
    )
    rows = _bind_v2_rows(
        checkpoint,
        [
            {
                "turn_index": 0,
                "prompt": {"messages": checkpoint_messages},
                "response": {"message": {"role": "assistant", "content": "reason"}},
            }
        ],
    )

    out = encode_teacher_continuation_sample(
        sample=Sample(prompt="p", index=0, group_index=0, metadata={}),
        tokenizer=tokenizer,
        checkpoint=checkpoint,
        sft_rows=rows,
        loss_mask_type="qwen3_5",
    )

    assert out.tokens[: len(prompt_ids)] == prompt_ids
    assert out.metadata["teacher_checkpoint_suffix_tokenization"] is True
    assert sum(out.loss_mask) > 0
    assert tokenizer.decode(out.tokens) == tokenizer.apply_chat_template(
        checkpoint_messages + [rows[0]["response"]["message"]],
        tokenize=False,
    )


def test_encode_rejects_empty_trainable_suffix(monkeypatch):
    from examples.claudecode_ags.step_reconstruct import teacher_segments as mod

    class _StubMaskGen:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def get_loss_mask(self, messages, tools=None):
            del messages, tools
            return [1, 2, 3], [0, 0, 0]

        def get_response_lengths(self, loss_masks):
            return [0 for _ in loss_masks]

    monkeypatch.setattr(mod, "MultiTurnLossMaskGenerator", _StubMaskGen)
    sample = Sample(prompt="p", index=0, group_index=0, metadata={})
    checkpoint = SimpleNamespace(
        checkpoint_id="main-0-empty",
        chat_messages=[{"role": "user", "content": "fix"}],
        tools_schema=None,
        prompt_ids=[1, 2, 3],
        prompt_sha256=prompt_ids_sha256([1, 2, 3]),
        tools_sha256=canonical_sha256(None),
    )
    rows = [
        {
            "turn_index": 0,
            "response": {"message": {"role": "assistant", "content": "x"}},
            "prompt": {"messages": checkpoint.chat_messages},
        }
    ]
    with pytest.raises(ValueError, match="no_trainable_tokens"):
        encode_teacher_continuation_sample(
            sample=sample,
            tokenizer=_FakeTokenizer(),
            checkpoint=checkpoint,
            sft_rows=_bind_v2_rows(checkpoint, rows),
        )


def test_rebuild_rejects_prompt_history_drift():
    checkpoint_messages = [{"role": "user", "content": "fix"}]
    rows = [
        {
            "turn_index": 0,
            "prompt": {"messages": [{"role": "user", "content": "resume handshake"}]},
            "response": {"message": {"role": "assistant", "content": "patch"}},
        }
    ]
    with pytest.raises(TeacherContextIntegrityError, match="prompt_history_mismatch"):
        rebuild_messages_from_teacher_turns(checkpoint_messages, rows)


def test_load_rejects_duplicate_turn_indices(tmp_path):
    path = tmp_path / "mixed.sft_turns.jsonl"
    path.write_text(
        '\n'.join(
            [
                '{"turn_index":0,"version":2}',
                '{"turn_index":0,"version":2}',
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(TeacherContextIntegrityError, match="non_contiguous_turns"):
        load_sft_turn_rows(str(path))


def test_load_binds_rows_to_unique_attempt_filename(tmp_path):
    # Build the production filename shape: <session-prefix>.<attempt-id>.sft_turns.jsonl.
    path = tmp_path / f"{'a' * 16}.{'b' * 16}.sft_turns.jsonl"
    path.write_text(
        json.dumps(
            {
                "turn_index": 0,
                "session_id_sha256": "a" * 64,
                "conditioning": {"attempt_id": "c" * 16},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(TeacherContextIntegrityError, match="attempt_path_id_mismatch"):
        load_sft_turn_rows(str(path))


def test_encode_rejects_old_v1_jsonl(monkeypatch):
    from examples.claudecode_ags.step_reconstruct import teacher_segments as mod

    class _StubMaskGen:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    monkeypatch.setattr(mod, "MultiTurnLossMaskGenerator", _StubMaskGen)
    checkpoint = SimpleNamespace(
        checkpoint_id="main-v1",
        chat_messages=[{"role": "user", "content": "fix"}],
        tools_schema=None,
        prompt_ids=[1],
        prompt_sha256=prompt_ids_sha256([1]),
        tools_sha256=canonical_sha256(None),
    )
    rows = [
        {
            "version": 1,
            "turn_index": 0,
            "prompt": {"messages": checkpoint.chat_messages},
            "response": {"message": {"role": "assistant", "content": "patch"}},
        }
    ]
    with pytest.raises(TeacherContextIntegrityError, match="unsupported_log_version"):
        encode_teacher_continuation_sample(
            sample=Sample(prompt="p", index=0, group_index=0, metadata={}),
            tokenizer=_FakeTokenizer(),
            checkpoint=checkpoint,
            sft_rows=rows,
        )


def test_encode_rejects_tools_schema_mismatch(monkeypatch):
    from examples.claudecode_ags.step_reconstruct import teacher_segments as mod

    class _StubMaskGen:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    monkeypatch.setattr(mod, "MultiTurnLossMaskGenerator", _StubMaskGen)
    checkpoint = SimpleNamespace(
        checkpoint_id="main-tools",
        chat_messages=[{"role": "user", "content": "fix"}],
        tools_schema=None,
        prompt_ids=[1],
        prompt_sha256=prompt_ids_sha256([1]),
        tools_sha256=canonical_sha256(None),
    )
    rows = _bind_v2_rows(
        checkpoint,
        [
            {
                "turn_index": 0,
                "prompt": {"messages": checkpoint.chat_messages},
                "response": {"message": {"role": "assistant", "content": "patch"}},
            }
        ],
    )
    rows[0]["prompt"]["tools_schema"] = []
    rows[0]["prompt"]["tools_sha256"] = canonical_sha256([])
    with pytest.raises(TeacherContextIntegrityError, match="tools_schema_mismatch"):
        encode_teacher_continuation_sample(
            sample=Sample(prompt="p", index=0, group_index=0, metadata={}),
            tokenizer=_FakeTokenizer(),
            checkpoint=checkpoint,
            sft_rows=rows,
        )


def test_encode_rejects_mixed_attempts(monkeypatch):
    from examples.claudecode_ags.step_reconstruct import teacher_segments as mod

    class _StubMaskGen:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    monkeypatch.setattr(mod, "MultiTurnLossMaskGenerator", _StubMaskGen)
    checkpoint = SimpleNamespace(
        checkpoint_id="main-attempt",
        chat_messages=[{"role": "user", "content": "fix"}],
        tools_schema=None,
        prompt_ids=[1],
        prompt_sha256=prompt_ids_sha256([1]),
        tools_sha256=canonical_sha256(None),
    )
    rows = _bind_v2_rows(
        checkpoint,
        [
            {
                "turn_index": index,
                "prompt": {"messages": checkpoint.chat_messages},
                "response": {"message": {"role": "assistant", "content": "patch"}},
            }
            for index in range(2)
        ],
    )
    rows[1]["conditioning"] = {**rows[1]["conditioning"], "attempt_id": "attempt-b"}
    with pytest.raises(TeacherContextIntegrityError, match="mixed_attempts"):
        encode_teacher_continuation_sample(
            sample=Sample(prompt="p", index=0, group_index=0, metadata={}),
            tokenizer=_FakeTokenizer(),
            checkpoint=checkpoint,
            sft_rows=rows,
        )


def test_encode_rejects_token_prefix_mismatch(monkeypatch):
    from examples.claudecode_ags.step_reconstruct import teacher_segments as mod

    class _StubMaskGen:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def get_loss_mask(self, messages, tools=None):
            del messages, tools
            return [9, 2, 20], [0, 0, 1]

    monkeypatch.setattr(mod, "MultiTurnLossMaskGenerator", _StubMaskGen)
    checkpoint = SimpleNamespace(
        checkpoint_id="main-token",
        chat_messages=[{"role": "user", "content": "fix"}],
        tools_schema=None,
        prompt_ids=[1, 2],
        prompt_sha256=prompt_ids_sha256([1, 2]),
        tools_sha256=canonical_sha256(None),
    )
    rows = _bind_v2_rows(
        checkpoint,
        [
            {
                "turn_index": 0,
                "prompt": {"messages": checkpoint.chat_messages},
                "response": {"message": {"role": "assistant", "content": "patch"}},
            }
        ],
    )
    with pytest.raises(TeacherContextIntegrityError, match="token_prefix_mismatch"):
        encode_teacher_continuation_sample(
            sample=Sample(prompt="p", index=0, group_index=0, metadata={}),
            tokenizer=_FakeTokenizer(),
            checkpoint=checkpoint,
            sft_rows=rows,
        )
