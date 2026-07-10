"""Mocked wiring test: pluggable binary reward + fan_out in generate()."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from examples.claudecode_ags.swe_eval.base import EvalResult
from slime.agent.segment_trajectory import TokenSegment
from slime.utils.misc import SingletonMeta
from slime.utils.types import Sample
from tests.claudecode_ags.fake_sandbox import FakeSandbox


class _Tok:
    def decode(self, ids, skip_special_tokens=False):
        return "x" * len(ids)


def _seg(kind: str, n_resp: int = 2) -> TokenSegment:
    return TokenSegment(
        prompt_ids=[1],
        response_ids=list(range(10, 10 + n_resp)),
        loss_mask=[1] * n_resp,
        rollout_log_probs=[0.0] * n_resp,
        metadata={"segment_kind": kind},
    )


@pytest.fixture(autouse=True)
def _clear_adapter_singleton():
    SingletonMeta._instances = {}
    yield
    SingletonMeta._instances = {}


def test_filter_cc_env_skips_swe_aliases():
    from examples.claudecode_ags.generate import _filter_cc_env

    filtered = _filter_cc_env(
        {
            "ANTHROPIC_BASE_URL": "http://x",
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "8192",
            "BASH_MAX_TIMEOUT_MS": "900000",
            "API_TIMEOUT_MS": "1200000",
            "SWE_AGENT_TIME_BUDGET_SEC": "1800",
            "PATH": "/usr/bin",
        }
    )
    assert "ANTHROPIC_BASE_URL" in filtered
    assert "CLAUDE_CODE_MAX_OUTPUT_TOKENS" in filtered
    assert "BASH_MAX_TIMEOUT_MS" in filtered
    assert "API_TIMEOUT_MS" in filtered
    assert "SWE_AGENT_TIME_BUDGET_SEC" not in filtered
    assert "PATH" not in filtered


def test_custom_cc_reward_arg_default():
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "slime" / "utils" / "arguments.py"
    text = src.read_text()
    assert "--custom-cc-reward-function-path" in text
    assert 'default="examples.claudecode_ags.rewards.default.compose"' in text
    # Placed near the custom generate hook.
    gen_idx = text.index("--custom-generate-function-path")
    reward_idx = text.index("--custom-cc-reward-function-path")
    assert reward_idx > gen_idx
    assert reward_idx - gen_idx < 800


def test_generate_wires_binary_reward_and_fan_out(monkeypatch):
    import examples.claudecode_ags.generate as gen

    SingletonMeta._instances = {}

    fake_adapter = MagicMock()
    fake_adapter.open_session = MagicMock()
    fake_adapter.finish_session = AsyncMock(return_value=[_seg("wipe"), _seg("final", n_resp=1)])
    fake_adapter.shutdown_session = AsyncMock()

    fake_state = SimpleNamespace(
        tokenizer=_Tok(),
        max_context_len=0,
        adapter=fake_adapter,
        adapter_url="http://adapter:18001",
    )
    monkeypatch.setattr(gen, "_AdapterService", lambda _args: fake_state)

    sandboxes: list[FakeSandbox] = []

    def _fake_make_sandbox(_image: str, **_kwargs):
        sb = FakeSandbox()
        sandboxes.append(sb)
        return sb

    monkeypatch.setattr(gen, "make_sandbox", _fake_make_sandbox)
    monkeypatch.setattr(gen.agent_runtime, "prepare_workspace", AsyncMock())
    monkeypatch.setattr(gen.agent_runtime, "install_toolchain", AsyncMock())
    monkeypatch.setattr(
        gen.agent_runtime,
        "run_claude",
        AsyncMock(return_value={"exit_code": 0}),
    )
    monkeypatch.setattr(gen.agent_runtime, "git_diff", AsyncMock(return_value="diff --git a/x b/x\n"))
    monkeypatch.setattr(
        gen.simple_cmd,
        "evaluate",
        AsyncMock(return_value=EvalResult(resolved=True, applied_cleanly=True, details={"exit_code": 0})),
    )

    sample = Sample(
        index=7,
        group_index=1,
        prompt="fix it",
        metadata={
            "image": "img:tag",
            "workdir": "/testbed",
            "problem_statement": "bug",
            "eval_cmd": "pytest -q",
            "instance_id": "repo__1",
        },
    )
    args = SimpleNamespace(
        custom_cc_reward_function_path="examples.claudecode_ags.rewards.default.compose",
        hf_checkpoint="unused",
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
    )

    out = asyncio.run(gen.generate(args, sample, {"temperature": 0.0}, evaluation=False))

    assert len(out) == 2
    assert out[0].reward == 0.5
    assert out[1].reward == 0.5
    assert out[0].rollout_id == 7
    assert out[1].rollout_id == 7
    assert out[0].metadata["reward_details"]["mode"] == "binary"
    assert out[0].metadata["grading_solved"] is True
    assert out[0].metadata["base_eval"]["resolved"] is True

    gen.agent_runtime.run_claude.assert_awaited()
    call_kwargs = gen.agent_runtime.run_claude.await_args.kwargs
    assert call_kwargs["env"]["ANTHROPIC_BASE_URL"] == "http://adapter:18001"
    assert call_kwargs["env"]["ANTHROPIC_AUTH_TOKEN"] == sample.session_id
    assert not any(k.startswith("SWE_") for k in call_kwargs["env"])

    fake_adapter.finish_session.assert_awaited()
    assert len(sandboxes) == 2  # agent sandbox + fresh eval sandbox
